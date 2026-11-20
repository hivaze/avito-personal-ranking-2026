"""GPU step-time measurement of PSEUDO-DIRECT (tests/test_pseudo.py G2): the real train split on ONE GPU (CUDA_VISIBLE_DEVICES set by
the caller; GPU N for this agent), the stage-E recipe's model (d128, retr16, attn_gate, Muon split k/v, domain, user0, sim_bias, K9
aux head, cvnd_pre2s inputs), the same N training batches (seeded) run twice: --pseudo_ratio 0 (the real step: dense + forward + loss +
aux + backward + clip + AdamW/Muon step) and --pseudo_ratio 1 (the same plus the pseudo build / forward / loss). Each mode starts from
the same initial weights; 5 warm-up steps are excluded; torch.cuda.synchronize around every step. Also records the pseudo-feed
statistics per batch (count, mean M', positives, mean H', t, g, draws dropped, eligible share) and the peak GPU memory.
  usage: CUDA_VISIBLE_DEVICES=N OMP_NUM_THREADS=12 python tests/time_steps.py [n_steps=40]  (from nn_x/w11/pseudo) -> logs/time_steps.json
"""
import copy
import json
import math
import os
import sys
import time
import types

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.dirname(HERE)
sys.path.insert(0, D)
assert os.environ.get("CUDA_VISIBLE_DEVICES", "") in tuple(str(i) for i in range(8)), "set CUDA_VISIBLE_DEVICES to one GPU id"
import numpy as np  # noqa: E402
import torch  # noqa: E402

from common import ROOT, feature_stats  # noqa: E402
from data import SplitData  # noqa: E402
from model import N_DENSE, Ranker, listwise_loss  # noqa: E402
import cd_arms11 as C11  # noqa: E402
import train as T  # noqa: E402 (XT presets)

N_STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 40
WARM = 5
dev = torch.device("cuda:0")
torch.backends.cuda.matmul.allow_tf32 = True
torch.manual_seed(951)
np.random.seed(951)
t0 = time.time()
tr = SplitData("train", dev, max_hist=256)
xt_names = T.XT_PRESETS["cvnd_pre2s"]
tr.attach_xt(xt_names)
xt_rel = [xt_names.index(n) for n in T.XT_REL["cvnd_pre2s"]]
xt_idcols = [i for i, n in enumerate(xt_names) if n.startswith(("seq_cv", "it_hist"))]
xt_relonly = [xt_names.index(n) for n in T.XT_RELONLY["cvnd_pre2s"]]
pr = np.load(ROOT / "eda/labels_position/out/train_prandom.npz")
prand_np = np.nan_to_num(pr["p_random"].astype(np.float32), nan=0.0)
prand_np[tr.M_np < 50] = 0.0
prand = torch.from_numpy(prand_np).to(dev)
cand = np.where((prand_np > 0.9) & (tr.M_np >= 50))[0]
hold = np.sort(np.random.default_rng(12345).choice(cand, min(30000, len(cand)), replace=False))  # exactly train.py's --holdout 30000
feeds = np.setdiff1d(np.arange(tr.n), hold)
rng = np.random.default_rng(951)
model = Ranker(d=128, heads=4, hist_layers=1, blocks=2, drop=0.1, use_pos_tower=False, aux_pos=True, ff=2, sim_bias=1,
               n_dense=N_DENSE + 2 + len(xt_names) + 2 * len(xt_rel) - len(xt_relonly), domain=True, xt_names=xt_names, xt_rel=xt_rel,
               xt_drop=0.0, xt_idcols=xt_idcols, xt_relonly=xt_relonly, attn_gate=True, split_kv=True, retr_m=16).to(dev)
model.eval()
mean, std = feature_stats(tr, feeds, dense_fn=model.dense)
model.train()
model.set_norm(mean, std)
tr.dom = torch.stack([1 - prand, prand, torch.zeros_like(prand)], 1)
state0 = copy.deepcopy(model.state_dict())
batches = tr.batches(feeds, 32768, 512, shuffle=True, rng=rng)[:N_STEPS]
print(f"[time] data + model ready in {time.time() - t0:.0f}s; {len(batches)} batches; B {np.mean([len(b) for b in batches]):.1f} mean", flush=True)
a = types.SimpleNamespace(pseudo_ratio=1.0, pseudo_w=0.3, pseudo_gap_min=16, pseudo_gap_max=64, pseudo_tmax=63, max_hist=256, pseudo_noset=0,
                          pseudo_head=1, seed=951, loss="softmax", loss_norm="feed", max_feeds=512, prand_base=-1.0, aux_pos=0.5, aux_tau=10.0,
                          aux_norm="batch", aux_norm_c=1.695, lr=2e-3, wd=0.02, intent=0.0, trunc_off=0.0, trunc_lifts="", rdrop_cd=0.0, sam_cd=0.0,
                          val_mix=0.0, hzcd=False, hcap="", arch="tf", sched="wsd")


def make_opts(model):
    muon_names = {f"{mn}.weight" for mn, mod in model.named_modules() if isinstance(mod, torch.nn.Linear) and min(mod.weight.shape) > 1}
    rest = [(n, p) for n, p in model.named_parameters() if n not in muon_names]
    decay = [p for n, p in rest if p.ndim >= 2 and "emb" not in n]
    no_decay = [p for n, p in rest if not (p.ndim >= 2 and "emb" not in n)]
    opts = [torch.optim.AdamW([{"params": decay, "weight_decay": a.wd}, {"params": no_decay, "weight_decay": 0.0}], lr=a.lr, betas=(0.9, 0.98))]
    P_muon = [p for n, p in model.named_parameters() if n in muon_names]
    opts.append(torch.optim.Muon(P_muon, lr=a.lr, weight_decay=a.wd, momentum=0.95, nesterov=True, adjust_lr_fn="match_rms_adamw"))
    return opts


def run(mode):
    model.load_state_dict(state0)
    model.train()
    opts = make_opts(model)
    pseudo = None
    if mode == "on":
        ok_np = C11.holdout_user_mask(tr, hold)
        pseudo = C11.PseudoTrainer(a, model, tr, torch.from_numpy(ok_np), opts, t_end=10 ** 9, head_seed=1)
    times, st = [], dict(P=[], B=[], M=[], H=[], t=[], g=[], drop_tgt=[], drop_neg=[], blocked=[], eligible_rows=[])
    torch.cuda.reset_peak_memory_stats()
    for i, fb in enumerate(batches):
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        for opt in opts:
            for g in opt.param_groups:
                g["lr"] = a.lr
        b = tr.gather(fb, with_pos=True)
        X = model.dense(b)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = model(b, X, use_pos=True)
        y, m = b["y"], b["cmask"]
        loss = listwise_loss(lg, y, m, None, norm=a.loss_norm, denom=a.max_feeds)
        tgt = torch.softmax((-b["pos"].float() / a.aux_tau).masked_fill(~m, -1e4), 1) * m
        lp = torch.log_softmax(model.aux_out.masked_fill(~m, -1e4), 1)
        fw_aux = 1 - prand[b["fidx"]]
        loss = loss + a.aux_pos * (-(tgt * lp).sum(1) * fw_aux).sum() / fw_aux.sum().clamp_min(1e-6)
        if pseudo is not None:
            loss = pseudo.combine(model, a, b, loss, prand, None)
        for opt in opts:
            opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for opt in opts:
            opt.step()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t1) * 1000)
        if pseudo is not None:
            v = pseudo.acc.cpu().numpy()
            st["P"].append(int(pseudo.last_P))
            st["B"].append(len(fb))
            if pseudo.last_P:
                st["M"].append(float(v[2] / max(v[1], 1)))
            st["drop_tgt"].append(float(v[6]))
            st["drop_neg"].append(float(v[7]))
            st["blocked"].append(float(v[8]))
            cemb = tr.c_emb[b["ci"]].long()
            el = C11.eligible_targets(b, cemb, a.pseudo_tmax) & pseudo.src_ok[b["fidx"]][:, None]
            st["eligible_rows"].append(float(el.any(1).float().mean()))
            pseudo.acc.zero_()
            pseudo.n_draws = 0
            pseudo.n_steps_seen = 0
    peak = torch.cuda.max_memory_allocated() / 2 ** 30
    return np.array(times), st, peak


out = {}
for mode in ("off", "on", "off", "on"):  # interleaved: two measurements per mode (GPU clock drift)
    times, st, peak = run(mode)
    key = mode if mode not in out else mode + "2"
    out[key] = dict(ms_median=float(np.median(times[WARM:])), ms_mean=float(times[WARM:].mean()), ms_p90=float(np.quantile(times[WARM:], 0.9)), peak_gb=peak, n=len(times) - WARM)
    print(f"[time] {key}: median {out[key]['ms_median']:.1f} ms/step, mean {out[key]['ms_mean']:.1f}, p90 {out[key]['ms_p90']:.1f}, peak mem {peak:.1f} GB", flush=True)
    if mode == "on" and key == "on":
        Pn, Bn = np.array(st["P"]), np.array(st["B"])
        out["pseudo"] = dict(per_batch_mean=float(Pn.mean()), B_mean=float(Bn.mean()), share_of_rows=float(Pn.sum() / Bn.sum()), positives_per_feed=1,
                             total=int(Pn.sum()), M_mean=float(np.mean(st["M"])) if st["M"] else float("nan"),
                             drop_tgt_share=float(np.sum(st["drop_tgt"]) / Bn.sum()), drop_neg_share=float(np.sum(st["drop_neg"]) / Bn.sum()),
                             blocked_share=float(np.sum(st["blocked"]) / Bn.sum()), eligible_rows_share=float(np.mean(st["eligible_rows"])))
# detailed per-feed stats of one pass through the batches (H', t, g, M') with a fresh trainer (build only; no training)
ok_np = C11.holdout_user_mask(tr, hold)
pt = C11.PseudoTrainer(a, model, tr, torch.from_numpy(ok_np), make_opts(model), t_end=10 ** 9, head_seed=1)
agg = dict(M=[], H=[], t=[], g=[], hsrc=[])
for fb in batches:
    b = tr.gather(fb, with_pos=True)
    bp, _ = pt.build(b)
    if bp is None:
        continue
    vm = bp["pseudo_valid"].cpu().numpy()
    agg["M"].append(bp["M"].cpu().numpy()[vm])
    agg["H"].append(bp["H"].cpu().numpy()[vm])
    agg["t"].append(bp["pseudo_t"].cpu().numpy()[vm])
    agg["g"].append(bp["pseudo_g"].cpu().numpy()[vm])
    agg["hsrc"].append(b["H"][bp["pseudo_src"]].cpu().numpy()[vm])
cat = {k: np.concatenate(v) for k, v in agg.items()}
out["pseudo"].update(M_mean=float(cat["M"].mean()), M_median=float(np.median(cat["M"])), H_mean=float(cat["H"].mean()), H_source_mean=float(cat["hsrc"].mean()),
                     t_mean=float(cat["t"].mean()), g_mean=float(cat["g"].mean()), t_hist=np.bincount(cat["t"], minlength=64).tolist(),
                     g_min=int(cat["g"].min()), g_max=int(cat["g"].max()), n=int(len(cat["M"])))
# PRIMARY = the second (warm) pass of each mode: the first pass pays the kernel-selection warm-up of every new (B, M, K) shape (data.py),
# which a 98K-step run amortises over ~20 shapes; the cold pass is reported as information
out["ratio"] = float(out["on2"]["ms_median"] / out["off2"]["ms_median"])
out["ratio_cold_first_pass"] = float(out["on"]["ms_median"] / out["off"]["ms_median"])
out["off_ms_median"] = float(out["off2"]["ms_median"])
out["on_ms_median"] = float(out["on2"]["ms_median"])
out["n_steps"] = N_STEPS
out["gpu"] = os.environ["CUDA_VISIBLE_DEVICES"]
out["real_batch"] = dict(B_mean=float(np.mean([len(b) for b in batches])), M_mean=float(np.mean([tr.M_np[b].mean() for b in batches])))
print(f"[time] RATIO on/off (warm pass) = {out['ratio']:.3f} ({out['on_ms_median']:.1f} / {out['off_ms_median']:.1f} ms/step; cold first pass {out['ratio_cold_first_pass']:.2f}); pseudo-feeds per batch "
      f"{out['pseudo']['per_batch_mean']:.1f} of {out['pseudo']['B_mean']:.1f} rows ({out['pseudo']['share_of_rows']:.2f}); M' mean {out['pseudo']['M_mean']:.1f} "
      f"(real M {out['real_batch']['M_mean']:.1f}); H' mean {out['pseudo']['H_mean']:.1f} (source H {out['pseudo']['H_source_mean']:.1f}); t mean "
      f"{out['pseudo']['t_mean']:.1f}; g mean {out['pseudo']['g_mean']:.1f}; dropped: no target {out['pseudo']['drop_tgt_share']:.3f}, no negative "
      f"{out['pseudo']['drop_neg_share']:.3f}; blocked (holdout users) {out['pseudo']['blocked_share']:.3f}; rows with an eligible target "
      f"{out['pseudo']['eligible_rows_share']:.3f}", flush=True)
json.dump(out, open(f"{D}/logs/time_steps.json", "w"), indent=1)
