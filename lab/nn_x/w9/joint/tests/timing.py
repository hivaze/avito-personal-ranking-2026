"""Step-time / peak-memory measurement of the JOINT arm vs the base on the SAME real training batches (one GPU, train.py's
optimiser setup, bf16 autocast, default SDPA kernels as in training). Prints ms/step, steps/s, peak GB, the eval cost
(val predict + 30K holdout eval) and the projected 10-epoch (98,350 steps + 40 evals) and 2-epoch (19,670 steps + 20 evals) wall
times alone on the GPU.  usage: CUDA_VISIBLE_DEVICES=<g> OMP_NUM_THREADS=12 python tests/timing.py [n_steps=100] [layers=4,3,2]"""
import math
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
W9 = os.path.dirname(HERE)
sys.path.insert(0, W9)
from common import eval_subset, feature_stats, predict  # noqa: E402
from data import SplitData  # noqa: E402
from model import N_DENSE, Ranker, listwise_loss  # noqa: E402
from train import XT_PRESETS, XT_REL, XT_RELONLY  # noqa: E402

NS = int(sys.argv[1]) if len(sys.argv) > 1 else 100
LAYERS = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "4,3,2").split(",")]
dev = torch.device("cuda:0")
torch.backends.cuda.matmul.allow_tf32 = True
torch.manual_seed(0)
tr = SplitData("train", dev)
va = SplitData("val", dev)
xt = XT_PRESETS["cvnd_pre2s"]
for sd in (tr, va):
    sd.attach_xt(xt)
xt_rel = [xt.index(n) for n in XT_REL["cvnd_pre2s"]]
xt_idcols = [i for i, n in enumerate(xt) if n.startswith(("seq_cv", "it_hist"))]
xt_relonly = [xt.index(n) for n in XT_RELONLY["cvnd_pre2s"]]
n_dense = N_DENSE + 2 + len(xt) + 2 * len(xt_rel) - len(xt_relonly)
pr = np.load("<repo>/eda/labels_position/out/train_prandom.npz")
prand_np = np.nan_to_num(pr["p_random"].astype(np.float32), nan=0.0)
prand_np[tr.M_np < 50] = 0.0
prand = torch.from_numpy(prand_np).to(dev)
tr.dom = torch.stack([1 - prand, prand, torch.zeros_like(prand)], 1)
cand = np.where((prand_np > 0.9) & (tr.M_np >= 50))[0]
hold = np.sort(np.random.default_rng(12345).choice(cand, 30000, replace=False))
hold_rows = np.concatenate([np.arange(o, o + m) for o, m in zip(tr.cand_off_np[hold], tr.M_np[hold])])
rng = np.random.default_rng(1)
feeds = np.setdiff1d(np.sort(rng.choice(tr.n, 400000, replace=False)), hold)
batches = tr.batches(feeds, 32768, 512, shuffle=True, rng=rng)[:NS + 10]
print(f"[timing] {len(batches)} batches; feeds/step mean {np.mean([len(b) for b in batches]):.0f}; M-pad values {sorted(set(int(-(-tr.M_np[b].max() // 16) * 16) for b in batches))[:12]}...", flush=True)


def make(joint_layers):
    kw = dict(blocks=0, hist_layers=0, joint_layers=joint_layers) if joint_layers else dict(blocks=2, hist_layers=1)
    torch.manual_seed(0)
    m = Ranker(d=128, heads=4, drop=0.1, use_hist=True, use_set=True, use_pos_tower=False, n_dense=n_dense, aux_pos=True, ff=2,
               sim_bias=1, domain=True, xt_names=xt, xt_rel=xt_rel, xt_drop=0.0, xt_idcols=xt_idcols, xt_relonly=xt_relonly,
               attn_gate=True, split_kv=True, retr_m=16, **kw).to(dev)
    m.eval()
    mean, std = feature_stats(tr, feeds, dense_fn=m.dense)
    m.train()
    m.set_norm(mean, std)
    muon_names = {f"{mn}.weight" for mn, mod in m.named_modules() if isinstance(mod, torch.nn.Linear) and min(mod.weight.shape) > 1}
    rest = [(n, p) for n, p in m.named_parameters() if n not in muon_names]
    decay = [p for n, p in rest if p.ndim >= 2 and "emb" not in n]
    no_decay = [p for n, p in rest if not (p.ndim >= 2 and "emb" not in n)]
    opts = [torch.optim.AdamW([{"params": decay, "weight_decay": 0.02}, {"params": no_decay, "weight_decay": 0.0}], lr=1e-3, betas=(0.9, 0.98)),
            torch.optim.Muon([p for n, p in m.named_parameters() if n in muon_names], lr=1e-3, weight_decay=0.02, momentum=0.95,
                             nesterov=True, adjust_lr_fn="match_rms_adamw")]
    return m, opts


def step(m, opts, fb):
    b = tr.gather(fb, with_pos=True)
    X = m.dense(b)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        lg = m(b, X, use_pos=True)
    y, msk = b["y"], b["cmask"]
    loss = listwise_loss(lg, y, msk, None, norm="feed", denom=512)
    tgt = torch.softmax((-b["pos"].float() / 10.0).masked_fill(~msk, -1e4), 1) * msk
    lp = torch.log_softmax(m.aux_out.masked_fill(~msk, -1e4), 1)
    fw_aux = 1 - prand[b["fidx"]]
    loss = loss + 0.5 * (-(tgt * lp).sum(1) * fw_aux).sum() / fw_aux.sum().clamp_min(1e-6)
    for o in opts:
        o.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    for o in opts:
        o.step()
    return float(loss)


res = {}
for L in [0] + LAYERS:
    name = "base" if L == 0 else f"joint{L}"
    m, opts = make(L)
    npar = sum(p.numel() for p in m.parameters()) / 1e6
    for fb in batches[:10]:  # warm-up (kernel selection per shape)
        step(m, opts, fb)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for fb in batches[10:10 + NS]:
        step(m, opts, fb)
    torch.cuda.synchronize()
    dt = (time.time() - t0) / NS
    peak = torch.cuda.max_memory_allocated() / 2 ** 30
    t1 = time.time()
    pv = predict(m, va)
    ndh = eval_subset(m, tr, hold, hold_rows)
    torch.cuda.synchronize()
    te = time.time() - t1
    proj10 = (98350 * dt + 40 * te) / 60
    proj2 = (19670 * dt + 20 * te) / 60
    res[name] = dict(ms=dt * 1000, peak=peak, eval_s=te, proj10=proj10, proj2=proj2, params=npar)
    print(f"[timing] {name:7s} params {npar:.3f}M | {dt * 1000:7.1f} ms/step ({1 / dt:5.1f} steps/s) | peak {peak:5.1f} GB | eval (val + 30K holdout) "
          f"{te:5.1f} s | projected alone: 10 epochs {proj10:5.0f} min, 2 epochs {proj2:5.0f} min", flush=True)
    del m, opts
    torch.cuda.empty_cache()
b = res["base"]
for k, v in res.items():
    if k != "base":
        print(f"[timing] {k}: {v['ms'] / b['ms']:.2f}x the base step time, {v['proj10'] / b['proj10']:.2f}x the projected 10-epoch wall time", flush=True)
