"""CPU tests of the w12/moe screen (2026-11-19): copy lab/nn_x/w12/moe/train.py (= lab/nn_x/w5/train.py md5 7c1e3588 + the arm flags)
and model_moe.py (= lab/nn_x/w5/model.py md5 2464dbb6 + the routed modules), helpers moe_arms.py.

  cd <repo> && CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 PYTHONDONTWRITEBYTECODE=1 \
      .venv/bin/python lab/nn_x/w12/moe/tests/test_moe_cpu.py        -> lab/nn_x/w12/moe/logs/test_moe_cpu.log (+ .json)

T0 static
  (a) the w5 originals are the designed versions (md5 7c1e3588 / 2464dbb6);
  (b) AST strip proof: train.py and model_moe.py minus exactly the expected moe additions (tests/strip_moe.py) == the w5 originals
      (docstrings dropped); the removal lists are asserted in full.
T1 flag-off identity (CPU, fixed seed)
  (a) model level, synthetic batch with padding, cfg of the stored student w5_e10_kd05w192_s951 (and the arm-C cfg blocks 3 / ff 1):
      w5 model.Ranker vs model_moe.Ranker built under the same seed: identical state dict (keys and tensors); bit-identical train-mode
      logits (same dropout seed), K9 aux output, loss and every gradient; bit-identical eval logits; identical cfg.
  (b) trainer level (tests/e2e_moe_cpu.py, miniature split, the E recipe at mini scale with the KD teacher, kd_w 0.5, d 192, cooldown +
      pool batches, evals every 10 steps, 50 steps, 1 thread): lab/nn_x/w5/train.py vs this train.py with every flag off: identical
      training-batch sequence, train-mode logits and loss values; identical logged metric lines; bit-identical selected / @last / @10 /
      @20 checkpoints, val + holdout preds (selected and @last), cliplog arrays; stored args == w5 args + the 9 moe defaults; no moe
      line; checkpoint cfg == w5 cfg.
T2 arms (model level, CPU)
  shapes; parameter counts as designed (A 2,714,816 total / 2,272,448 active, B 2,720,960 / 2,426,048, C 2,270,340); every reference
  tensor identical and new keys only under .moe. / .attn.vo.; eval logits at init == reference bit for bit (A, B), train-mode step-0
  logits too for B (no extra RNG draw); the trainer's optimiser split (routers out of Muon into AdamW no-decay, stacks only in
  MuonBatched, every parameter in exactly one group); gradient reaches every expert slice (w2 / wv / wo at step 0; w1 / w2 / wv / wo /
  every router row after one optimiser step); MuonBatched == torch.optim.Muon per slice; masked nanquantile == quantile on the real
  tokens; the quantile balancer moves skewed loads (low-diversity drifting stream) toward uniform (and eta 0 does not); checkpoint round
  trip; no NaN.
T3 50-step CPU training loop per arm through the trainer (e2e wrapper, miniature split): A, B, C complete; finite losses; A / B log a
  'w12/moe step' diagnostics line at every eval and store it in the json history; the trained checkpoint has moved rbias / expert stacks
  (mechanism live) and keeps the arm cfg; B's step-0 train logits equal the flag-off run's; C has 60 Muon matrices.
"""
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time

sys.dont_write_bytecode = True
assert os.environ.get("CUDA_VISIBLE_DEVICES", "x") == "", "run with CUDA_VISIBLE_DEVICES=''"
NT = min(12, int(os.environ.get("OMP_NUM_THREADS", "12")))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

torch.set_num_threads(NT)
HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.dirname(HERE)  # lab/nn_x/w12/moe
ROOT = "<repo>"
W5 = f"{ROOT}/lab/nn_x/w5"
PY = f"{ROOT}/.venv/bin/python"
OUT = f"{HERE}/e2e_out"
os.makedirs(f"{D}/logs", exist_ok=True)
for p_ in (HERE, D):
    if p_ not in sys.path:
        sys.path.insert(0, p_)
sys.path.insert(2, W5)  # [D, HERE, W5]: model_moe / moe_arms from D, the w5 model module as 'model'
import model as Mo  # noqa: E402  (lab/nn_x/w5/model.py)
import model_moe as Mm  # noqa: E402
from moe_arms import MuonBatched, setup_moe  # noqa: E402
from strip_moe import strip  # noqa: E402

assert os.path.dirname(Mo.__file__) == W5 and os.path.dirname(Mm.__file__) == D, (Mo.__file__, Mm.__file__)
LOG = open(f"{D}/logs/test_moe_cpu.log", "w")
RES, FAILS, N = {}, [], [0]
KD = "<repo>/nn_x/w12/kd/out/teacher_mean11.npy"
E2E = ("--no_pos --aux_pos 0.5 --aux_mode listwise --aux_tau 10 --sim_bias --user0 --domain --holdout 30000 --select_on holdout "
       "--loss_norm feed --xt cvnd_pre2s --xt_drop 0 --muon --lr 2e-3 --sched wsd --cooldown_frac 0.5 --cooldown_mix 0.3 "
       "--aux_anneal 0.1 --retr_m 16 --attn_gate --epochs 2.5 --eval_every 10 --no_test --save_last --save_holdout "
       f"--select_min_step 10 --cliplog 5 --warmup 2 --cand_budget 4096 --max_feeds 8 --seed 951 --save_steps 10,20 --kd_file {KD} "
       "--kd_w 0.5 --d 192")
ARMS = {"A": "--moe_ffn 8 --moe_w 48 --moe_k 2 --moe_scale 2.0 --moe_xdrop 0.1 --moe_eta 0.1",
        "B": "--vo_moe 4 --vo_k 2 --vo_eta 0.1", "C": "--blocks 3 --ff 1"}
MOE_DEFAULTS = dict(moe_ffn=0, moe_w=48, moe_k=2, moe_scale=2.0, moe_xdrop=0.1, moe_eta=0.1, vo_moe=0, vo_k=2, vo_eta=0.1)
EXP = dict(ref=(2118848, 2118848), A=(2714816, 2272448), B=(2720960, 2426048), C=(2270340, 2270340))


def P(s):
    print(s, flush=True)
    LOG.write(s + "\n")
    LOG.flush()


def check(name, ok, info=""):
    N[0] += 1
    ok = bool(ok)
    if not ok:
        FAILS.append(name)
    P(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({info})" if info != "" else ""))
    return ok


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


# ------------------------------------------------------------------------------------------------ T0 static
EXP_TRAIN_REMOVED = 26
EXP_MODEL_DEFS = {"def quantile_shift", "def RouteSite", "def RoutedFFN", "def SwitchVO", "def _forward_moe_vo", "def _forward_moe",
                  "def moe_modules"}


def t0_static():
    P("== T0 static")
    check("w5 originals are the designed versions (train.py md5 7c1e3588, model.py md5 2464dbb6)",
          md5(f"{W5}/train.py").startswith("7c1e3588") and md5(f"{W5}/model.py").startswith("2464dbb6"),
          (md5(f"{W5}/train.py")[:8], md5(f"{W5}/model.py")[:8]))
    d1, r1 = strip(f"{D}/train.py")
    d0, _ = strip(f"{W5}/train.py", moe_side=False)
    stmts = [x for x in r1 if x.startswith("stmt")]
    check("train.py minus the moe additions == lab/nn_x/w5/train.py (AST, docstrings dropped)", d1 == d0, f"{len(r1)} removals")
    check("train.py removals are exactly: import rename, moe_arms import, 9 flags, --init guard, **moe_kwargs(a), setup / adamw_groups / "
          "MuonBatched / update_bias / summary / hist blocks",
          len(r1) == EXP_TRAIN_REMOVED and r1[0] == "rename import model_moe -> model" and sum("add_argument" in x for x in stmts) == 9
          and sum(x.startswith("kw **moe_kwargs") for x in r1) == 1 and not any(x.startswith(("def", "args")) for x in r1)
          and all(re.search(r"moe", x) for x in r1[1:]), r1)
    d1, r2 = strip(f"{D}/model_moe.py")
    d0, _ = strip(f"{W5}/model.py", moe_side=False)
    check("model_moe.py minus the moe additions == lab/nn_x/w5/model.py (AST, docstrings dropped)", d1 == d0, f"{len(r2)} removals")
    defs = {x for x in r2 if x.startswith("def")}
    args = [x for x in r2 if x.startswith("args")]
    check("model_moe.py removals: the routed classes + moe-only methods, the qmask / tmask / 9 Ranker arguments, guarded statements only",
          defs == EXP_MODEL_DEFS and len(args) == 3 and "stmt MOE_PRIME = 155003" in r2
          and all(re.search(r"moe|\.vo\b|vo_|qmask|tmask|MOE_PRIME|RoutedFFN|SwitchVO", x) or x.endswith(":")  # emptied compound header
                  for x in r2 if x.startswith("stmt")),
          sorted(defs))
    RES["t0"] = dict(train_removed=r1, model_removed=r2)


# ------------------------------------------------------------------------------------------------ synthetic batch
CFG = torch.load(f"{W5}/ckpt/w5_e10_kd05w192_s951.pt", map_location="cpu", weights_only=False)["cfg"]
D_ = CFG["d"]


def batch(B=6, M=40, K=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    b = {}
    Mi = torch.randint(5, M + 1, (B,), generator=g)
    Mi[0] = M
    b["cmask"] = torch.arange(M)[None, :] < Mi[:, None]
    b["ce_raw"] = torch.randint(-60, 60, (B, M, 128), generator=g).float()
    b["cmiss"] = torch.rand(B, M, generator=g) < 0.05
    b["ccat"] = torch.randint(0, 54, (B, M), generator=g)
    b["cf"] = torch.randn(B, M, 5, generator=g)
    b["ue_raw"] = torch.randint(-60, 60, (B, 128), generator=g).float()
    b["user0"] = torch.zeros(B, dtype=torch.bool)
    b["M"] = Mi
    b["xt"] = torch.randn(B, M, len(CFG["xt_names"]), generator=g)
    b["dom"] = torch.tensor([1.0, 0.0, 0.0]).expand(B, 3)
    Hi = torch.randint(0, K + 1, (B,), generator=g)
    Hi[0] = K
    b["he_raw"] = torch.randint(-60, 60, (B, K, 128), generator=g).float()
    b["hmask"] = torch.arange(K)[None, :] < Hi[:, None]
    b["hemb_ok"] = (torch.rand(B, K, generator=g) < 0.85) & b["hmask"]
    b["het"] = torch.randint(1, 11, (B, K), generator=g) * b["hmask"]
    b["hes"] = torch.randint(0, 30, (B, K), generator=g)
    b["hcat"] = torch.randint(0, 54, (B, K), generator=g)
    b["H"] = Hi
    y = ((torch.rand(B, M, generator=g) < 0.08) & b["cmask"]).float()
    y[:, 0] = 1.0
    return b, y


def build(mod, cfg, seed=951):
    torch.manual_seed(seed)
    return mod.Ranker(**cfg)


def fwd_loss(m, b, y, seed=5):
    torch.manual_seed(seed)
    bb = dict(b)  # dense() adds the normalised embeddings that forward() reads
    X = m.dense(bb)
    lg = m(bb, X, use_pos=True)
    loss = Mo.listwise_loss(lg, y, b["cmask"]) + 0.5 * Mo.listwise_loss(m.aux_out, y, b["cmask"])
    return lg, loss


def opts_for(m, a_lr=2e-3, wd=0.02):
    """the trainer's optimiser construction (train.py lines 'muon_names' .. 'opts.append(moe.optimizer(a))'), verbatim logic."""
    class A_:
        lr, muon = a_lr, True
    A_.wd = wd
    muon_names = {f"{mn}.weight" for mn, mod in m.named_modules() if isinstance(mod, torch.nn.Linear) and min(mod.weight.shape) > 1}
    moe = setup_moe(A_, m, muon_names, lambda *x: None) if m.moe_on else None
    rest = [(n, p) for n, p in m.named_parameters() if n not in muon_names]
    decay = [p for n, p in rest if p.ndim >= 2 and "emb" not in n]
    no_decay = [p for n, p in rest if not (p.ndim >= 2 and "emb" not in n)]
    if moe is not None:
        decay, no_decay = moe.adamw_groups(rest)
    opts = [torch.optim.AdamW([{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}], lr=a_lr, betas=(0.9, 0.98))]
    P_muon = [p for n, p in m.named_parameters() if n in muon_names]
    opts.append(torch.optim.Muon(P_muon, lr=a_lr, weight_decay=wd, momentum=0.95, nesterov=True, adjust_lr_fn="match_rms_adamw"))
    if moe is not None:
        opts.append(moe.optimizer(A_))
    return opts, moe, muon_names, decay, no_decay


# ------------------------------------------------------------------------------------------------ T1a model-level identity
def t1_model():
    P("\n== T1a flag-off identity, model level (synthetic batch with padding, CPU, fixed seeds)")
    b, y = batch()
    for tag, cfg in (("student cfg (w5_e10_kd05w192_s951)", CFG), ("arm-C cfg (blocks 3, ff 1)", {**CFG, "blocks": 3, "ff": 1})):
        m0, m1 = build(Mo, cfg), build(Mm, cfg)
        s0, s1 = m0.state_dict(), m1.state_dict()
        same_sd = s0.keys() == s1.keys() and all(torch.equal(s0[k], s1[k]) for k in s0)
        m0.train(), m1.train()
        lg0, l0 = fwd_loss(m0, b, y)
        lg1, l1 = fwd_loss(m1, b, y)
        l0.backward()
        l1.backward()
        g0, g1 = dict(m0.named_parameters()), dict(m1.named_parameters())
        same_g = g0.keys() == g1.keys() and all((g0[k].grad is None and g1[k].grad is None) or torch.equal(g0[k].grad, g1[k].grad) for k in g0)
        m0.eval(), m1.eval()
        with torch.no_grad():
            e0, e1 = m0(dict(b), use_pos=False), m1(dict(b), use_pos=False)
        check(f"{tag}: identical state dict ({len(s0)} tensors), train-mode logits, aux output, loss, all {len(g0)} gradients, eval logits, cfg",
              same_sd and torch.equal(lg0, lg1) and torch.equal(m0.aux_out, m1.aux_out) and torch.equal(l0, l1) and same_g
              and torch.equal(e0, e1) and m0.cfg == m1.cfg and not m1.moe_on,
              f"loss {float(l0.detach()):.6f}, params {sum(p.numel() for p in m0.parameters()):,}")


# ------------------------------------------------------------------------------------------------ e2e runs
def e2e_cmd(train_file, v, extra, rec):
    return [PY, f"{HERE}/e2e_moe_cpu.py", train_file, f"{OUT}/{v}", rec, "--variant", v] + E2E.split() + extra.split()


def run_parallel(jobs, threads):
    """jobs: list of (name, cmd); at most 2 concurrent processes."""
    env = {**os.environ, "OMP_NUM_THREADS": str(threads), "MKL_NUM_THREADS": str(threads), "CUDA_VISIBLE_DEVICES": "",
           "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": ""}
    res, run = {}, []
    pending = list(jobs)
    while pending or run:
        while pending and len(run) < 2:
            name, cmd = pending.pop(0)
            fo = open(f"{OUT}/{name}.stdout", "w")
            run.append((name, subprocess.Popen(cmd, stdout=fo, stderr=subprocess.STDOUT, cwd=OUT, env=env), time.time(), fo))
        time.sleep(1)
        for it in list(run):
            name, pr, t0, fo = it
            if pr.poll() is not None:
                fo.close()
                res[name] = (pr.returncode, time.time() - t0)
                P(f"  ran {name} (rc {pr.returncode}, {time.time() - t0:.0f}s)")
                run.remove(it)
    return res


def metric_lines(log):
    return [re.sub(r" [\d.]+ min", "", ln) for ln in log.splitlines() if ln.startswith("step ") or ln.startswith("BEST")]


def ck_equal(a, b):
    ca = torch.load(a, map_location="cpu", weights_only=False)
    cb = torch.load(b, map_location="cpu", weights_only=False)
    sa, sb = ca["state_dict"], cb["state_dict"]
    return sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa) and ca["cfg"] == cb["cfg"]


def split_rec(r, i):
    o = r["lg_model_off"]
    return r["lg_model"][o[i]:o[i + 1]]


def t1_e2e():
    P("\n== T1b flag-off identity, trainer level (e2e_moe_cpu.py, miniature split, E recipe at mini scale with the KD teacher, 1 thread)")
    names = {"ref": "w5_moe_e2e_ref", "off": "w5_moe_e2e_off"}
    rec = {k: f"{OUT}/rec_{k}.npz" for k in names}
    res = run_parallel([("ref", e2e_cmd(f"{W5}/train.py", names["ref"], "", rec["ref"])),
                        ("off", e2e_cmd(f"{D}/train.py", names["off"], "", rec["off"]))], threads=1)
    assert all(rc == 0 for rc, _ in res.values()), res
    ra, rb = dict(np.load(rec["ref"])), dict(np.load(rec["off"]))
    nb = len(ra["total"])
    check("OFF vs live w5: identical training-batch sequence (incl. cooldown pool draws), 50 steps",
          np.array_equal(ra["fidx"], rb["fidx"]) and np.array_equal(ra["off"], rb["off"]) and nb == 50, f"{nb} steps")
    check("OFF vs live w5: bitwise identical train-mode logits of every step and identical loss values",
          np.array_equal(ra["lg_model"], rb["lg_model"]) and np.array_equal(ra["lg_model_off"], rb["lg_model_off"])
          and np.array_equal(ra["total"], rb["total"]) and np.isfinite(ra["total"]).all(), f"loss[0] {ra['total'][0]:.6f} loss[-1] {ra['total'][-1]:.6f}")
    lg = {k: open(f"{OUT}/{v}/logs/{v}.log").read() for k, v in names.items()}
    ma, mb = metric_lines(lg["ref"]), metric_lines(lg["off"])
    check("OFF vs live w5: identical logged loss / val / holdout / BEST lines; no w12/moe line in the OFF log",
          ma == mb and len(ma) == 6 and "w12/moe" not in lg["off"], mb[-1])
    ck = lambda k, suf: f"{OUT}/{names[k]}/ckpt/{names[k]}{suf}.pt"  # noqa: E731
    same_ck = all(ck_equal(ck("ref", s), ck("off", s)) for s in ("", "@last", "@10", "@20"))
    pr = lambda k, f: np.load(f"{OUT}/{names[k]}/preds/{f.format(v=names[k])}")  # noqa: E731
    pfiles = ["nn_{v}_trainonly_val.npy", "nn_{v}@last_trainonly_val.npy", "holdout/nn_{v}_holdout.npy", "holdout/nn_{v}@last_holdout.npy"]
    same_pr = all(np.array_equal(pr("ref", f), pr("off", f)) for f in pfiles)
    ca, cb = (np.load(f"{OUT}/{names[k]}/logs/{names[k]}_cliplog.npz") for k in ("ref", "off"))
    same_cl = ca.files == cb.files and all(np.array_equal(ca[f], cb[f]) for f in ca.files)
    check("OFF vs live w5: bit-identical selected / @last / @10 / @20 checkpoints (tensors + cfg), val + holdout preds (selected and "
          "@last), cliplog arrays", same_ck and same_pr and same_cl, (same_ck, same_pr, same_cl))
    ja, jb = (json.load(open(f"{OUT}/{names[k]}/logs/{names[k]}.json")) for k in ("ref", "off"))
    d_args = {k: (ja["args"].get(k), jb["args"].get(k)) for k in set(ja["args"]) | set(jb["args"]) if ja["args"].get(k) != jb["args"].get(k)}
    exp_d = {"variant": (names["ref"], names["off"]), **{k: (None, v) for k, v in MOE_DEFAULTS.items()}}
    hist = lambda j: [(h["step"], h["loss"], h["val_ndcg"], h["holdout_ndcg"]) for h in j["history"]]  # noqa: E731
    check("OFF: stored args == w5 args + the 9 moe defaults; same best_step / val_ndcg / history (no 'moe' key)",
          d_args == exp_d and ja["best_step"] == jb["best_step"] and ja["val_ndcg"] == jb["val_ndcg"] and hist(ja) == hist(jb)
          and not any("moe" in h for h in jb["history"]), sorted(d_args))
    RES["t1_e2e"] = dict(lines=mb, steps=nb)
    return rb


# ------------------------------------------------------------------------------------------------ T2 arms, model level
def arm_cfg(arm):
    if arm == "A":
        return {**CFG, "moe_ffn": 8, "moe_w": 48, "moe_k": 2, "moe_scale": 2.0, "moe_xdrop": 0.1, "moe_eta": 0.1}
    if arm == "B":
        return {**CFG, "vo_moe": 4, "vo_k": 2, "vo_eta": 0.1}
    return {**CFG, "blocks": 3, "ff": 1}


def active(m):
    n = sum(p.numel() for p in m.parameters())
    for _, mod in m.moe_modules():
        if isinstance(mod, Mm.RoutedFFN):
            n -= (mod.n_exp - mod.k) * (mod.w1[0].numel() + mod.w2[0].numel())
        else:
            n -= mod.h * (mod.E - mod.k) * (mod.wv[0].numel() + mod.wo[0].numel())
    return n


def t2_arms():
    P("\n== T2 arms, model level (CPU)")
    b, y = batch(B=8, M=48, K=96, seed=3)
    ref = build(Mm, CFG)
    ref_sd = ref.state_dict()
    ref_muon = {f"{mn}.weight" for mn, mod in ref.named_modules() if isinstance(mod, torch.nn.Linear) and min(mod.weight.shape) > 1}
    for arm in ("A", "B", "C"):
        m = build(Mm, arm_cfg(arm))
        tot, act = sum(p.numel() for p in m.parameters()), active(m)
        mods = m.moe_modules()
        if arm == "C":
            n_muon = sum(1 for mn, mod in m.named_modules() if isinstance(mod, torch.nn.Linear) and min(mod.weight.shape) > 1)
            check("C: total parameters 2,270,340 (+151,492), 60 Muon matrices, 7 attention blocks, FFN hidden 192, no routed module",
                  (tot, act) == EXP["C"] and n_muon == 60 and len(m.cross) == 3 and len(m.sets) == 3 and len(m.hist_enc) == 1
                  and m.cross[0].ff[0].weight.shape == (192, 192) and not mods and not m.moe_on, f"{tot:,}, {n_muon} Muon matrices")
            continue
        check(f"{arm}: total / active parameters as designed {EXP[arm][0]:,} / {EXP[arm][1]:,}", (tot, act) == EXP[arm], f"{tot:,} / {act:,}")
        if arm == "A":
            shp = all(mod.w1.shape == (8, 48, 192) and mod.w2.shape == (8, 48, 192) and mod.router.weight.shape == (8, 192)
                      and mod.router.bias is None and mod.rbias.shape == (1, 8) and float(mod.w2.abs().max()) == 0 for _, mod in mods)
            names = [n for n, _ in mods]
            check("A: 4 RoutedFFN at cross.0, sets.0, cross.1, sets.1 (as .moe siblings of the dense ff); w1 / w2 [8,48,192], router "
                  "[8,192] no bias, rbias [1,8], w2 zero-init", shp and sorted(names) == ["cross.0.moe", "cross.1.moe", "sets.0.moe", "sets.1.moe"]
                  and all(not hasattr(blk, "moe") or blk.moe is None for blk in m.hist_enc), names)
        else:
            shp = all(mod.wv.shape == (16, 192, 48) and mod.wo.shape == (16, 48, 192) and mod.rv.weight.shape == (16, 192)
                      and mod.ro.weight.shape == (16, 192) and mod.rbias.shape == (2, 4, 4) and float(mod.wv.abs().max()) == 0
                      and float(mod.wo.abs().max()) == 0 for _, mod in mods)
            names = [n for n, _ in mods]
            check("B: 2 SwitchVO at cross.0.attn.vo, cross.1.attn.vo; wv [16,192,48], wo [16,48,192], rv / ro [16,192], rbias [2,4,4]; "
                  "wv / wo zero-init; none in the set blocks", shp and names == ["cross.0.attn.vo", "cross.1.attn.vo"]
                  and all(blk.attn.vo is None for blk in m.sets), names)
        sd = m.state_dict()
        new = [k for k in sd if k not in ref_sd]
        check(f"{arm}: every reference tensor present and identical (same init); new keys only under .moe. / .attn.vo.",
              all(k in sd and torch.equal(sd[k], ref_sd[k]) for k in ref_sd)
              and all((".moe." in k) if arm == "A" else (".attn.vo." in k) for k in new), f"{len(ref_sd)} reference + {len(new)} new tensors")
        m.eval(), ref.eval()
        with torch.no_grad():
            e0, e1 = ref(dict(b), use_pos=False), m(dict(b), use_pos=False)
        check(f"{arm}: eval logits at init == reference bit for bit", torch.equal(e0, e1), f"max |diff| {float((e0 - e1).abs().max()):.1e}")
        m.train(), ref.train()
        lg_r, _ = fwd_loss(ref, b, y)
        lg_m, loss = fwd_loss(m, b, y)
        if arm == "B":
            check("B: train-mode step-0 logits == reference bit for bit (no extra RNG draw)", torch.equal(lg_r, lg_m))
        # optimiser split
        opts, moe, muon_names, decay, no_decay = opts_for(m)
        routers = {f"{n}.{r}.weight" for n, mod in mods for r in ("router", "rv", "ro") if hasattr(mod, r)}
        stacks = {f"{n}.{s}" for n, mod in mods for s in ("w1", "w2", "wv", "wo") if hasattr(mod, s)}
        pid = {id(p): n for n, p in m.named_parameters()}
        groups = [[pid[id(p)] for p in g["params"]] for o in opts for g in o.param_groups]
        flat = [n for g in groups for n in g]
        check(f"{arm}: optimiser split: Muon set == the reference 46 matrices (routers taken out), routers in AdamW no-decay, stacks only "
              "in MuonBatched, every parameter in exactly one group",
              muon_names == ref_muon and routers <= set(groups[1]) and not (routers & set(groups[0])) and set(groups[3]) == stacks
              and isinstance(opts[2], MuonBatched) and sorted(flat) == sorted(pid.values()) and len(flat) == len(set(flat)),
              f"AdamW decay {len(groups[0])} / no-decay {len(groups[1])}, Muon {len(groups[2])}, MuonBatched {len(groups[3])}")
        # gradients at step 0
        loss.backward()
        if arm == "A":
            g0 = all(bool((mod.w2.grad.flatten(1).abs().sum(1) > 0).all()) for _, mod in mods)
            check("A: step 0: the gradient reaches w2 of every expert of every module (w1 / router are exactly 0 while w2 = 0)",
                  g0 and all(float(mod.w1.grad.abs().max()) == 0 and float(mod.router.weight.grad.abs().max()) == 0 for _, mod in mods))
        else:
            g0 = all(bool((mod.wv.grad.flatten(1).abs().sum(1) > 0).all()) and bool((mod.wo.grad.flatten(1).abs().sum(1) > 0).all())
                     for _, mod in mods)
            check("B: step 0: the gradient reaches wv and wo of every (head, expert) slice (routers exactly 0 while wv = wo = 0)",
                  g0 and all(float(mod.rv.weight.grad.abs().max()) == 0 and float(mod.ro.weight.grad.abs().max()) == 0 for _, mod in mods))
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        for o in opts:
            o.step()
        moe.update_bias()
        for o in opts:
            o.zero_grad(set_to_none=True)
        _, loss2 = fwd_loss(m, b, y, seed=6)
        loss2.backward()
        ok = True
        for _, mod in mods:
            for s in ("w1", "w2", "wv", "wo"):
                if hasattr(mod, s):
                    ok &= bool((getattr(mod, s).grad.flatten(1).abs().sum(1) > 0).all())
            for r in ("router", "rv", "ro"):
                if hasattr(mod, r):
                    ok &= bool((getattr(mod, r).weight.grad.abs().sum(1) > 0).all())
        fin = all(torch.isfinite(p).all() and (p.grad is None or torch.isfinite(p.grad).all()) for p in m.parameters())
        moved = all(float(mod.rbias.abs().max()) > 0 for _, mod in mods)
        check(f"{arm}: after one optimiser step: every expert slice ({'w1, w2' if arm == 'A' else 'wv, wo'}) and every router row get a "
              "nonzero gradient; rbias moved; all parameters / gradients finite", ok and fin and moved and torch.isfinite(loss2))
        # checkpoint round trip
        m.eval()
        with torch.no_grad():
            e_a = m(dict(b), use_pos=False)
        m2 = Mm.Ranker(**m.cfg)
        m2.load_state_dict(m.state_dict())
        m2.eval()
        with torch.no_grad():
            e_b = m2(dict(b), use_pos=False)
        check(f"{arm}: checkpoint round trip (Ranker(**cfg) + strict load_state_dict incl. rbias) gives identical eval logits; cfg carries "
              "the arm keys", torch.equal(e_a, e_b) and all(torch.equal(x.rbias, y_.rbias) for (_, x), (_, y_) in zip(mods, m2.moe_modules()))
              and (("moe_ffn" in m.cfg) if arm == "A" else ("vo_moe" in m.cfg)))


def t2_units():
    P("\n== T2 units: MuonBatched, masked quantile, balancer on a synthetic stream")
    for shp in ((48, 192), (192, 48)):
        torch.manual_seed(1)
        w0 = torch.randn(3, *shp) * 0.05
        ps = [torch.nn.Parameter(w0[i].clone()) for i in range(3)]
        pb = torch.nn.Parameter(w0.clone())
        o1 = torch.optim.Muon(ps, lr=2e-3, weight_decay=0.02, momentum=0.95, nesterov=True, adjust_lr_fn="match_rms_adamw")
        o2 = MuonBatched([pb], lr=2e-3, weight_decay=0.02)
        for _ in range(5):
            gr = torch.randn(3, *shp)
            for i in range(3):
                ps[i].grad = gr[i].clone()
            pb.grad = gr.clone()
            o1.step()
            o2.step()
        diff = max(float((ps[i] - pb[i]).abs().max()) for i in range(3))
        check(f"MuonBatched [3,{shp[0]},{shp[1]}] == torch.optim.Muon on each slice after 5 steps (max |diff| 0)", diff == 0.0,
              f"diff {diff:.1e}, movement {float((pb - w0).abs().max()):.1e}")
    g = torch.Generator().manual_seed(0)
    z = torch.randn(500, 4, 8, generator=g)
    tm = torch.rand(500, generator=g) < 0.7
    q1, q2 = Mm.quantile_shift(z, 2, tm), Mm.quantile_shift(z[tm], 2)
    check("quantile_shift with a token mask (nanquantile) == quantile over the real tokens only", torch.allclose(q1, q2, atol=1e-6, rtol=0),
          f"max |diff| {float((q1 - q2).abs().max()):.1e}")
    z = z + torch.linspace(0, 1.5, 8)  # skewed: the high experts overloaded
    T = int(tm.sum())
    q = Mm.quantile_shift(z, 2, tm)

    def load(zz):
        return (zz >= zz.topk(2, -1).values[..., -1:])[tm].float().mean(0)  # [G, E] selection share of each expert per token

    l0, ok_over, ok_under = load(z), True, True
    for e in range(8):
        zz = z.clone()
        zz[..., e] -= q[:, e]
        le = load(zz)[:, e]
        for gi in range(4):
            if l0[gi, e] > 0.25:   # overloaded: exact
                ok_over &= abs(float(le[gi]) - 0.25) <= 1.0 / T + 1e-6
            else:                  # underloaded: lifted toward k/E, never lowered, never past it
                ok_under &= float(le[gi]) >= float(l0[gi, e]) - 1e-6 and float(le[gi]) <= 0.25 + 1.0 / T + 1e-6
    l1 = load(z - q[None])
    d0, d1 = float((l0 - 0.25).abs().max()), float((l1 - 0.25).abs().max())
    check("quantile_shift (design definition, theta = (k+1)-th score): shifting one OVERLOADED expert by its q gives it exactly k/E "
          "(within 1/T); an underloaded expert is lifted toward k/E, never lowered or overshot; the full eta-1 shift of all experts cuts "
          "the max load deviation by more than half", ok_over and ok_under and d1 < 0.5 * d0,
          f"T {T}; max |load - 0.25| {d0:.3f} -> {d1:.3f}")

    def stream(mod, kind, steps, eta):
        """low-diversity tokens (x = base + 0.03 noise) with the router drifting toward expert 0 every step (the A5 failure mode)."""
        gg = torch.Generator().manual_seed(7)
        base = torch.randn(192, generator=gg)
        ents = []
        for it in range(steps + 1):
            x = base + 0.03 * torch.randn(2048, 192, generator=gg)
            msk = torch.rand(2048, generator=gg) < 0.8
            with torch.no_grad():
                if kind == "ffn":
                    mod.router.weight[0] += 1e-3 * base / base.norm() ** 2  # logit drift 1e-3 / step
                    mod(x, msk, shared=torch.ones(2048, 192))
                    st = mod.sites[0].stats(mod.rbias)
                elif kind == "V":
                    mod.rv.weight[0::mod.E] += 1e-3 * base / base.norm() ** 2
                    mod.value(x[None], msk[None], torch.ones(1, 2048, mod.h, mod.dh))
                    st = mod.sites[0].stats(mod.rbias[0])
                else:
                    mod.ro.weight[0::mod.E] += 1e-3 * base / base.norm() ** 2
                    o = torch.randn(1, 2048, mod.h, mod.dh, generator=gg)
                    mod.output(o, x[None], msk[None], torch.ones(1, 2048, 192))
                    st = mod.sites[1].stats(mod.rbias[1])
            mod.eta = eta
            mod.update_bias()
            ents.append(st["ent"])
        return ents

    torch.manual_seed(11)
    for kind in ("ffn", "V", "O"):
        res = {}
        for eta in (0.1, 0.0):
            torch.manual_seed(11)
            mod = Mm.RoutedFFN(192, 8, 48, 2, 2.0, 0.1, 0.1, eta) if kind == "ffn" else Mm.SwitchVO(192, 4, 4, 2, eta)
            mod.train()
            with torch.no_grad():  # a skewed start: expert 0 preferred
                if kind == "ffn":
                    mod.router.weight.mul_(0.3)
                    mod.router.weight[0] += 0.3
                else:
                    r = mod.rv if kind == "V" else mod.ro
                    r.weight.mul_(0.3)
                    r.weight[0::4] += 0.3
            res[eta] = stream(mod, kind, 300, eta)
        e1, e0 = res[0.1], res[0.0]
        nanfree = all(math.isfinite(x) for x in e1 + e0)
        check(f"balancer ({'RoutedFFN' if kind == 'ffn' else 'SwitchVO ' + kind}): skewed start + low-diversity drifting stream (logit drift "
              f"1e-3 / step toward expert 0): with eta 0.1 the load entropy rises from <= 0.5 to >= 0.9 by step 50, stays >= 0.95 over steps "
              f"100-300 and ends >= 0.98; without balancing (eta 0) it stays within +0.05 of the start value; no NaN",
              nanfree and e1[0] <= 0.5 + 1e-6 and e1[50] >= 0.9 and min(e1[100:]) >= 0.95 and e1[-1] >= 0.98 and max(e0) <= e0[0] + 0.05,
              f"eta 0.1: ent {e1[0]:.3f} -> {e1[10]:.3f} (10) -> {e1[50]:.3f} (50) -> {e1[100]:.3f} (100) -> {e1[-1]:.3f} (300), min over "
              f"100-300 {min(e1[100:]):.3f}; eta 0: {e0[0]:.3f} -> max {max(e0):.3f}")
        RES[f"balancer_{kind}"] = dict(eta01=[round(x, 4) for x in e1[::25]], eta0=[round(x, 4) for x in e0[::25]])


# ------------------------------------------------------------------------------------------------ T3 50-step training loops per arm
def t3_arms(rb_off):
    P("\n== T3 50-step CPU training loop per arm through the trainer (e2e_moe_cpu.py, miniature split, E recipe at mini scale)")
    names = {k: f"w5_moe_e2e_{k}" for k in ARMS}
    rec = {k: f"{OUT}/rec_{k}.npz" for k in ARMS}
    res = run_parallel([(k, e2e_cmd(f"{D}/train.py", names[k], ARMS[k], rec[k])) for k in ("A", "B", "C")], threads=max(1, NT // 2))
    for k in ("A", "B", "C"):
        v = names[k]
        rc = res[k][0]
        r = dict(np.load(rec[k])) if rc == 0 else None
        log = open(f"{OUT}/{v}/logs/{v}.log").read() if os.path.exists(f"{OUT}/{v}/logs/{v}.log") else ""
        js = json.load(open(f"{OUT}/{v}/logs/{v}.json")) if os.path.exists(f"{OUT}/{v}/logs/{v}.json") else {}
        ok = rc == 0 and r is not None and len(r["total"]) == 50 and np.isfinite(r["total"]).all() and np.isfinite(r["lg_model"]).all()
        check(f"{k}: 50 training steps complete (rc 0), every loss and train-mode logit finite, json + selected / @last checkpoints written",
              ok and js.get("best_step") is not None and os.path.exists(f"{OUT}/{v}/ckpt/{v}.pt") and os.path.exists(f"{OUT}/{v}/ckpt/{v}@last.pt"),
              f"loss {r['total'][0]:.4f} -> {r['total'][-10:].mean():.4f} (mean of the last 10); BEST step {js.get('best_step')} val "
              f"{js.get('val_ndcg', float('nan')):.5f}" if ok else f"rc {rc}")
        if not ok:
            continue
        ck = torch.load(f"{OUT}/{v}/ckpt/{v}@last.pt", map_location="cpu", weights_only=False)
        sd = ck["state_dict"]
        if k == "C":
            check("C: cfg blocks 3 / ff 1, 'params: 2.27M' and 'muon: 60 matrices' logged, no w12/moe line",
                  ck["cfg"]["blocks"] == 3 and ck["cfg"]["ff"] == 1 and "params: 2.27M" in log and "muon: 60 matrices" in log
                  and "w12/moe" not in log)
            continue
        lines = [ln for ln in log.splitlines() if ln.startswith("w12/moe step ")]
        hist_ok = all(isinstance(h.get("moe"), dict) and 0 < h["moe"]["ent_min"] <= 1 for h in js["history"])
        ents = [h["moe"]["ent_min"] for h in js["history"]]
        key = "moe_ffn" if k == "A" else "vo_moe"
        stacks = [kk for kk in sd if kk.endswith((".w2", ".wv", ".wo"))]
        biases = [kk for kk in sd if kk.endswith(".rbias")]
        check(f"{k}: setup line + a 'w12/moe step' diagnostics line at each of the 5 evals, json history carries them; arm cfg in the "
              "checkpoint; rbias and the zero-init stacks moved (mechanism live)",
              "w12/moe: " in log and len(lines) == 5 and hist_ok and key in ck["cfg"] and all(float(sd[kk].abs().max()) > 0 for kk in stacks + biases)
              and len(biases) == (4 if k == "A" else 2), f"min load entropy per eval {[round(x, 3) for x in ents]}")
        if k == "B":
            s0 = int(r["lg_model_off"][1])
            check("B: step-0 train-mode logits == the flag-off run's (exact reference start in training mode too)",
                  np.array_equal(r["lg_model"][:s0], rb_off["lg_model"][:s0]))
        RES[f"t3_{k}"] = dict(loss_first=float(r["total"][0]), loss_last10=float(r["total"][-10:].mean()), ents=ents, lines=lines[-1:])


def main():
    t_0 = time.time()
    P(f"test_moe_cpu.py  {time.strftime('%Y-%m-%d %H:%M:%S')}  threads {NT}  torch {torch.__version__}")
    shutil.rmtree(OUT, ignore_errors=True)
    os.makedirs(OUT, exist_ok=True)
    t0_static()
    t1_model()
    rb = t1_e2e()
    t2_arms()
    t2_units()
    t3_arms(rb)
    P(f"\n== {N[0] - len(FAILS)} of {N[0]} checks PASS" + (f"; FAILED: {FAILS}" if FAILS else "") + f"  ({time.time() - t_0:.0f} s)")
    RES["fails"], RES["n_checks"] = FAILS, N[0]
    json.dump(RES, open(f"{D}/logs/test_moe_cpu.json", "w"), indent=1, default=str)
    for f in os.listdir(OUT):  # keep the logs / records small: drop the miniature checkpoints and preds
        for sub in ("ckpt", "preds"):
            shutil.rmtree(f"{OUT}/{f}/{sub}", ignore_errors=True)
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
