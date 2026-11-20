"""Step 4: placebo of the nested read (DECLARATION.txt section D).

  P1 (decision placebo, "profile swap"): within (Moscow day x cold flag x 3-hour block of the feed time) the vector of raw counts
     (a, a24, b, c, d, e) is permuted across val candidate rows; the within-feed forms are recomputed; identical nested read.
  P2 (literal, information): the item-id column of the SOURCE rows (val + test) is permuted within each Moscow day and the
     counts of the val query rows (true items) are recomputed; identical nested read.
Usage:  s4_placebo.py P1 100 119      (kind, first seed, last seed inclusive)  -> appends to out/s4_placebo_<kind>.jsonl
"""
import json
import sys
import time

import numpy as np

sys.path.insert(0, "<repo>/nn_x/w12/fresh/pregate")
from lib import OUT, count, load_union
from nested import FOLD_SEEDS, Val, gain, nested_scores, scale

kind, s0, s1 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
USE_NB = len(sys.argv) > 4 and sys.argv[4] == "nb"  # numba GIL-free objective (same function), for the thread pool
ARMS_RUN = ["A", "B", "ALL"]
t0 = time.time()
V = Val()
assert abs(V.base_ndcg.mean() - 0.40271) < 2e-5
Xc, _ = V.control(); Xc, _ = scale(Xc)
WORKERS = 12  # 12 single-thread fits in one process (BLAS threads = 1 in the environment)
cm = np.load(f"{OUT}/cnt_main.npy")[:V.N]
# control and REAL arms through the same code path as the placebo draws (like for like); compared with s3
mats0 = {"control": Xc}
for arm in ARMS_RUN:
    mats0[arm] = scale(V.arm_matrix(cm, arm)[0])[0]
o0, _ = nested_scores(V, mats0, workers=WORKERS, use_nb=USE_NB)
ctl = o0["control"]
ref = np.load(f"{OUT}/s3_perfeed.npz")
dctl = max(float(np.abs(ctl[i] - ref[f"control_seed{s}"]).max()) for i, s in enumerate(FOLD_SEEDS))
real = {arm: gain(o0[arm], ctl) for arm in ARMS_RUN}
print(f"[{kind}] control vs s3: max per-feed abs diff {dctl:.2e}; REAL arms in this code path: " + " | ".join(
    f"{a} {real[a][1]:+.5f} (s3 {float(ref['d_main_' + a].mean()):+.5f}, max per-feed diff {float(np.abs(real[a][0] - ref['d_main_' + a]).max()):.1e})" for a in ARMS_RUN)
      + f" | {time.time() - t0:.0f}s", flush=True)
assert dctl < 1e-4
with open(f"{OUT}/s4_placebo_{kind}.jsonl", "a") as f:
    f.write(json.dumps({"kind": kind, "seed": "REAL", "path": "nb" if USE_NB else "np", **{a: [real[a][1], real[a][2], real[a][3]] for a in ARMS_RUN}}) + "\n")
if kind == "P1":
    day_r = V.day[V.fid]; blk = (((V.ts + 10800) % 86400) // 10800)[V.fid]
    strata = (day_r - day_r.min()) * 16 + blk * 2 + V.cold.astype(np.int64)
    order = np.argsort(strata, kind="stable"); ss = strata[order]
    bounds = np.flatnonzero(np.r_[True, ss[1:] != ss[:-1], True])
    print(f"[P1] {len(bounds) - 1} strata, smallest {np.diff(bounds).min()} rows", flush=True)
else:
    F, R = load_union()
    sday = F["day"][R["gid"]]
    sorder = np.argsort(sday, kind="stable"); sbounds = np.flatnonzero(np.r_[True, sday[sorder][1:] != sday[sorder][:-1], True])
    qi = R["item"][:V.N]; qg = R["gid"][:V.N]
for seed in range(s0, s1 + 1):
    t1 = time.time()
    rng = np.random.default_rng(seed)
    if kind == "P1":
        src = np.arange(V.N)
        for a, b in zip(bounds[:-1], bounds[1:]):
            idx = order[a:b]
            src[idx] = idx[rng.permutation(b - a)]
        c = cm[src]
    else:
        it = R["item"].copy()
        for a, b in zip(sbounds[:-1], sbounds[1:]):
            idx = sorder[a:b]
            it[idx] = R["item"][idx[rng.permutation(b - a)]]
        c = count(qi, qg, F, it, R["gid"])
    mats = {}
    for arm in ARMS_RUN:
        X, _ = V.arm_matrix(c, arm); mats[arm], _ = scale(X)
    out, _ = nested_scores(V, mats, workers=WORKERS, use_nb=USE_NB)
    rec = {"kind": kind, "seed": seed, "zero_share": [float((c[:, w] == 0).mean()) for w in range(6)], "mean": [float(c[:, w].mean()) for w in range(6)],
           "same_as_real_share": [float((c[:, w] == cm[:, w]).mean()) for w in range(6)],
           "corr_with_real": [float(np.corrcoef(c[:, w], cm[:, w])[0, 1]) for w in range(6)]}
    for arm in ARMS_RUN:
        d, g, se, z = gain(out[arm], ctl)
        rec[arm] = [g, se, z]
    with open(f"{OUT}/s4_placebo_{kind}.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"[{kind}] seed {seed}: " + " | ".join(f"{a} {rec[a][0]:+.5f} (z {rec[a][2]:+.2f})" for a in ARMS_RUN) +
          f" | corr with real counts a/b/d {rec['corr_with_real'][0]:.3f}/{rec['corr_with_real'][2]:.3f}/{rec['corr_with_real'][4]:.3f} | {time.time() - t1:.0f}s", flush=True)
print(f"[{kind}] DONE {time.time() - t0:.0f}s")
