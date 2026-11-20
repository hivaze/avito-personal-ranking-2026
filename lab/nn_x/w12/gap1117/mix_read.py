"""Descriptive (not a decision): the T1117 harness arms against the reference re-cools on validation re-weighted to the
public and private mix of feed size and cold share (pmix.weights(), as build_r4.py), plus M<50 / M>=50.

  cd <repo> && CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 .venv/bin/python nn_x/w12/gap1117/mix_read.py
"""
import sys
import numpy as np
sys.path.insert(0, "<repo>/nn_x/w12/recheck/pool_composition")
sys.path.insert(0, "<repo>/feat_new")
from pc_common import Meta, ndcg_fast, paths  # noqa: E402
import pmix  # noqa: E402

m = Meta(); w = pmix.weights(); big = m.Mv >= 50
pool = lambda a: np.mean([m.z("val", np.load(paths(f"w5_{st}cd_{a}_s{x}")[0])) for st in "kl" for x in range(951, 959)], 0)
nd = lambda a: ndcg_fast(pool(a), m.yv, m.ov, m.Mv)
r = nd("ref")
f = lambda d, wt=None: "{:+.5f} ({:.5f})".format(*map(float, pmix.wmean_se(d, wt))) if wt is not None else f"{d.mean():+.5f} ({d.std(ddof=1) / np.sqrt(len(d)):.5f})"
print(f"{'arm':7s} {'val':>18s} {'pmix-public':>18s} {'pmix-private':>18s} {'M<50':>18s} {'M>=50':>18s}")
for a in ["ref2", "wcl05", "wcl10", "ccl05", "fsw25t", "fsw50t", "kw03", "lr4", "e4", "hx"]:
    d = nd(a) - r
    print(f"{a:7s} {f(d):>18s} {f(d, w['public']):>18s} {f(d, w['private']):>18s} {f(d[~big]):>18s} {f(d[big]):>18s}", flush=True)
