"""C16 REGIME4 Step 0: scratch copy of nn_x/insights/data_usage/07_gbdt_arms.py (params unchanged) with arms
base | regime (extra training column post12 = 1[production feed (cls 0, p_random <= 0.5) and ts >= 2026-09-12 00:00 MSK],
inference value 0) | droppre (drop production feeds before Sep 12) | droppost (drop production feeds on/after Sep 12).
Training sample = 07's U (150K natural-mix train feeds) minus every 30K-holdout feed (asserted). Predicts val (07 val_X
layout) and the 30K holdout (built in the same 53-column layout from data/feats/train).
usage: python c16_arm.py prep | python c16_arm.py ARM SEED"""
import os, sys, json, time
for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS"]:
    os.environ[k] = "12"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.dont_write_bytecode = True
import numpy as np, polars as pl
from pathlib import Path
ROOT = Path("<repo>")
G = ROOT / "nn_x/insights/data_usage/out/gb"
OUT = Path(__file__).resolve().parent / "out"
T0 = time.time()
feats = json.load(open(G / "feats.json"))


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def hold_ids():
    f = pl.read_parquet(ROOT / "data/flat/train/feeds.parquet")
    M = f["M"].to_numpy().astype(np.int64)
    p = np.nan_to_num(np.load(ROOT / "eda/labels_position/out/train_prandom.npz")["p_random"].astype(np.float32), nan=0.0)
    p[M < 50] = 0.0
    hold = np.sort(np.random.default_rng(12345).choice(np.where((p > 0.9) & (M >= 50))[0], 30000, replace=False))
    return hold, f, M, p


if sys.argv[1] == "prep":
    hold, f, M, p = hold_ids()
    off = f["cand_off"].to_numpy().astype(np.int64)
    rows = np.concatenate([np.arange(off[i], off[i] + M[i]) for i in hold])
    X = np.empty((len(rows), len(feats)), np.float32)
    for j, n in enumerate(feats):
        X[:, j] = np.load(ROOT / f"data/feats/train/{n}.npy", mmap_mode="r")[rows]
    np.save(OUT / "c16_HO30_X.npy", X)
    # consistency: the 07 HO (10K of the 30K) rows must equal ours
    mh = np.load(G / "HO_meta.npz"); Xh10 = np.load(G / "HO_X.npy", mmap_mode="r")
    pos = np.searchsorted(hold, mh["fidx"]); assert np.array_equal(hold[pos], mh["fidx"])
    hoff = np.concatenate([[0], np.cumsum(M[hold])[:-1]])
    r = np.concatenate([np.arange(hoff[i], hoff[i] + M[hold][i]) for i in pos[:200]])
    a, b = X[r], np.asarray(Xh10[:len(r)])
    assert np.allclose(np.nan_to_num(a, nan=-7), np.nan_to_num(b, nan=-7)), "HO layout mismatch"
    log(f"prep done: 30K holdout X {X.shape}; 07 HO layout check OK on 200 feeds")
    sys.exit(0)

import lightgbm as lgb
arm, seed = sys.argv[1], int(sys.argv[2])
hold, f, Mall, p = hold_ids()
X = np.load(G / "U_X.npy", mmap_mode="r"); m = dict(np.load(G / "U_meta.npz"))
assert not np.isin(m["fidx"], hold).any(), "U contains holdout feeds"
keep_u = ~np.isin(m["fidx"], hold)  # all True (asserted); kept explicit
nf = len(m["M"]); fid = np.repeat(np.arange(nf), m["M"])
ts = f["timestamp"].to_numpy().astype(np.int64)[m["fidx"]]
day = (ts + 10800) // 86400
import datetime as dt
D12 = (dt.date(2026, 9, 12) - dt.date(1970, 1, 1)).days
prod = m["cls"] == 0
assert np.all((p[m["fidx"]] <= 0.5) == prod), "cls 0 != p_random <= 0.5"
post = prod & (day >= D12)
log(f"U feeds {nf}: production {prod.sum()} (post Sep-12 {post.sum()}, pre {(prod & ~post).sum()}), random-like {(~prod).sum()}")
keep = keep_u.copy(); addcol = None
if arm == "base":
    pass
elif arm == "regime":
    addcol = post[fid].astype(np.float32)
elif arm == "droppre":
    keep &= ~(prod & ~post)
elif arm == "droppost":
    keep &= ~post
else:
    raise ValueError(arm)
rows = keep[fid]
Xt = np.asarray(X)[rows]
if addcol is not None:
    Xt = np.hstack([Xt, addcol[rows][:, None]])
yt = m["y"][rows]; gt = m["M"][keep]
fn_ = feats + (["post12"] if addcol is not None else [])
params = dict(objective="lambdarank", lambdarank_truncation_level=1000, label_gain=[0, 1], learning_rate=0.05,
              num_leaves=63, min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
              lambda_l2=10.0, num_threads=12, verbose=-1, seed=seed, max_bin=255,
              lambdarank_position_bias_regularization=0.0)
ds = lgb.Dataset(Xt, yt, group=gt, feature_name=fn_, free_raw_data=True)
bst = lgb.train(params, ds, 400)
del Xt
log(f"{arm} s{seed}: trained on {len(gt)} feeds {len(yt)} rows")
Xv = np.asarray(np.load(G / "val_X.npy", mmap_mode="r")); Xh = np.load(OUT / "c16_HO30_X.npy")
if addcol is not None:
    Xv = np.hstack([Xv, np.zeros((len(Xv), 1), np.float32)]); Xh = np.hstack([Xh, np.zeros((len(Xh), 1), np.float32)])
pv = bst.predict(Xv, num_threads=12).astype(np.float32); ph = bst.predict(Xh, num_threads=12).astype(np.float32)
np.save(OUT / f"c16_{arm}_s{seed}_val.npy", pv); np.save(OUT / f"c16_{arm}_s{seed}_hold.npy", ph)
log(f"{arm} s{seed}: saved")
