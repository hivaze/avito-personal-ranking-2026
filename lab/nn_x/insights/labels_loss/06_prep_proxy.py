"""Cache feature matrices for the CPU GBDT loss proxy (07/08). 48 features = solvers/proxy_lib CONTENT + COLLAB
(log1p on count features, as the loss zoo). Sets:
  pool : 80K random-like train feeds (p_random > 0.5, M >= 50, holdout excluded) + shown rank / rank-from-end
  mix  : 120K train feeds drawn uniformly (holdout excluded; ~5% random-like, like the NN's plateau mix) + p, pos
  val  : all 14,423 val feeds;  hold : the 30K train holdout (same reconstruction as nn/train.py)
Writes out/proxy/<set>_{X,y,M,aux}.npy"""
import sys, os, time
import numpy as np
sys.path.insert(0, "<repo>/src")
from common import feed_meta, FLAT, FEATS
exec(open("<repo>/solvers/solvers/proxy_lib.py").read().split("def rows_of")[0].split("from common import *")[1])

OUT = "<repo>/nn_x/insights/labels_loss/out/proxy"
os.makedirs(OUT, exist_ok=True)
NAMES = CONTENT + COLLAB

def rows_of(off, M, fids):
    return np.concatenate([np.arange(off[i], off[i] + M[i]) for i in fids])

def load_X(split, rows):
    X = np.empty((len(rows), len(NAMES)), np.float32)
    for j, n in enumerate(NAMES):
        t0 = time.time()
        x = np.asarray(np.load(FEATS / split / f"{n}.npy", mmap_mode="r")[rows], dtype=np.float32)
        if n in LOG: x = np.log1p(np.maximum(x, 0))
        X[:, j] = x
        print(f"  {split} {n} {time.time()-t0:.1f}s", flush=True)
    return X

def within_rank(posv, Mv):
    n = len(Mv); fid = np.repeat(np.arange(n), Mv)
    o = np.lexsort((posv, fid)); r = np.empty(len(posv), np.int64)
    start = np.repeat(np.concatenate([[0], np.cumsum(Mv)[:-1]]), Mv); r[o] = np.arange(len(posv)) - start
    return r, np.repeat(Mv, Mv) - 1 - r

f, M, off = feed_meta("train")
pr = np.load("<repo>/eda/labels_position/out/train_prandom.npz")
p = np.nan_to_num(pr["p_random"].astype(np.float32), nan=0.0); p[M < 50] = 0
cand = np.where((p > 0.9) & (M >= 50))[0]
hold = np.sort(np.random.default_rng(12345).choice(cand, 30000, replace=False))
ok = np.ones(len(M), bool); ok[hold] = False
rng = np.random.default_rng(2026)
sets = {
    "pool": np.sort(rng.choice(np.where(ok & (p > 0.5))[0], 80000, replace=False)),
    "mix": np.sort(rng.choice(np.where(ok)[0], 120000, replace=False)),
    "hold": hold,
}
y_tr = np.load(FLAT / "train/cand_target.npy", mmap_mode="r"); pos_tr = np.load(FLAT / "train/cand_pos.npy", mmap_mode="r")
for name, fids in sets.items():
    if os.path.exists(f"{OUT}/{name}_X.npy"): continue
    r = rows_of(off, M, fids)
    y = np.asarray(y_tr[r]).astype(np.float32); posv = np.asarray(pos_tr[r]).astype(np.int64)
    rk, rke = within_rank(posv, M[fids])
    np.save(f"{OUT}/{name}_y.npy", y); np.save(f"{OUT}/{name}_M.npy", M[fids].astype(np.int64))
    np.save(f"{OUT}/{name}_aux.npy", np.stack([rk, rke]).astype(np.int32)); np.save(f"{OUT}/{name}_p.npy", p[fids])
    np.save(f"{OUT}/{name}_fidx.npy", fids)
    X = load_X("train", r); np.save(f"{OUT}/{name}_X.npy", X)
    print(name, len(fids), X.shape, flush=True)
fv, Mv, offv = feed_meta("val")
if not os.path.exists(f"{OUT}/val_X.npy"):
    r = np.arange(Mv.sum())
    np.save(f"{OUT}/val_y.npy", np.asarray(np.load(FLAT / "val/cand_target.npy")).astype(np.float32))
    np.save(f"{OUT}/val_M.npy", Mv.astype(np.int64))
    X = load_X("val", r); np.save(f"{OUT}/val_X.npy", X)
    print("val", X.shape)
