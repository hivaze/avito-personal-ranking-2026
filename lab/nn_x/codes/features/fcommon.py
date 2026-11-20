"""Shared loaders for the f1..f5 / target decoding study (nn_x/codes/features).
Train is sampled (seeded) because it has 246.7M candidate rows."""
import os
for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "POLARS_MAX_THREADS", "NUMBA_NUM_THREADS"]:
    os.environ.setdefault(k, "12")
import numpy as np, polars as pl
from pathlib import Path

ROOT = Path("<repo>")
FLAT = ROOT / "data/flat"
FEATS = ROOT / "data/feats"
OUT = ROOT / "nn_x/codes/features/out"
OUT.mkdir(parents=True, exist_ok=True)
FS = ["f1", "f2", "f3", "f4", "f5"]


def feeds(split):
    f = pl.read_parquet(FLAT / split / "feeds.parquet")
    if split == "train":
        z = np.load(ROOT / "eda/labels_position/out/train_prandom.npz")
        assert (z["fidx"] == f["fidx"].to_numpy()).all()
        f = f.with_columns(pl.Series("p_random", z["p_random"]))
    return f


def train_sample(kind="uniform", n=200_000, seed=0):
    """fidx of a train feed sample. kind: uniform | rl (random-like P>0.9 & M>=50) | prod (P<0.05 & M>=50)"""
    f = feeds("train")
    rng = np.random.default_rng(seed)
    M = f["M"].to_numpy(); p = f["p_random"].to_numpy()
    if kind == "uniform":
        pool = np.arange(len(f))
    elif kind == "rl":
        pool = np.where((p > 0.9) & (M >= 50))[0]
    elif kind == "prod":
        pool = np.where((p < 0.05) & (M >= 50))[0]
    else:
        raise ValueError(kind)
    n = min(n, len(pool))
    return np.sort(rng.choice(pool, n, replace=False))


def rows_of(split, fidx):
    f = feeds(split)
    off = f["cand_off"].to_numpy(); M = f["M"].to_numpy()
    starts = off[fidx]; lens = M[fidx]
    rows = np.repeat(starts - np.cumsum(np.r_[0, lens[:-1]]), lens) + np.arange(lens.sum())
    return rows


def load(split, fidx=None, extra_cand=(), extra_feats=()):
    """Return dict of candidate-level arrays (rows of the selected feeds) plus feed-level arrays broadcast.
    Keys: fidx(row->feed index), f1..f5, cat, item, target, pos (train), user, ts, M, H, p_random (train), plus extras."""
    f = feeds(split)
    if fidx is None:
        fidx = np.arange(len(f))
    rows = rows_of(split, fidx)
    d = {}
    M = f["M"].to_numpy()[fidx]
    d["feed"] = np.repeat(np.arange(len(fidx)), M)  # local feed id 0..n-1
    d["gfidx"] = np.repeat(fidx, M)
    for k in FS:
        d[k] = np.asarray(np.load(FLAT / split / f"cand_{k}.npy", mmap_mode="r")[rows])
    d["cat"] = np.asarray(np.load(FLAT / split / "cand_category_id.npy", mmap_mode="r")[rows]).astype(np.int16)
    d["item"] = np.asarray(np.load(FLAT / split / "cand_item_id.npy", mmap_mode="r")[rows])
    d["emb_idx"] = np.asarray(np.load(FLAT / split / "cand_emb_idx.npy", mmap_mode="r")[rows])
    if split != "test":
        d["target"] = np.asarray(np.load(FLAT / split / "cand_target.npy", mmap_mode="r")[rows]).astype(np.int8)
    if split == "train":
        d["pos"] = np.asarray(np.load(FLAT / split / "cand_pos.npy", mmap_mode="r")[rows])
        d["p_random"] = np.repeat(f["p_random"].to_numpy()[fidx], M)
    for c in ["user_id", "timestamp", "M", "H"]:
        d[c] = np.repeat(f[c].to_numpy()[fidx], M)
    for c in extra_cand:
        d[c] = np.asarray(np.load(FLAT / split / f"cand_{c}.npy", mmap_mode="r")[rows])
    for c in extra_feats:
        d[c] = np.asarray(np.load(FEATS / split / f"{c}.npy", mmap_mode="r")[rows]).astype(np.float32)
    d["n_feeds"] = len(fidx)
    d["rows"] = rows
    return d


def feed_auc(score, y, feed):
    """Mean per-feed AUC (feeds with >=1 pos and >=1 neg), ties = 0.5. Returns (mean, se, n_feeds)."""
    o = np.lexsort((score, feed))
    s = score[o]; yy = y[o].astype(np.float64); g = feed[o]
    # average ranks within feed with ties
    n = len(s)
    brk = np.r_[True, (g[1:] != g[:-1]) | (s[1:] != s[:-1])]
    grp = np.cumsum(brk) - 1
    first = np.flatnonzero(brk)
    cnt = np.diff(np.r_[first, n])
    # rank within feed: position index - feed start
    fbrk = np.r_[True, g[1:] != g[:-1]]
    fstart = np.flatnonzero(fbrk)
    fid = np.cumsum(fbrk) - 1
    pos_in_feed = np.arange(n) - fstart[fid]
    avg_rank = (pos_in_feed[first] + (cnt - 1) / 2.0 + 1)[grp]
    npos = np.bincount(fid, yy)
    ntot = np.bincount(fid)
    nneg = ntot - npos
    rsum = np.bincount(fid, avg_rank * yy)
    ok = (npos > 0) & (nneg > 0)
    auc = (rsum[ok] - npos[ok] * (npos[ok] + 1) / 2) / (npos[ok] * nneg[ok])
    return auc.mean(), auc.std() / np.sqrt(ok.sum()), int(ok.sum())


def eta2(x, groups):
    """Share of variance of x explained by group means."""
    x = x.astype(np.float64)
    _, inv = np.unique(groups, return_inverse=True)
    s = np.bincount(inv, x); c = np.bincount(inv)
    m = s / c
    resid = x - m[inv]
    return 1 - resid.var() / x.var()


def eta2_multi(x, keys, min_count=1):
    """eta^2 for a composite key (tuple of int arrays), restricted to groups with >= min_count rows."""
    k = keys[0].astype(np.int64)
    for a in keys[1:]:
        _, a2 = np.unique(a, return_inverse=True)
        k = k * (a2.max() + 1) + a2
    _, inv = np.unique(k, return_inverse=True)
    c = np.bincount(inv)
    sel = c[inv] >= min_count
    x = x[sel].astype(np.float64); inv = inv[sel]
    _, inv = np.unique(inv, return_inverse=True)
    s = np.bincount(inv, x); c = np.bincount(inv)
    resid = x - (s / c)[inv]
    return 1 - resid.var() / x.var(), sel.mean()
