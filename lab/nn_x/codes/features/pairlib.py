"""Pair construction shared by 03/04: frequent-item pairs co-occurring in train feeds."""
import sys; sys.path.insert(0, "<repo>/nn_x/codes/features")
from fcommon import *

FLOORS = {"f1": (1/56, 55/56), "f2": (0.0, 63/64), "f3": (1/62, 61/62), "f5": (1/62, 61/62), "f4": (1/64, 63/64)}


def build_pairs(n_feeds=400_000, seed=1, min_item=20, cap=30):
    fidx = train_sample("uniform", n_feeds, seed)
    d = load("train", fidx)
    item = d["item"].astype(np.int64)
    u, inv, cnt = np.unique(item, return_inverse=True, return_counts=True)
    keep = cnt[inv] >= min_item
    rng = np.random.default_rng(0)
    idx = np.flatnonzero(keep)
    feed = d["feed"][idx]
    r = rng.random(len(idx))
    o = np.lexsort((r, feed)); idx = idx[o]; feed = feed[o]
    fstart = np.r_[True, feed[1:] != feed[:-1]]
    rank = np.arange(len(idx)) - np.flatnonzero(fstart)[np.cumsum(fstart) - 1]
    idx = idx[rank < cap]; feed = feed[rank < cap]
    fstart = np.r_[True, feed[1:] != feed[:-1]]
    starts = np.flatnonzero(fstart); lens = np.diff(np.r_[starts, len(idx)])
    A = []; B = []
    for k in range(2, cap + 1):
        fs = starts[lens == k]
        if len(fs) == 0: continue
        ti, tj = np.triu_indices(k, 1)
        A.append((fs[:, None] + ti[None, :]).ravel()); B.append((fs[:, None] + tj[None, :]).ravel())
    A = idx[np.concatenate(A)]; B = idx[np.concatenate(B)]
    ia = item[A]; ib = item[B]
    sw = ia > ib
    A, B = np.where(sw, B, A), np.where(sw, A, B)
    key = item[A] * (item.max() + 1) + item[B]
    ukey, kinv, kcnt = np.unique(key, return_inverse=True, return_counts=True)
    return d, A, B, ukey, kinv, kcnt


def pair_share(d, A, B, kinv, nkeys, k):
    v = d[k]; va = v[A]; vb = v[B]
    lo, hi = FLOORS[k]
    special = (np.abs(va - lo) < 1e-6) | (np.abs(va - hi) < 1e-6) | (va == 0)
    eq = (va == vb) & ~special
    valid = ~((va == vb) & special)
    n_ = np.bincount(kinv[valid], minlength=nkeys); e_ = np.bincount(kinv[valid], eq[valid], minlength=nkeys)
    return e_, n_
