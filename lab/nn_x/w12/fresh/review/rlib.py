"""Reviewer's own helpers (no import from the builder's directory, none from src/ except where stated for a cross-check)."""
import glob

import numba as nb
import numpy as np
import polars as pl

ROOT = "<repo>"
OUT = f"{ROOT}/nn_x/w12/fresh/review/out"
BOUND = 1790497920  # src/pmix.py BOUND (public-like = test feeds before it)
MSK = 10800


def load_raw(cols=("feed_id", "timestamp", "user_id", "item_id")):
    return {sp: pl.concat([pl.read_parquet(f, columns=list(cols)) for f in sorted(glob.glob(f"{ROOT}/data/competition/{sp}/impressions/part-*.parquet"))])
            for sp in ("val", "test")}


@nb.njit(parallel=True, cache=False)
def ndcg_feeds(s, y, off, M):
    """Per-feed NDCG, binary gain, log2 discount, pessimistic ties (a tied negative is ranked above the positive),
    feeds without a positive -> 1. O(M * npos) per feed."""
    nf = len(M)
    out = np.ones(nf)
    for f in nb.prange(nf):
        o = off[f]; m = M[f]
        npos = 0
        for r in range(o, o + m):
            if y[r] > 0:
                npos += 1
        if npos == 0:
            continue
        dcg = 0.0
        for p in range(o, o + m):
            if y[p] <= 0:
                continue
            sp = s[p]; rk = 0
            for r in range(o, o + m):
                if s[r] > sp:
                    rk += 1
                elif s[r] == sp and r != p:
                    if y[r] <= 0 or r < p:
                        rk += 1
            dcg += 1.0 / np.log2(rk + 2.0)
        idcg = 0.0
        for k in range(npos):
            idcg += 1.0 / np.log2(k + 2.0)
        out[f] = dcg / idcg
    return out


@nb.njit(parallel=True, cache=False)
def ndcg_grid(base, x, grid, y, off, M, out):
    """out[g, f] = NDCG of feed f for the score base + grid[g] * x."""
    nf = len(M); ng = len(grid)
    for f in nb.prange(nf):
        o = off[f]; m = M[f]
        npos = 0
        for r in range(o, o + m):
            if y[r] > 0:
                npos += 1
        if npos == 0:
            for g in range(ng):
                out[g, f] = 1.0
            continue
        idcg = 0.0
        for k in range(npos):
            idcg += 1.0 / np.log2(k + 2.0)
        s = np.empty(m)
        for g in range(ng):
            w = grid[g]
            for r in range(m):
                s[r] = base[o + r] + w * x[o + r]
            dcg = 0.0
            for p in range(m):
                if y[o + p] <= 0:
                    continue
                sp = s[p]; rk = 0
                for r in range(m):
                    if s[r] > sp:
                        rk += 1
                    elif s[r] == sp and r != p:
                        if y[o + r] <= 0 or r < p:
                            rk += 1
                dcg += 1.0 / np.log2(rk + 2.0)
            out[g, f] = dcg / idcg


@nb.njit(parallel=True, cache=False)
def feed_forms(off, M, n, r_out, z_out):
    """Within-feed percent rank of n (average ties) minus nothing, and within-feed z of log1p(n) (0 when sd is 0)."""
    for f in nb.prange(len(M)):
        o = off[f]; m = M[f]
        s1 = 0.0; s2 = 0.0
        for k in range(m):
            l = np.log1p(n[o + k]); s1 += l; s2 += l * l
        mu = s1 / m; var = s2 / m - mu * mu
        sd = np.sqrt(var) if var > 1e-12 else 0.0
        for k in range(m):
            v = n[o + k]
            lt = 0; eq = 0
            for j in range(m):
                if n[o + j] < v:
                    lt += 1
                elif n[o + j] == v:
                    eq += 1
            r_out[o + k] = (lt + (eq - 1) / 2.0) / (m - 1) if m > 1 else 0.5
            z_out[o + k] = (np.log1p(v) - mu) / sd if sd > 0 else 0.0


def zfeed(x, fid, M):
    x = np.asarray(x, np.float64)
    nf = len(M)
    mu = np.bincount(fid, x, nf) / M
    sd = np.sqrt(np.bincount(fid, (x - mu[fid]) ** 2, nf) / M)
    return (x - mu[fid]) / (sd[fid] + 1e-9)


class Val:
    def __init__(self):
        f = pl.read_parquet(f"{ROOT}/data/flat/val/feeds.parquet")
        self.feed_id = f["feed_id"].to_numpy(); self.nf = len(f)
        self.M = f["M"].to_numpy().astype(np.int64); self.off = f["cand_off"].to_numpy().astype(np.int64)
        assert (self.off == np.concatenate([[0], np.cumsum(self.M)[:-1]])).all()
        self.ts = f["timestamp"].to_numpy().astype(np.int64); self.uid = f["user_id"].to_numpy().astype(np.int64)
        self.fid = np.repeat(np.arange(self.nf), self.M); self.N = len(self.fid)
        self.item = np.load(f"{ROOT}/data/flat/val/cand_item_id.npy").astype(np.int64)
        self.y = np.load(f"{ROOT}/data/flat/val/cand_target.npy").astype(np.float64)
        ft = np.load(f"{ROOT}/preds/ens_FD16P20kd_ft_val.npy").astype(np.float64)
        to = np.load(f"{ROOT}/preds/ens_FD16P20kd_to_val.npy").astype(np.float64)
        self.b = zfeed(0.5 * zfeed(ft, self.fid, self.M) + 0.5 * zfeed(to, self.fid, self.M), self.fid, self.M)
        self.cold = np.load(f"{ROOT}/data/feats/val/tte_it_imp.npy") == 0
        self.day = (self.ts + MSK) // 86400
        self.base = ndcg_feeds(self.b, self.y, self.off, self.M)

    def folds(self, seed, u0_one_group=False, nfold=5):
        """My folds: groups = users (uid-0 feeds singletons unless u0_one_group), sorted, shuffled, array_split."""
        g = self.uid.copy()
        if not u0_one_group:
            z = self.uid == 0
            g[z] = -1 - np.arange(z.sum())
        ug = np.unique(g)
        rng = np.random.default_rng(seed)
        sh = rng.permutation(ug)
        fold_of = {}
        for k, part in enumerate(np.array_split(sh, nfold)):
            for u in part:
                fold_of[int(u)] = k
        return np.array([fold_of[int(u)] for u in g], np.int64)

    def forms(self, n):
        n = np.ascontiguousarray(n, np.float64)
        r = np.empty(self.N); z = np.empty(self.N)
        feed_forms(self.off, self.M, n, r, z)
        return r, z


def psi(a, b, edges=np.array([-0.5, 0.5, 1.5, 2.5, 4.5, 9.5, 19.5, 1e12])):
    pa = np.histogram(a, edges)[0] / max(len(a), 1); pb = np.histogram(b, edges)[0] / max(len(b), 1)
    pa, pb = np.clip(pa, 1e-6, None), np.clip(pb, 1e-6, None)
    return float(np.sum((pa - pb) * np.log(pa / pb)))
