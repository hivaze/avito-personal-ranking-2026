"""Helpers for the head-of-list analysis (fast NDCG, cross-fitting)."""
import os
for v in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "POLARS_MAX_THREADS", "NUMBA_NUM_THREADS"]:
    os.environ[v] = "12"
import sys
sys.path.insert(0, "<repo>/lab/nn_x/errfix_1117/code")
import numpy as np
import polars as pl
import numba as nb
import kit

_D = 1.0 / np.log2(np.arange(2, 302))


@nb.njit(cache=True)
def _ndcg(s, y, off, sz, D):
    out = np.empty(len(sz))
    for k in range(len(sz)):
        o = off[k]; m = sz[k]
        ss = s[o:o + m]; yy = y[o:o + m]
        # sort by -s, then label asc (negatives first in ties)
        key = np.empty(m)
        for i in range(m):
            key[i] = -ss[i] * 1.0
        order = np.argsort(key, kind="mergesort")
        # handle ties pessimistic: within equal key, negatives first
        dcg = 0.0
        i = 0
        pos = 0
        while i < m:
            j = i
            while j + 1 < m and key[order[j + 1]] == key[order[i]]:
                j += 1
            # block i..j tie: negatives first
            npos_blk = 0
            for t in range(i, j + 1):
                npos_blk += yy[order[t]]
            nneg = (j - i + 1) - npos_blk
            for t in range(npos_blk):
                dcg += D[i + nneg + t]
            i = j + 1
        npos = 0
        for i in range(m):
            npos += yy[i]
        if npos == 0:
            out[k] = 1.0
            continue
        idcg = 0.0
        for i in range(npos):
            idcg += D[i]
        out[k] = dcg / idcg
    return out


class Ctx:
    def __init__(self, df=None):
        self.df = kit.load() if df is None else df
        self.off, self.sz = kit.feed_layout(self.df)
        self.y = self.df["label"].to_numpy().astype(np.int64)
        self.score = self.df["score"].to_numpy().astype(np.float64)
        self.base = _ndcg(self.score, self.y, self.off, self.sz, _D)
        self.fold_row = kit.folds(self.df)
        self.fold_feed = self.fold_row[self.off]
        self.feed_of_row = np.repeat(np.arange(len(self.sz)), self.sz)

    def col(self, c):
        return self.df[c].to_numpy()

    def nd(self, s):
        return _ndcg(np.asarray(s, dtype=np.float64), self.y, self.off, self.sz, _D)

    def delta(self, s, mask_feed=None):
        d = self.nd(s) - self.base
        if mask_feed is not None:
            d = d[mask_feed]
        return d.mean(), d.std(ddof=1) / np.sqrt(len(d)), d

    def rep(self, s, name=""):
        m, se, d = self.delta(s)
        ch = (d != 0).mean()
        print(f"{name:40s} delta={m:+.5f} se={se:.5f} z={m/se if se>0 else 0:+.2f} feeds_changed_ndcg={ch:.3f}")
        return m, se

    def crossfit(self, make_scores, grid, verbose=False):
        """make_scores(param) -> new score array. Choose param on 4 folds (mean delta), evaluate on 5th.
        Returns crossfit mean, se, chosen params per fold, and per-feed oof deltas."""
        D = {}
        for p in grid:
            D[p] = self.nd(make_scores(p)) - self.base
        oof = np.zeros(len(self.sz))
        chosen = []
        for f in range(5):
            tr = self.fold_feed != f
            te = ~tr
            best = max(grid, key=lambda p: D[p][tr].mean())
            chosen.append(best)
            oof[te] = D[best][te]
        m = oof.mean(); se = oof.std(ddof=1) / np.sqrt(len(oof))
        if verbose:
            for p in grid:
                print("   ", p, f"{D[p].mean():+.5f}")
        return m, se, chosen, oof, D
