"""Fast helpers for the segments analysis (vectorised NDCG, cross-fitting)."""
from __future__ import annotations
import os
for v in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "POLARS_MAX_THREADS", "NUMBA_NUM_THREADS"]:
    os.environ.setdefault(v, "12")
import sys
sys.path.insert(0, "<repo>/lab/nn_x/errfix_1117/code")
import numpy as np
import polars as pl
import kit

_DISC = 1.0 / np.log2(np.arange(2, 400))


class Ctx:
    def __init__(self, df: pl.DataFrame):
        self.df = df
        self.off, self.sz = kit.feed_layout(df)
        self.nf = len(self.sz)
        self.feed = np.repeat(np.arange(self.nf), self.sz)
        self.y = df["label"].to_numpy().astype(np.float64)
        self.npos = np.bincount(self.feed, weights=self.y, minlength=self.nf)
        cs = np.cumsum(_DISC)
        self.idcg = np.where(self.npos > 0, cs[np.maximum(self.npos.astype(int) - 1, 0)], 1.0)
        self.base = df["score"].to_numpy().astype(np.float64)
        self.base_nd = self.ndcg(self.base)
        fo = kit.folds(df)
        self.fold_row = fo
        self.fold = fo[self.off]

    def ndcg(self, s: np.ndarray) -> np.ndarray:
        order = np.lexsort((self.y, -s, self.feed))
        pos = np.arange(len(s)) - self.off[self.feed[order]]
        g = self.y[order] * _DISC[pos]
        dcg = np.bincount(self.feed[order], weights=g, minlength=self.nf)
        out = dcg / self.idcg
        out[self.npos == 0] = 1.0
        return out

    def delta(self, s: np.ndarray, mask_feed: np.ndarray | None = None):
        d = self.ndcg(s) - self.base_nd
        if mask_feed is not None:
            d = d[mask_feed]
        return d

    def stats(self, s):
        d = self.delta(s)
        se = d.std(ddof=1) / np.sqrt(len(d))
        return d.mean(), se, d.mean() / se if se > 0 else 0.0

    def rowify(self, featfeed):
        return featfeed[self.feed]

    def crossfit(self, make_scores, grid):
        """make_scores(param) -> new score array. Select param on 4 folds, evaluate on the 5th.
        Returns (oof_delta_mean, se, z, chosen params per fold, per-feed oof delta)."""
        D = np.stack([self.ndcg(make_scores(p)) - self.base_nd for p in grid])  # (G, nf)
        oof = np.zeros(self.nf)
        chosen = []
        for k in range(5):
            tr = self.fold != k
            te = ~tr
            best = int(np.argmax(D[:, tr].mean(1)))
            chosen.append(grid[best])
            oof[te] = D[best, te]
        se = oof.std(ddof=1) / np.sqrt(self.nf)
        return oof.mean(), se, oof.mean() / se, chosen, D


def zf(x, off, sz, feed):
    """within-feed z-score"""
    x = np.asarray(x, dtype=np.float64)
    n = sz[feed]
    m = np.bincount(feed, weights=x) / sz
    v = np.bincount(feed, weights=(x - m[feed]) ** 2) / sz
    sd = np.sqrt(v)
    sd[sd < 1e-9] = 1.0
    return (x - m[feed]) / sd[feed]
