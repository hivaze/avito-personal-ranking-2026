"""Shared helpers for the deep_dive analysis (fast vectorised NDCG, cross-fitting)."""
import os
for v in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "POLARS_MAX_THREADS", "NUMBA_NUM_THREADS"]:
    os.environ[v] = "12"
import sys
sys.path.insert(0, "<repo>/lab/nn_x/errfix_1117/code")
import numpy as np
import polars as pl
import kit

_DISC = 1.0 / np.log2(np.arange(2, 400))


class Ctx:
    """Pre-computed layout for fast NDCG of many score vectors."""

    def __init__(self, df):
        self.df = df
        self.off, self.sz = kit.feed_layout(df)
        self.nf = len(self.sz)
        self.fid = np.repeat(np.arange(self.nf), self.sz)
        self.y = df["label"].to_numpy().astype(np.float64)
        npos = np.bincount(self.fid, weights=self.y, minlength=self.nf)
        self.npos = npos
        cs = np.r_[0, np.cumsum(_DISC)]
        self.idcg = cs[npos.astype(int)]
        self.base = df["score"].to_numpy().astype(np.float64)
        self.base_ndcg = self.ndcg(self.base)
        self.folds_row = kit.folds(df)
        self.folds_feed = self.folds_row[self.off]

    def ranks(self, s):
        # rank within feed, ties pessimistic (negatives first): sort by fid, -s, y
        order = np.lexsort((self.y, -s, self.fid))
        pos = np.empty(len(s), np.int64)
        pos[order] = np.arange(len(s))
        return pos - self.off[self.fid] + 1  # 1-based

    def ndcg(self, s):
        r = self.ranks(np.asarray(s, np.float64))
        dcg = np.bincount(self.fid, weights=self.y * _DISC[r - 1], minlength=self.nf)
        out = np.where(self.npos > 0, dcg / np.maximum(self.idcg, 1e-12), 1.0)
        return out

    def delta(self, s, mask_feed=None):
        d = self.ndcg(s) - self.base_ndcg
        if mask_feed is not None:
            d = d[mask_feed]
        return d.mean(), d.std(ddof=1) / np.sqrt(len(d))

    def feed_delta(self, s):
        return self.ndcg(s) - self.base_ndcg

    def crossfit(self, make_scores, grid, k=5, verbose=False):
        """make_scores(param) -> score vector. Choose param on 4 folds, evaluate on the 5th.
        Returns (cf_delta, cf_se, cf_z, chosen params per fold, per-feed oof deltas, in-sample best)."""
        fd = {}
        for p in grid:
            fd[p] = self.feed_delta(make_scores(p))
        oof = np.zeros(self.nf)
        chosen = []
        for f in range(k):
            tr = self.folds_feed != f
            te = ~tr
            best = max(grid, key=lambda p: fd[p][tr].mean())
            chosen.append(best)
            oof[te] = fd[best][te]
        m = oof.mean(); se = oof.std(ddof=1) / np.sqrt(self.nf)
        ins = max(grid, key=lambda p: fd[p].mean())
        if verbose:
            for p in grid:
                print("   ", p, round(fd[p].mean(), 5), round(fd[p].std(ddof=1) / np.sqrt(self.nf), 5))
        return dict(cf=m, se=se, z=m / se if se > 0 else 0, chosen=chosen, ins_best=ins,
                    ins_delta=fd[ins].mean(), oof=oof, n_variants=len(grid))


def load():
    return kit.load()
