"""Helpers for the history-intent error analysis (fast NDCG, cross-fitting)."""
import os
for v in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "POLARS_MAX_THREADS", "NUMBA_NUM_THREADS"]:
    os.environ[v] = "12"
import sys
sys.path.insert(0, "<repo>/lab/nn_x/errfix_1117/code")
import numpy as np
import polars as pl
import kit

DISC = 1.0 / np.log2(np.arange(2, 400))


class Data:
    def __init__(self):
        self.df = kit.load()
        df = self.df
        self.off, self.sz = kit.feed_layout(df)
        self.nf = len(self.sz)
        self.feed = np.repeat(np.arange(self.nf), self.sz)
        self.y = df["label"].to_numpy().astype(np.float64)
        self.s = df["score"].to_numpy().astype(np.float64)
        self.fold_row = kit.folds(df)
        self.fold = self.fold_row[self.off]
        npos = np.add.reduceat(self.y, self.off)
        self.npos = npos
        cum = np.r_[0, np.cumsum(DISC)]
        self.idcg = cum[npos.astype(int)]
        self.base = self.ndcg(self.s)

    def col(self, c):
        return self.df[c].to_numpy()

    def ndcg(self, s):
        """per-feed NDCG, ties pessimistic"""
        order = np.lexsort((self.y, -s, self.feed))
        pos = np.arange(len(s)) - np.repeat(self.off, self.sz)
        ys = self.y[order]
        g = ys * DISC[pos]
        dcg = np.add.reduceat(g, self.off)
        out = np.where(self.npos > 0, dcg / np.maximum(self.idcg, 1e-12), 1.0)
        return out

    def delta(self, s, mask=None):
        d = self.ndcg(s) - self.base
        if mask is not None:
            d = d[mask]
        return d.mean(), d.std(ddof=1) / np.sqrt(len(d))

    def crossfit(self, make, grid, verbose=False):
        """make(w) -> new score array. choose w on 4 folds (mean delta), evaluate on 5th.
        returns oof mean delta, se, z, chosen ws, and in-sample curve."""
        per_w = {}
        for w in grid:
            per_w[w] = self.ndcg(make(w)) - self.base
        oof = np.zeros(self.nf)
        chosen = []
        for k in range(5):
            tr = self.fold != k
            best = max(grid, key=lambda w: per_w[w][tr].mean())
            chosen.append(best)
            oof[self.fold == k] = per_w[best][self.fold == k]
        m, se = oof.mean(), oof.std(ddof=1) / np.sqrt(self.nf)
        curve = {w: (per_w[w].mean(), per_w[w].std(ddof=1) / np.sqrt(self.nf)) for w in grid}
        if verbose:
            for w in grid:
                print(f"   w={w:+.4g}: in-sample {curve[w][0]:+.5f} (se {curve[w][1]:.5f})")
        return dict(oof=m, se=se, z=m / se if se > 0 else 0, chosen=chosen, curve=curve)


def zfeed(x, off, sz):
    """within-feed z-score"""
    feed = np.repeat(np.arange(len(sz)), sz)
    mu = np.add.reduceat(x, off) / sz
    xc = x - mu[feed]
    var = np.add.reduceat(xc * xc, off) / sz
    return xc / np.sqrt(var[feed] + 1e-12)


def clogit_fit(X, y, off, sz, l2=1e-4, b0=None):
    """Conditional logit (multinomial within feed; each positive one choice).
    X: rows x k. returns coefficients, negative log-lik."""
    from scipy.optimize import minimize
    feed = np.repeat(np.arange(len(sz)), sz)
    npos = np.add.reduceat(y, off)
    keep_feed = npos > 0

    def f(b):
        u = X @ b
        mx = np.maximum.reduceat(u, off)
        e = np.exp(u - mx[feed])
        se = np.add.reduceat(e, off)
        lse = np.log(se) + mx
        ll = (y * u).sum() - (npos * lse)[keep_feed].sum()
        p = e / se[feed]
        g = X.T @ y - X.T @ (p * npos[feed])
        return -ll + l2 * (b @ b), -g + 2 * l2 * b

    b0 = np.zeros(X.shape[1]) if b0 is None else b0
    r = minimize(f, b0, jac=True, method="L-BFGS-B")
    return r.x, r.fun


def clogit_expect(X, b, y, off, sz):
    feed = np.repeat(np.arange(len(sz)), sz)
    npos = np.add.reduceat(y, off)
    u = X @ b
    mx = np.maximum.reduceat(u, off)
    e = np.exp(u - mx[feed])
    p = e / np.add.reduceat(e, off)[feed]
    return p * npos[feed]
