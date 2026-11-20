"""Helpers for the popularity/freshness lens: fast NDCG, conditional-logit residuals, cross-fitting."""
import os
import sys

for v in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "POLARS_MAX_THREADS", "NUMBA_NUM_THREADS"]:
    os.environ[v] = "12"
sys.path.insert(0, "<repo>/lab/nn_x/errfix_1117/code")
import kit  # noqa: E402
import numba  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

DISC = 1.0 / np.log2(np.arange(2, 400))


@numba.njit(cache=True)
def _ndcg(scores, labels, off, sz, disc):
    out = np.empty(len(sz))
    for k in range(len(sz)):
        o = off[k]
        m = sz[k]
        s = scores[o:o + m]
        y = labels[o:o + m]
        npos = 0
        for i in range(m):
            npos += y[i]
        if npos == 0:
            out[k] = 1.0
            continue
        ideal = 0.0
        for i in range(npos):
            ideal += disc[i]
        # pessimistic ties: for each positive, rank = 1 + #(s_j > s_i) + #(s_j == s_i and y_j == 0)
        # positives tied among themselves: order arbitrary among positives, but they occupy consecutive slots
        # compute exactly with a sort: key (-s, y)
        idx = np.argsort(-s, kind="mergesort")
        # within ties, put negatives first
        dcg = 0.0
        i = 0
        while i < m:
            j = i
            while j + 1 < m and s[idx[j + 1]] == s[idx[i]]:
                j += 1
            # block i..j tie
            nb = j - i + 1
            npb = 0
            for t in range(i, j + 1):
                npb += y[idx[t]]
            for t in range(nb - npb, nb):
                dcg += disc[i + t]
            i = j + 1
        out[k] = dcg / ideal
    return out


class Data:
    def __init__(self, columns=None):
        self.df = kit.load(columns)
        self.off, self.sz = kit.feed_layout(self.df)
        self.y = self.df["label"].to_numpy().astype(np.int64)
        self.s = self.df["score"].to_numpy().astype(np.float64)
        self.fold_row = kit.folds(self.df)
        self.fold_feed = self.fold_row[self.off]
        self.feed_of_row = np.repeat(np.arange(len(self.sz)), self.sz)
        self.base = self.ndcg(self.s)

    def col(self, c):
        return self.df[c].to_numpy().astype(np.float64)

    def ndcg(self, scores):
        return _ndcg(np.asarray(scores, dtype=np.float64), self.y, self.off, self.sz, DISC)

    def delta(self, scores, feeds=None):
        d = self.ndcg(scores) - self.base
        if feeds is not None:
            d = d[feeds]
        return d.mean(), d.std(ddof=1) / np.sqrt(len(d))

    def feedwise(self, x, fn):
        """apply fn (numpy reduce like np.max) per feed, broadcast back to rows"""
        r = np.empty(len(self.sz))
        for k, (o, m) in enumerate(zip(self.off, self.sz)):
            r[k] = fn(x[o:o + m])
        return r[self.feed_of_row]

    def within_rank(self, x):
        """rank (1 = highest) of x within each feed, ties by order"""
        r = np.empty(len(x))
        for o, m in zip(self.off, self.sz):
            idx = np.argsort(-x[o:o + m], kind="stable")
            rr = np.empty(m)
            rr[idx] = np.arange(1, m + 1)
            r[o:o + m] = rr
        return r

    def crossfit(self, make, grid, k=5, verbose=False):
        """make(param) -> new scores (all rows). Choose param by in-fold mean delta on 4 folds, eval on 5th.

        Returns dict with crossfit delta (mean over all feeds of OOF deltas), se, z, chosen params,
        and the full in-sample curve."""
        per_param = []
        for p in grid:
            per_param.append(self.ndcg(make(p)) - self.base)
        D = np.array(per_param)  # P x F
        oof = np.empty(D.shape[1])
        chosen = []
        for f in range(k):
            tr = self.fold_feed != f
            te = ~tr
            j = int(np.argmax(D[:, tr].mean(1)))
            chosen.append(grid[j])
            oof[te] = D[j, te]
        res = dict(cf=oof.mean(), se=oof.std(ddof=1) / np.sqrt(len(oof)), chosen=chosen,
                   insample=[(g, float(D[i].mean()), float(D[i].std(ddof=1) / np.sqrt(D.shape[1])))
                             for i, g in enumerate(grid)])
        res["z"] = res["cf"] / res["se"] if res["se"] > 0 else 0.0
        if verbose:
            print("crossfit %.5f se %.5f z %.2f chosen %s" % (res["cf"], res["se"], res["z"], chosen))
            for g, m, s in res["insample"]:
                print("   ", g, "%.5f (%.5f)" % (m, s))
        return res


def softmax_p(s, off, sz, npos, a=1.0):
    p = np.empty(len(s))
    for o, m, n in zip(off, sz, npos):
        e = np.exp(a * (s[o:o + m] - s[o:o + m].max()))
        p[o:o + m] = n * e / e.sum()
    return p


def fit_temp(D, s):
    npos = np.add.reduceat(D.y, D.off)
    best = None
    for a in np.arange(0.6, 2.01, 0.05):
        p = softmax_p(s, D.off, D.sz, np.ones_like(npos), a)
        # conditional log lik of positives (multinomial approx)
        ll = (D.y * np.log(np.maximum(p, 1e-12))).sum()
        if best is None or ll > best[1]:
            best = (a, ll)
    return best
