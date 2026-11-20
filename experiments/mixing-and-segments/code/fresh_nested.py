# The nested read used to judge a candidate input on CPU before any training: a per-feed
# conditional logit (listwise softmax) on top of the final pool score, with and without the new
# columns, fitted on four user-grouped folds and scored on the fifth, 5 folds x 3 seeds.
# How it was run: imported by s2_auc.py, s3_nested.py and s4_placebo.py of
# lab/nn_x/w12/fresh/pregate; s3_nested.py was run as in its docstring (CPU, 12 threads, 579 s).
# What it produced: nested gain of the strictly-earlier counts over the pool score -0.00044 +-
# 0.00015 in this family (results/fresh_s3_nested.log); an independent re-run with a direct NDCG
# read gave -0.00004 +- 0.00020 (results/fresh_check_nested.log).
# Kept as it ran: paths and names refer to the lab tree (see lab/ and docs/glossary.md).
# Original: lab/nn_x/w12/fresh/pregate/nested.py
"""Nested residual read of the fresh-exposure forms on val (DECLARATION.txt section C). Library used by s2 / s3 / s4.

PRIMARY family: per-feed conditional logit (listwise softmax, multi-positive), L2 = 10, L-BFGS-B, user-grouped 5 folds x 3 seeds.
"""
import sys

import numba as nb
import numpy as np
import polars as pl
from scipy.optimize import minimize

sys.path.insert(0, "<repo>/nn_x/w12/fresh/pregate")
sys.path.insert(0, "<repo>/src")
from lib import FEATS, FLAT, OUT, ROOT, WIN, forms
from metrics import ndcg_per_feed

LAMBDA = 10.0
ARMS = {"A": ["a", "a24"], "B": ["b", "c", "d"], "ALL": ["a", "a24", "b", "c", "d"], "L": ["e"]}
FOLD_SEEDS = (0, 1, 2)
NFOLD = 5


def zf(x, fid, M, nf):
    """Within-feed z exactly as src/stack_val.feed_rank_z[:, 1]: (v - mean) / (std + 1e-9), float32 input."""
    x = np.asarray(x, np.float32).astype(np.float64)
    mu = np.bincount(fid, x, nf) / M
    var = np.bincount(fid, (x - mu[fid]) ** 2, nf) / M
    return (x - mu[fid]) / (np.sqrt(var)[fid] + 1e-9)


class Val:
    """Val rows: pool score, labels, feed structure, cold flag, slices."""

    def __init__(self):
        f = pl.read_parquet(f"{FLAT}/val/feeds.parquet")
        self.nf = len(f)
        self.M = f["M"].to_numpy().astype(np.int64); self.off = f["cand_off"].to_numpy().astype(np.int64)
        self.ts = f["timestamp"].to_numpy().astype(np.int64); self.uid = f["user_id"].to_numpy().astype(np.int64)
        self.fid = np.repeat(np.arange(self.nf), self.M); self.N = len(self.fid)
        self.y = np.load(f"{FLAT}/val/cand_target.npy").astype(np.float64)
        ft = np.load(f"{ROOT}/preds/ens_FD16P20kd_ft_val.npy"); to = np.load(f"{ROOT}/preds/ens_FD16P20kd_to_val.npy")
        z = lambda x: zf(x, self.fid, self.M, self.nf)
        self.b = z(0.5 * z(ft) + 0.5 * z(to))
        self.cold = (np.load(f"{FEATS}/val/tte_it_imp.npy") == 0)
        self.xp = np.load(f"{FEATS}/val/xp_imp_c50.npy").astype(np.float64)
        lnM = np.log(self.M[self.fid].astype(np.float64)); self.lm = lnM - lnM.mean()
        self.npos = np.bincount(self.fid, self.y, self.nf)
        self.day = (self.ts + 10800) // 86400
        self.cold_share = np.bincount(self.fid, self.cold, self.nf) / self.M
        self.base_ndcg = ndcg_per_feed(self.b, self.y, self.off, self.M)

    def control(self):
        c = self.cold.astype(np.float64)
        X = np.stack([self.b * self.lm, c, c * self.b, c * self.lm], 1)
        return X, ["b*lm", "cold", "cold*b", "cold*lm"]

    def window_cols(self, n, tag):
        """8 columns of one window from its raw count n (float array over val rows)."""
        c = self.cold.astype(np.float64)
        r, z = forms(self.off, self.M, n)
        l = np.log1p(n); i = (n >= 1).astype(np.float64)
        X = np.stack([np.minimum(n, 50.0), l, r - 0.5, z, i, c * i, l * c, l * self.lm], 1)
        return X, [f"{tag}:{k}" for k in ("rawc50", "log1p", "rank", "z", "ind", "cold&shown", "log1p*cold", "log1p*lm")]

    def arm_matrix(self, counts, arm):
        Xc, names = self.control()
        cols = [Xc]
        for w in ARMS[arm]:
            X, nm = self.window_cols(counts[:, WIN.index(w)].astype(np.float64), w)
            cols.append(X); names = names + nm
        return np.hstack(cols), names

    def folds(self, seed):
        g = np.where(self.uid == 0, -1 - np.arange(self.nf), self.uid)
        ug, inv = np.unique(g, return_inverse=True)
        perm = np.random.default_rng(seed).permutation(len(ug))
        return (perm % NFOLD)[inv]


def scale(X):
    sd = X.std(0)
    sd[sd < 1e-12] = 1.0
    return X / sd, sd


@nb.njit(nogil=True, fastmath=True, cache=False)
def _fg_nb(th, b, X, offs, npos, yb, yX, lam, g):
    """Same objective / gradient as Fitter.fit's numpy closure; single thread, releases the GIL (thread-pool fits)."""
    k = X.shape[1]; nfe = len(npos)
    loss = -th[0] * yb
    g[0] = -yb
    for j in range(k):
        loss += -yX[j] * th[1 + j] + 0.5 * lam * th[1 + j] * th[1 + j]
        g[1 + j] = -yX[j] + lam * th[1 + j]
    mm = 0
    for f in range(nfe):
        mm = max(mm, offs[f + 1] - offs[f])
    s = np.empty(mm)
    for f in range(nfe):
        o0 = offs[f]; o1 = offs[f + 1]
        m = -1e300
        for r in range(o0, o1):
            v = th[0] * b[r]
            for j in range(k):
                v += X[r, j] * th[1 + j]
            s[r - o0] = v
            if v > m:
                m = v
        se = 0.0
        for r in range(o0, o1):
            e = np.exp(s[r - o0] - m); s[r - o0] = e; se += e
        loss += npos[f] * (m + np.log(se))
        c = npos[f] / se
        for r in range(o0, o1):
            pr = s[r - o0] * c
            g[0] += pr * b[r]
            for j in range(k):
                g[1 + j] += pr * X[r, j]
    return loss


class Fitter:
    """Conditional logit on a fixed set of training rows (feeds with >= 1 positive only; other feeds add nothing to the loss)."""

    def __init__(self, V, train_feed_mask):
        keep_f = train_feed_mask & (V.npos > 0)
        self.rows = np.flatnonzero(keep_f[V.fid])
        Mk = V.M[keep_f]
        self.offs = np.concatenate([[0], np.cumsum(Mk)[:-1]]).astype(np.int64)
        self.fidk = np.repeat(np.arange(len(Mk)), Mk)
        self.y = V.y[self.rows]; self.npos = V.npos[keep_f]
        self.b = V.b[self.rows]
        self.ysum_b = float(self.y @ self.b)

    def fit(self, X, lam=LAMBDA, w0=None, maxiter=300, use_nb=False):
        """X: scaled columns over ALL val rows (b excluded). Returns (beta, w).
        use_nb: the numba single-thread GIL-free objective (same function; used by the thread-pool placebo runs)."""
        Xt = np.ascontiguousarray(X[self.rows]) if X is not None else np.zeros((len(self.rows), 0))
        k = Xt.shape[1]
        yX = self.y @ Xt
        offs, fidk, npos, b = self.offs, self.fidk, self.npos, self.b
        if use_nb:
            offs_ext = np.concatenate([offs, [len(b)]]).astype(np.int64)
            yb = self.ysum_b

            def fgn(th):
                g = np.empty(k + 1)
                return _fg_nb(th, b, Xt, offs_ext, npos, yb, yX, lam, g), g

            th0 = np.zeros(k + 1); th0[0] = 1.0
            r = minimize(fgn, th0, jac=True, method="L-BFGS-B", options=dict(maxiter=maxiter, maxfun=maxiter * 2, gtol=1e-6, ftol=1e-12))
            return r.x, r

        def fg(th):
            s = th[0] * b + (Xt @ th[1:] if k else 0.0)
            m = np.maximum.reduceat(s, offs)
            e = np.exp(s - m[fidk])
            se = np.add.reduceat(e, offs)
            lse = m + np.log(se)
            loss = -(th[0] * self.ysum_b + yX @ th[1:]) + npos @ lse + 0.5 * lam * (th[1:] @ th[1:])
            p = e / se[fidk] * npos[fidk]
            g = np.empty(k + 1)
            g[0] = p @ b - self.ysum_b
            g[1:] = p @ Xt - yX + lam * th[1:]
            return loss, g

        th0 = np.zeros(k + 1); th0[0] = 1.0
        if w0 is not None:
            th0[:len(w0)] = w0
        r = minimize(fg, th0, jac=True, method="L-BFGS-B", options=dict(maxiter=maxiter, maxfun=maxiter * 2, gtol=1e-6, ftol=1e-12))
        return r.x, r


def nested_scores(V, mats, seeds=FOLD_SEEDS, verbose=False, warm=None, workers=1, use_nb=False):
    """mats: {name: scaled matrix over val rows or None (= b only)}. Returns {name: [per-seed per-feed NDCG]} and thetas.
    workers > 1: the (fold, arm) fits of one fold seed run in a thread pool (set the BLAS threads to 1 in the environment;
    numpy releases the GIL in the matrix products / ufuncs, so the 12-thread budget is used by 12 single-thread fits)."""
    from concurrent.futures import ThreadPoolExecutor
    out = {k: [] for k in mats}; thetas = {k: [] for k in mats}; iters = []
    for sd in seeds:
        fold = V.folds(sd)
        sc = {k: np.zeros(V.N) for k in mats}
        fits = [Fitter(V, fold != kf) for kf in range(NFOLD)]

        def job(args):
            kf, k = args
            th, r = fits[kf].fit(mats[k], use_nb=use_nb)
            return kf, k, th, r.nit

        jobs = [(kf, k) for kf in range(NFOLD) for k in mats]
        if workers > 1:
            with ThreadPoolExecutor(workers) as ex:
                done = list(ex.map(job, jobs))
        else:
            done = [job(j) for j in jobs]
        for kf, k, th, nit in done:
            te_rows = (fold == kf)[V.fid]
            X = mats[k]
            thetas[k].append(th); iters.append(nit)
            sc[k][te_rows] = th[0] * V.b[te_rows] + (X[te_rows] @ th[1:] if X is not None else 0.0)
        for k in mats:
            out[k].append(ndcg_per_feed(sc[k], V.y, V.off, V.M))
        if verbose:
            print(f"  fold seed {sd}: " + " | ".join(f"{k} {out[k][-1].mean():.5f}" for k in mats) + f" | max nit {max(iters)}", flush=True)
    return out, thetas


def gain(arm_list, ctl_list):
    """Per-feed d_f = mean over fold seeds of (arm - control); returns d_f, GAIN, SE, z."""
    d = np.mean([a - c for a, c in zip(arm_list, ctl_list)], 0)
    g = d.mean(); se = d.std() / np.sqrt(len(d))
    return d, float(g), float(se), float(g / se if se > 0 else 0.0)


def wstat(d, w):
    m = np.average(d, weights=w); neff = w.sum() ** 2 / (w ** 2).sum()
    return float(m), float(np.sqrt(np.average((d - m) ** 2, weights=w) / neff))
