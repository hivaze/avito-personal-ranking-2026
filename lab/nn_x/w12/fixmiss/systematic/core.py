"""Shared machinery for the systematic-miss scan (CPU only).

- user halves A/B (hash of user_id; user-0 feeds by a hash of the feed index)
- one global temperature T fitted on half A (listwise softmax CE over each positive, as in training)
- Plackett-Luce re-draws of the labels from the model's own softmax (same #positives per feed, Gumbel top-k)
- exact per-positive split of 1 - NDCG (k-th best positive vs ideal position k), for actual and re-drawn labels
- conditional-logit offsets (fit on one half, apply to the other) and paired NDCG deltas with pmix views
"""
from __future__ import annotations

import sys
from pathlib import Path

import numba as nb
import numpy as np
import polars as pl
from scipy.optimize import minimize, minimize_scalar

ROOT = Path("<repo>")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
import pmix  # noqa: E402
from metrics import ndcg_per_feed  # noqa: E402

DISC = 1.0 / np.log2(np.arange(2, 400))


def load_table(cols=None):
    return pl.read_parquet(HERE / "data/val_table.parquet", columns=cols)


def layout():
    F = pl.read_parquet(ROOT / "artifacts/data/flat/val/feeds.parquet")
    M = F["M"].to_numpy().astype(np.int64)
    off = F["cand_off"].to_numpy().astype(np.int64)
    return F, M, off


def halves(F: pl.DataFrame) -> np.ndarray:
    """Per-feed half 0 (A) / 1 (B): splitmix64 hash of user_id; guests (user 0) by hash of feed index."""
    u = F["user_id"].to_numpy().astype(np.uint64)
    key = np.where(u == 0, np.arange(len(u), dtype=np.uint64) + np.uint64(10**12), u)
    with np.errstate(over="ignore"):
        z = key + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
    return (z & np.uint64(1)).astype(np.int8)


@nb.njit(cache=True)
def _feed_lse(x, off, M):
    out = np.empty(len(M))
    for f in range(len(M)):
        o = off[f]; m = M[f]
        mx = -1e300
        for i in range(o, o + m):
            if x[i] > mx:
                mx = x[i]
        acc = 0.0
        for i in range(o, o + m):
            acc += np.exp(x[i] - mx)
        out[f] = mx + np.log(acc)
    return out


def softmax_p(x, off, M):
    fid = np.repeat(np.arange(len(M)), M)
    return np.exp(x - _feed_lse(x, off, M)[fid])


def nll(x, y, off, M, feed_mask=None):
    """Sum over positives of -log softmax_feed(x)_i (optionally over a subset of feeds)."""
    fid = np.repeat(np.arange(len(M)), M)
    lse = _feed_lse(x, off, M)
    t = -(x - lse[fid]) * y
    per = np.bincount(fid, t, len(M))
    return per[feed_mask].sum() if feed_mask is not None else per.sum()


def fit_T(s, y, off, M, feed_mask):
    r = minimize_scalar(lambda lt: nll(s / np.exp(lt), y, off, M, feed_mask), bounds=(-3, 2), method="bounded")
    return float(np.exp(r.x))


@nb.njit(cache=True)
def model_rank(s, off, M):
    """0-based position of each row in its feed under descending score (stable by row order)."""
    rk = np.empty(len(s), np.int64)
    for f in range(len(M)):
        o = off[f]; m = M[f]
        order = np.argsort(-s[o:o + m], kind="mergesort")
        for j in range(m):
            rk[o + order[j]] = j
    return rk


@nb.njit(cache=True)
def _pos_loss_feed(ranks, disc):
    k = len(ranks)
    rs = np.sort(ranks)
    idcg = 0.0
    for j in range(k):
        idcg += disc[j]
    out = np.empty(k)
    for j in range(k):
        out[j] = (disc[j] - disc[rs[j]]) / idcg
    return rs, out


@nb.njit(cache=True)
def actual_pos_loss(y, rk, off, M, disc):
    """Per-row loss attribution: for positives, (D(k) - D(rank_k)) / IDCG, k = order among the feed's positives."""
    L = np.zeros(len(y))
    for f in range(len(M)):
        o = off[f]; m = M[f]
        idx = np.empty(m, np.int64); n = 0
        for i in range(o, o + m):
            if y[i] > 0:
                idx[n] = i; n += 1
        if n == 0:
            continue
        ranks = rk[idx[:n]]
        rs, lo = _pos_loss_feed(ranks, disc)
        # map back: sort idx by rank
        ordr = np.argsort(ranks, kind="mergesort")
        for j in range(n):
            L[idx[ordr[j]]] = lo[j]
    return L


@nb.njit(cache=True)
def redraw(x, rk, off, M, npos, R, seed, disc):
    """R Plackett-Luce draws of npos[f] positives per feed from softmax(x). Returns (rows[R, P], loss[R, P])."""
    np.random.seed(seed)
    P = 0
    for f in range(len(M)):
        P += npos[f]
    rows = np.empty((R, P), np.int64); loss = np.empty((R, P))
    for r in range(R):
        c = 0
        for f in range(len(M)):
            o = off[f]; m = M[f]; k = npos[f]
            if k == 0:
                continue
            g = np.empty(m)
            for j in range(m):
                g[j] = x[o + j] - np.log(-np.log(np.random.random() + 1e-300) + 1e-300)
            top = np.argsort(-g)[:k]
            ranks = np.empty(k, np.int64)
            for j in range(k):
                ranks[j] = rk[o + top[j]]
            ordr = np.argsort(ranks, kind="mergesort")
            rs, lo = _pos_loss_feed(ranks, disc)
            for j in range(k):
                rows[r, c + j] = o + top[ordr[j]]
                loss[r, c + j] = lo[j]
            c += k
    return rows, loss


def zfeed(x, off, M):
    fid = np.repeat(np.arange(len(M)), M)
    mu = np.bincount(fid, x, len(M)) / M
    sd = np.sqrt(np.maximum(np.bincount(fid, x * x, len(M)) / M - mu ** 2, 1e-12))
    return (x - mu[fid]) / sd[fid]


# ---------------- conditional-logit offsets (cross-fitted by half) ----------------

def clogit_fit(base, X, y, off, M, feed_mask, l2=1e-3):
    """Fit b in logit = base + X b by minimizing listwise CE over positives of the feeds in feed_mask."""
    fid = np.repeat(np.arange(len(M)), M)
    rmask = feed_mask[fid]
    Xs, bs, ys = X[rmask], base[rmask], y[rmask]
    Ms = M[feed_mask]; offs = np.concatenate([[0], np.cumsum(Ms)[:-1]]).astype(np.int64)
    fids = np.repeat(np.arange(len(Ms)), Ms)
    npos = np.bincount(fids, ys, len(Ms))
    k = X.shape[1]

    def fg(b):
        x = bs + Xs @ b
        lse = _feed_lse(x, offs, Ms)
        p = np.exp(x - lse[fids])
        f = -(ys * (x - lse[fids])).sum() + l2 * len(Ms) * (b @ b)
        # gradient: -sum_pos X_i + sum_feeds npos * sum_i p_i X_i
        g = -(ys @ Xs) + ((npos[fids] * p) @ Xs) + 2 * l2 * len(Ms) * b
        return f, g

    r = minimize(fg, np.zeros(k), jac=True, method="L-BFGS-B")
    return r.x


def crossfit_offsets(base, X, y, off, M, half, l2=1e-3):
    """Fit on A apply to B and vice versa. Returns (new score array, bA, bB)."""
    fid = np.repeat(np.arange(len(M)), M)
    bA = clogit_fit(base, X, y, off, M, half == 0, l2)
    bB = clogit_fit(base, X, y, off, M, half == 1, l2)
    new = base.copy()
    rB = half[fid] == 1
    new[rB] = base[rB] + X[rB] @ bA
    new[~rB] = base[~rB] + X[~rB] @ bB
    return new, bA, bB


def report_delta(nd_new, nd_base, M, half=None, label=""):
    W = pmix.weights()
    d = nd_new - nd_base
    out = {}
    out["val"] = pmix.wmean_se(d)
    out["pub"] = pmix.wmean_se(d, W["public"])
    out["priv"] = pmix.wmean_se(d, W["private"])
    out["M<50"] = pmix.wmean_se(d[M < 50])
    out["M>=50"] = pmix.wmean_se(d[M >= 50])
    if half is not None:
        out["A"] = pmix.wmean_se(d[half == 0]); out["B"] = pmix.wmean_se(d[half == 1])
    out["changed"] = float((np.abs(d) > 1e-12).mean())
    txt = " | ".join(f"{k} {v[0]:+.5f}±{v[1]:.5f}" for k, v in out.items() if k != "changed")
    print(f"{label:40s} {txt} | chg {out['changed']:.2f}", flush=True)
    return out
