"""Shared helpers for the noise quantification (CPU only).

Val layout, labels, best-file score (0.5*ft + 0.5*to of ens_FD16P20_m04_02kd), exact NDCG (pessimistic ties),
user halves (same splitmix64 hash as lab/nn_x/w12/fixmiss/systematic/core.py), temperature fit on half A,
and a numba Plackett-Luce re-draw of labels (Gumbel top-K, K = the feed's actual number of positives) that returns
the per-feed NDCG of every draw under the model's own ranking.
"""
from __future__ import annotations

import os
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "12")
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numba as nb
import numpy as np
import polars as pl
from scipy.optimize import minimize_scalar

ROOT = Path("<repo>")
HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)
PRED = ROOT / "artifacts/predictions"
FLAT = ROOT / "artifacts/data/flat"
DISC = 1.0 / np.log2(np.arange(2, 402, dtype=np.float64))
N_PUB, N_PRIV = 8562, 20033


def layout(split="val"):
    F = pl.read_parquet(FLAT / split / "feeds.parquet")
    return F, F["M"].to_numpy().astype(np.int64), F["cand_off"].to_numpy().astype(np.int64)


def labels():
    return np.load(FLAT / "val/cand_target.npy").astype(np.float64)


def pool_score(key):
    """Final file score = 0.5 * fine-tuned branch + 0.5 * train-only branch (as in the ensemble recipe)."""
    ft = np.load(PRED / f"ens_{key}_ft_val.npy").astype(np.float64)
    to = np.load(PRED / f"ens_{key}_to_val.npy").astype(np.float64)
    return 0.5 * ft + 0.5 * to


BEST = "FD16P20_m04_02kd"


def ndcg_per_feed(scores, targets, offsets, M):
    """Organisers' NDCG: binary gain, log2 discount, pessimistic ties, feeds without positives -> 1."""
    scores = np.asarray(scores, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    n = len(M)
    fid = np.repeat(np.arange(n), M)
    order = np.lexsort((targets, -scores, fid))
    t_sorted = targets[order]
    rank = np.arange(len(order)) - np.repeat(offsets, M)
    disc = 1.0 / np.log2(rank + 2.0)
    dcg = np.bincount(fid, weights=t_sorted * disc, minlength=n)
    npos = np.bincount(fid, weights=targets, minlength=n)
    idcg = np.bincount(fid, weights=(rank < np.repeat(npos, M)) * disc, minlength=n)
    out = np.ones(n)
    m = idcg > 0
    out[m] = dcg[m] / idcg[m]
    return out


def halves(F):
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
def feed_lse(x, off, M):
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
    return np.exp(x - feed_lse(x, off, M)[fid])


def nll(x, y, off, M, feed_mask=None):
    """Sum over positives of -log softmax_feed(x)_i (the listwise CE of training, one term per positive)."""
    fid = np.repeat(np.arange(len(M)), M)
    per = np.bincount(fid, -(x - feed_lse(x, off, M)[fid]) * y, len(M))
    return per[feed_mask].sum() if feed_mask is not None else per.sum()


def fit_T(s, y, off, M, feed_mask):
    r = minimize_scalar(lambda lt: nll(s / np.exp(lt), y, off, M, feed_mask), bounds=(-3, 2), method="bounded")
    return float(np.exp(r.x))


@nb.njit(cache=True)
def model_rank(s, off, M):
    """0-based position of each row in its feed under descending score."""
    rk = np.empty(len(s), np.int64)
    for f in range(len(M)):
        o = off[f]; m = M[f]
        order = np.argsort(-s[o:o + m], kind="mergesort")
        for j in range(m):
            rk[o + order[j]] = j
    return rk


@nb.njit(cache=True)
def redraw_ndcg(x, rk, off, M, npos, R, seed, disc):
    """R Plackett-Luce draws of npos[f] distinct positives per feed from softmax(x) (Gumbel top-K).
    Returns nd[R, F]: per-feed NDCG of the fixed ranking rk under each drawn label set (feeds with K=0 -> 1)."""
    np.random.seed(seed)
    F = len(M)
    nd = np.empty((R, F), np.float32)
    idcg = np.zeros(F)
    for f in range(F):
        for j in range(npos[f]):
            idcg[f] += disc[j]
    for r in range(R):
        for f in range(F):
            o = off[f]; m = M[f]; k = npos[f]
            if k == 0:
                nd[r, f] = 1.0
                continue
            if k == 1:
                best = -1e300; bi = 0
                for j in range(m):
                    g = x[o + j] - np.log(-np.log(np.random.random() + 1e-300) + 1e-300)
                    if g > best:
                        best = g; bi = j
                nd[r, f] = disc[rk[o + bi]] / idcg[f]
            else:
                g = np.empty(m)
                for j in range(m):
                    g[j] = x[o + j] - np.log(-np.log(np.random.random() + 1e-300) + 1e-300)
                top = np.argsort(-g)[:k]
                d = 0.0
                for j in range(k):
                    d += disc[rk[o + top[j]]]
                nd[r, f] = d / idcg[f]
    return nd


def exact_k1(p, rk, off, M, disc=DISC):
    """Exact E and Var of NDCG for a 1-positive feed with click ~ p (per feed, all feeds; valid where K=1)."""
    fid = np.repeat(np.arange(len(M)), M)
    d = disc[rk]
    mu = np.bincount(fid, p * d, len(M))
    m2 = np.bincount(fid, p * d * d, len(M))
    return mu, m2 - mu ** 2


def entropy_per_feed(p, off, M):
    fid = np.repeat(np.arange(len(M)), M)
    return np.bincount(fid, -p * np.log(np.maximum(p, 1e-300)), len(M))


def se(x):
    return float(np.std(x, ddof=1) / np.sqrt(len(x)))
