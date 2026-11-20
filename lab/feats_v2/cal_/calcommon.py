"""Shared helpers for the cal_ (calendar / time-of-day context) feature family."""
import sys, os, json, time
import numpy as np
import polars as pl
import numba as nb

ROOT = "<repo>"
FLAT = f"{ROOT}/data/flat"
FEATS = f"{ROOT}/data/feats"
OUT = f"{ROOT}/feats_v2/cal_/out"
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, f"{ROOT}/src")
nb.set_num_threads(int(os.environ.get("NB_THREADS", "16")))

MSK = 3 * 3600
PUBLIC_END_TS = 1790497920  # public/private boundary in test
# hour buckets (MSK): night 0-6, morning 6-9, 9-12, 12-15, 15-18, 18-21, 21-24
HB_EDGES = np.array([0, 6, 9, 12, 15, 18, 21, 24])
N_HB = 7
N_CAT = 54


def feeds(split):
    return pl.read_parquet(f"{FLAT}/{split}/feeds.parquet")


def cand(split, name, mmap=True):
    return np.load(f"{FLAT}/{split}/cand_{name}.npy", mmap_mode="r" if mmap else None)


def hist(split, name, mmap=True):
    return np.load(f"{FLAT}/{split}/hist_{name}.npy", mmap_mode="r" if mmap else None)


def time_parts(ts):
    """ts unix seconds -> dict of MSK calendar parts (per feed)."""
    t = ts.astype(np.int64) + MSK
    day = t // 86400                      # MSK day number
    sec = t % 86400
    hour = sec // 3600
    dow = (day + 3) % 7                   # 1970-01-01 was Thursday -> Mon=0 .. Sun=6
    weekend = (dow >= 5).astype(np.int8)
    hb = np.searchsorted(HB_EDGES, hour, side="right") - 1
    return dict(day=day, sec=sec, hour=hour, dow=dow, weekend=weekend, hb=hb)


def day_of_month(day):
    u, inv = np.unique(day, return_inverse=True)
    dm = np.array([np.datetime64(int(d), "D").astype(object).day for d in u])
    return dm[inv]


@nb.njit(parallel=True, cache=True)
def within_feed_rank(pos, off, M):
    """ordinal rank of pos within each feed (0 = top) and rank from end."""
    r = np.empty(len(pos), np.int16)
    re = np.empty(len(pos), np.int16)
    for f in nb.prange(len(M)):
        o = off[f]; m = M[f]
        idx = np.argsort(pos[o:o + m], kind="mergesort")
        for k in range(m):
            r[o + idx[k]] = k
            re[o + idx[k]] = m - 1 - k
    return r, re


R_EDGES = np.array([0, 1, 2, 3, 4, 5, 7, 10, 15, 20, 30, 50, 100, 150, 400])


def theta_index(r, re):
    """position cell id: rank bucket (14) x end flag (last, second-last, third-last, other) -> 56 cells"""
    rb = np.searchsorted(R_EDGES, r, side="right") - 1
    ef = np.where(re == 0, 0, np.where(re == 1, 1, np.where(re == 2, 2, 3)))
    return (rb * 4 + ef).astype(np.int16)


@nb.njit(parallel=True, cache=True)
def expected_pos(theta_row, y, off, M):
    """per-row expected positives under a position-only model: npos_f * theta_i / sum_j theta_j"""
    e = np.empty(len(y), np.float32)
    for f in nb.prange(len(M)):
        o = off[f]; m = M[f]
        s = 0.0; npos = 0.0
        for j in range(o, o + m):
            s += theta_row[j]; npos += y[j]
        for j in range(o, o + m):
            e[j] = npos * theta_row[j] / s
    return e


@nb.njit(parallel=True, cache=True)
def ndcg_nb(scores, targets, offsets, M, mode):
    """per-feed full-list NDCG; mode 0 pessimistic ties, 2 expected (random tie break)"""
    n = len(M)
    out = np.ones(n)
    for f in nb.prange(n):
        o = offsets[f]; m = M[f]
        npos = 0
        for j in range(m):
            if targets[o + j] > 0:
                npos += 1
        if npos == 0:
            continue
        s = scores[o:o + m]; t = targets[o:o + m]
        idx = np.argsort(-s)
        dcg = 0.0
        i = 0
        while i < m:
            j = i; sp = 0
            while j < m and s[idx[j]] == s[idx[i]]:
                if t[idx[j]] > 0:
                    sp += 1
                j += 1
            g = j - i
            if sp > 0:
                if mode == 0:
                    for rr in range(j - sp, j):
                        dcg += 1.0 / np.log2(rr + 2.0)
                else:
                    avg = 0.0
                    for rr in range(i, j):
                        avg += 1.0 / np.log2(rr + 2.0)
                    dcg += sp * avg / g
            i = j
        idcg = 0.0
        for rr in range(npos):
            idcg += 1.0 / np.log2(rr + 2.0)
        out[f] = dcg / idcg
    return out


def ndcg_pf(scores, targets, offsets, M, mode=0):
    return ndcg_nb(np.asarray(scores, np.float64), np.asarray(targets, np.float64), offsets.astype(np.int64),
                   M.astype(np.int64), mode)


def single_ndcg(x, y, off, M, seed=0):
    """single-feature NDCG both directions with random tie-break; NaN -> bottom."""
    res = {}
    x = np.asarray(x, np.float64)
    for nm, sgn in [("pos", 1.0), ("neg", -1.0)]:
        s = np.where(np.isnan(x), -1e30, sgn * np.nan_to_num(x, nan=0.0))
        res[nm] = float(ndcg_pf(s, y, off, M, mode=2).mean())  # mode 2 = exact expected NDCG under random tie-break
    return res


def save_feat(split, name, arr):
    arr = np.asarray(arr, dtype=np.float32)
    np.save(f"{FEATS}/{split}/{name}.npy", arr)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)
