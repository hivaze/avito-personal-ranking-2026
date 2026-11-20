"""Shared helpers for labels/position EDA."""
import sys, os
import numpy as np
import polars as pl
import numba as nb

ROOT = '<repo>'
FLAT = f'{ROOT}/data/flat'
OUT = f'{ROOT}/eda/labels_position/out'
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, f'{ROOT}/src')


def feeds(split):
    return pl.read_parquet(f'{FLAT}/{split}/feeds.parquet')


def cand(split, name, mmap=True):
    return np.load(f'{FLAT}/{split}/cand_{name}.npy', mmap_mode='r' if mmap else None)


@nb.njit(parallel=True, cache=True)
def ndcg_nb(scores, targets, offsets, M, k, mode):
    """Per-feed NDCG. mode: 0 pessimistic ties, 1 optimistic ties, 2 expected (random tie break).
    k<=0 -> full list. Feeds without positives -> 1."""
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
        s = scores[o:o + m]
        t = targets[o:o + m]
        idx = np.argsort(-s)
        dcg = 0.0
        i = 0
        while i < m:
            j = i
            sp = 0
            while j < m and s[idx[j]] == s[idx[i]]:
                if t[idx[j]] > 0:
                    sp += 1
                j += 1
            g = j - i
            if sp > 0:
                if mode == 0:
                    a = j - sp
                elif mode == 1:
                    a = i
                else:
                    a = -1
                if a >= 0:
                    for r in range(a, a + sp):
                        if k <= 0 or r < k:
                            dcg += 1.0 / np.log2(r + 2.0)
                else:
                    tot = 0.0
                    for r in range(i, j):
                        if k <= 0 or r < k:
                            tot += 1.0 / np.log2(r + 2.0)
                    dcg += sp * tot / g
            i = j
        idcg = 0.0
        for r in range(npos):
            if k <= 0 or r < k:
                idcg += 1.0 / np.log2(r + 2.0)
        out[f] = dcg / idcg
    return out


def ndcg_pf(scores, targets, offsets, M, k=0, mode=0):
    return ndcg_nb(np.ascontiguousarray(scores, dtype=np.float64), np.ascontiguousarray(targets, dtype=np.int8),
                   np.ascontiguousarray(offsets, dtype=np.int64), np.ascontiguousarray(M, dtype=np.int64), int(k), int(mode))


def paired(a, b):
    """mean diff and SE of per-feed paired difference a-b"""
    d = a - b
    return d.mean(), d.std(ddof=1) / np.sqrt(len(d))


def subsample_feeds(F, n, seed=0):
    rng = np.random.default_rng(seed)
    sel = np.sort(rng.choice(len(F), size=n, replace=False))
    return sel


def gather(split, sel, F, names):
    """Gather candidate arrays for selected feed indices; returns dict + new offsets, M."""
    off = F['cand_off'].to_numpy()[sel]; M = F['M'].to_numpy()[sel]
    idx = np.concatenate([np.arange(o, o + m) for o, m in zip(off, M)]) if len(sel) < 50000 else _ranges(off, M)
    res = {}
    for nm in names:
        a = cand(split, nm)
        res[nm] = np.asarray(a[idx])
    newoff = np.concatenate([[0], np.cumsum(M)[:-1]]).astype(np.int64)
    return res, newoff, M


def _ranges(off, M):
    tot = int(M.sum())
    starts = np.repeat(off - np.concatenate([[0], np.cumsum(M)[:-1]]), M)
    return starts + np.arange(tot)


@nb.njit(parallel=True, cache=True)
def gap_stats(pos, off, M):
    """per feed: number of missing positions below max pos, mean relative position of gaps, gaps in top-10"""
    n = len(M)
    ngap = np.zeros(n); relgap = np.full(n, np.nan); gap_top10 = np.zeros(n)
    for f in nb.prange(n):
        o = off[f]; m = M[f]
        mx = 0
        for j in range(o, o + m):
            if pos[j] > mx:
                mx = pos[j]
        present = np.zeros(mx + 1, np.bool_)
        for j in range(o, o + m):
            present[pos[j]] = True
        s = 0.0; c = 0; t10 = 0
        for p in range(mx + 1):
            if not present[p]:
                c += 1; s += p / max(mx, 1)
                if p < 10:
                    t10 += 1
        ngap[f] = c; gap_top10[f] = t10
        if c > 0:
            relgap[f] = s / c
    return ngap, relgap, gap_top10
