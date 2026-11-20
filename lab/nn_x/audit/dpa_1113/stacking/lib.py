"""Shared helpers for the second-level (stacking) lens. Read-only on project files; writes only to this dir."""
import os
for _k in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'POLARS_MAX_THREADS', 'NUMBA_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[_k] = '12'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import sys
sys.path.insert(0, '<repo>/src')
from pathlib import Path
import numpy as np
import numba as nb

ROOT = Path('<repo>')
P = ROOT / 'preds'
HD = P / 'holdout'
OUT = Path('/tmp/scratchpad/dpa/stacking')
CACHE = OUT / 'cache'

FAM = {
    'F10': [f'v23_gru_pre2s_s{s}' for s in range(501, 507)],
    'F7x': [f'r6_cd_m02_r{s}' for s in range(101, 108)],
    'F7x70': [f'r6_cd_m02_p70k_r{s}' for s in range(101, 108)],
}
MEMBERS = [m for f in FAM for m in FAM[f]]
FAMIDX = np.array([i for i, f in enumerate(FAM) for _ in FAM[f]])  # member -> family index


@nb.njit(parallel=True, cache=True)
def feed_z(off, M, x):
    out = np.empty(len(x), np.float32)
    for f in nb.prange(len(M)):
        o, m = off[f], M[f]
        v = x[o:o + m].astype(np.float64)
        mu = v.mean(); sd = v.std() + 1e-9
        for k in range(m):
            out[o + k] = (v[k] - mu) / sd
    return out


@nb.njit(parallel=True, cache=True)
def feed_pct(off, M, x):
    """within-feed percentile rank 0..1 (average-free: argsort of argsort, ties broken by index)."""
    out = np.empty(len(x), np.float32)
    for f in nb.prange(len(M)):
        o, m = off[f], M[f]
        r = np.argsort(np.argsort(x[o:o + m]))
        for k in range(m):
            out[o + k] = r[k] / (m - 1) if m > 1 else 0.5
    return out


@nb.njit(parallel=True, cache=True)
def ndcg_nb(off, M, y, s):
    """Per-feed NDCG, binary gain, log2 discount, pessimistic ties (negatives first among equal scores), no positives -> 1.
    Identical to src/metrics.ndcg_per_feed (verified in build_cache.py)."""
    n = len(M)
    out = np.ones(n)
    for f in nb.prange(n):
        o, m = off[f], M[f]
        npos = 0
        for k in range(m):
            if y[o + k] > 0.5:
                npos += 1
        if npos == 0:
            continue
        idcg = 0.0
        for r in range(npos):
            idcg += 1.0 / np.log2(r + 2.0)
        # rank of positive i = #{j: s_j > s_i} + #{neg j: s_j == s_i} + #{pos j<i: s_j == s_i}
        dcg = 0.0
        for i in range(m):
            if y[o + i] < 0.5:
                continue
            si = s[o + i]
            r = 0
            for j in range(m):
                sj = s[o + j]
                if sj > si:
                    r += 1
                elif sj == si and j != i:
                    if y[o + j] < 0.5 or j < i:
                        r += 1
            dcg += 1.0 / np.log2(r + 2.0)
        out[f] = dcg / idcg
    return out


def se_cl(d, cl, mask=None, w=None):
    """SE of the mean of per-feed d: max(iid, user-clustered); optional mask."""
    if mask is not None:
        d = d[mask]; cl = cl[mask]
    n = len(d); mu = d.mean()
    iid = d.std() / np.sqrt(n)
    s = np.bincount(cl, d - mu); g = (s != 0).sum()
    clu = np.sqrt((s ** 2).sum()) / n * np.sqrt(g / max(g - 1, 1))
    return mu, max(iid, clu)


def wmean_se(d, w):
    m = np.average(d, weights=w)
    neff = w.sum() ** 2 / (w ** 2).sum()
    return m, np.sqrt(np.average((d - m) ** 2, weights=w) / neff)


class S:
    """Loaded split from the cache."""
    def __init__(self, name):
        d = np.load(CACHE / f'{name}_meta.npz')
        for k in d.files:
            setattr(self, k, d[k])
        self.name = name
        self.n = len(self.M)
        self.fid = np.repeat(np.arange(self.n), self.M)

    def Z(self, half):
        return np.load(CACHE / f'{self.name}_z_{half}.npy', mmap_mode='r')  # [20, rows] float32

    def ev(self, s):
        return ndcg_nb(self.off, self.M, self.y, np.ascontiguousarray(s, dtype=np.float64))

    def z(self, x):
        return feed_z(self.off, self.M, np.ascontiguousarray(x, dtype=np.float32))


def rep(tag, d, sp, extra='', pm=None):
    mu, se = se_cl(d, sp.cl)
    s = f'{tag:38s} d {mu:+.5f} +- {se:.5f} (z {mu / se:+.2f})'
    return s + extra, mu, se


@nb.njit(parallel=True, cache=True)
def wf_agree(off, M, a, b):
    """per-feed Spearman rank corr of a vs b and top-1 agreement."""
    n = len(M); rho = np.ones(n); top = np.ones(n)
    for f in nb.prange(n):
        o, m = off[f], M[f]
        if m < 2:
            continue
        ra = np.argsort(np.argsort(a[o:o + m])).astype(np.float64)
        rb = np.argsort(np.argsort(b[o:o + m])).astype(np.float64)
        ra -= ra.mean(); rb -= rb.mean()
        rho[f] = (ra * rb).sum() / np.sqrt((ra * ra).sum() * (rb * rb).sum())
        top[f] = 1.0 if np.argmax(a[o:o + m]) == np.argmax(b[o:o + m]) else 0.0
    return rho, top
