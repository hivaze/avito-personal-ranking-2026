"""Shared kernels for the feature audit (copies of the s01 functions, importable without running s01)."""
import os
for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS"]:
    os.environ.setdefault(k, "12")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from pathlib import Path
import numpy as np, polars as pl, numba as nb
from scipy.stats import rankdata

ROOT = Path("<repo>")
FLAT = ROOT / "data/flat"
FEATS = ROOT / "data/feats"
FEATS_NEW = ROOT / "data/feats_new"
BOUND = 1790497920


def feat_file(split, name):
    for d in (FEATS, FEATS_NEW):
        p = d / split / f"{name}.npy"
        if p.exists():
            return p
    return None


def meta(split):
    f = pl.read_parquet(FLAT / split / "feeds.parquet")
    return f, f["M"].to_numpy().astype(np.int64), f["cand_off"].to_numpy().astype(np.int64)


def rows_of(off, M, fidx):
    return np.concatenate([np.arange(o, o + m) for o, m in zip(off[fidx], M[fidx])])


def local_off(M):
    return np.concatenate([[0], np.cumsum(M)[:-1]]).astype(np.int64)


def train_subsets():
    """Same draws as s01: exact 30K holdout (trRL), 12K production feeds (trProd), 8K M<50 feeds (trSmall)."""
    ftr, Mtr, offtr = meta("train")
    pr = np.load(ROOT / "eda/labels_position/out/train_prandom.npz")
    p_rand = np.nan_to_num(pr["p_random"].astype(np.float32), nan=0.0); p_rand[Mtr < 50] = 0.0
    cand = np.where((p_rand > 0.9) & (Mtr >= 50))[0]
    hold = np.sort(np.random.default_rng(12345).choice(cand, min(30000, len(cand)), replace=False))
    rng = np.random.default_rng(2026)
    prod_pool = np.where((p_rand < 0.05) & (Mtr >= 50) & ~np.isin(np.arange(len(ftr)), hold))[0]
    prod = np.sort(rng.choice(prod_pool, 12000, replace=False))
    small = np.sort(rng.choice(np.where(Mtr < 50)[0], 8000, replace=False))
    return ftr, Mtr, offtr, {"trRL": hold, "trProd": prod, "trSmall": small}


def sub(x, n=250_000, seed=7):
    if len(x) <= n:
        return x
    return x[np.random.default_rng(seed).choice(len(x), n, replace=False)]


def auc2(a, b):
    a = np.where(np.isnan(a), -np.inf, a); b = np.where(np.isnan(b), -np.inf, b)
    x = np.concatenate([a, b]); r = rankdata(x)
    nb_, na_ = len(b), len(a)
    return float((r[na_:].sum() - nb_ * (nb_ + 1) / 2) / (na_ * nb_))


@nb.njit(cache=True)
def feed_agg(x, off, M):
    nF = len(M)
    mean = np.full(nF, np.nan); nan = np.zeros(nF); zero = np.zeros(nF); mx = np.full(nF, np.nan)
    for i in range(nF):
        o = off[i]; m = M[i]; s = 0.0; c = 0; z = 0; best = -1e30
        for k in range(m):
            v = x[o + k]
            if v != v:
                continue
            s += v; c += 1
            if v == 0.0:
                z += 1
            if v > best:
                best = v
        nan[i] = 1.0 - c / m
        if c > 0:
            mean[i] = s / c; zero[i] = z / c; mx[i] = best
    return mean, nan, zero, mx


@nb.njit(cache=True)
def feed_auc(x, y, off, M):
    tot = 0.0; n = 0
    for i in range(len(M)):
        o = off[i]; m = M[i]; npos = 0
        for k in range(m):
            if y[o + k] > 0:
                npos += 1
        if npos == 0 or npos == m:
            continue
        v = np.empty(m)
        for k in range(m):
            t = x[o + k]
            v[k] = -1e30 if t != t else t
        order = np.argsort(v)
        ranks = np.empty(m); i0 = 0
        while i0 < m:
            j = i0
            while j + 1 < m and v[order[j + 1]] == v[order[i0]]:
                j += 1
            avg = 0.5 * (i0 + j) + 1.0
            for k in range(i0, j + 1):
                ranks[order[k]] = avg
            i0 = j + 1
        s = 0.0
        for k in range(m):
            if y[o + k] > 0:
                s += ranks[k]
        tot += (s - npos * (npos + 1) / 2) / (npos * (m - npos)); n += 1
    return tot / max(n, 1), n


@nb.njit(cache=True)
def feed_z_nanmin(x, off, M):
    out = np.zeros(len(x))
    for i in range(len(M)):
        o = off[i]; m = M[i]; mn = 1e30; c = 0
        for k in range(m):
            v = x[o + k]
            if v == v:
                c += 1
                if v < mn:
                    mn = v
        if c == 0:
            continue
        s = 0.0; s2 = 0.0
        for k in range(m):
            v = x[o + k]
            if v != v:
                v = mn
            s += v; s2 += v * v
        mu = s / m; sd = np.sqrt(max(s2 / m - mu * mu, 0.0))
        if sd < 1e-9:
            continue
        for k in range(m):
            v = x[o + k]
            if v != v:
                v = mn
            out[o + k] = (v - mu) / sd
    return out


@nb.njit(cache=True)
def pctrank(x, off, M):
    """within-feed pct rank in [-0.5, 0.5], ties averaged, NaN -> below min"""
    out = np.zeros(len(x))
    for f_ in range(len(M)):
        o = off[f_]; m = M[f_]
        if m <= 1:
            continue
        v = np.empty(m)
        for k in range(m):
            t = x[o + k]
            v[k] = -1e30 if t != t else t
        order = np.argsort(v)
        i = 0
        while i < m:
            j = i
            while j + 1 < m and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = 0.5 * (i + j) / (m - 1) - 0.5
            for k in range(i, j + 1):
                out[o + order[k]] = avg
            i = j + 1
    return out


@nb.njit(cache=True)
def clogit_slope(z, y, off, M, iters=12):
    th = 0.0; ll0 = 0.0
    for it in range(iters + 1):
        g = 0.0; h = 0.0; ll = 0.0
        for i in range(len(M)):
            o = off[i]; m = M[i]; npos = 0.0
            for k in range(m):
                npos += y[o + k]
            if npos == 0:
                continue
            mx = -1e30
            for k in range(m):
                v = th * z[o + k]
                if v > mx:
                    mx = v
            Z = 0.0; ez = 0.0; ezz = 0.0
            for k in range(m):
                e = np.exp(th * z[o + k] - mx)
                Z += e; ez += e * z[o + k]; ezz += e * z[o + k] * z[o + k]
            Ez = ez / Z; Vz = max(ezz / Z - Ez * Ez, 0.0)
            spos = 0.0
            for k in range(m):
                if y[o + k] > 0:
                    spos += z[o + k]; ll += th * z[o + k] - (np.log(Z) + mx)
            g += -spos + npos * Ez; h += npos * Vz
        if it == 0:
            ll0 = ll
        if it == iters or h <= 0:
            return th, (ll - ll0) / len(M)
        step = -g / (h + 1e-9)
        if step > 1.0:
            step = 1.0
        if step < -1.0:
            step = -1.0
        th += step
    return th, 0.0
