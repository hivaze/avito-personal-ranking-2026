"""Shared helpers for the trd_ family (transductive population trends).

Core idea: a feed recorded at time t carries the user's most recent history events (hpos small = shortly before t).
Pooling these events over many feeds in a time window around t gives a dense "what the population engages with now"
signal. Everything is computed from a POOL of feeds of the same split:
  * val/test: pool = all feeds of the split (own feed always excluded)
  * train:    pool = a per-day random subsample of train feeds at val/test density (~7.2K feeds/day), so that
              window estimates have the same noise level as in val/test (identical definition in distribution).
"""
import json, sys
from pathlib import Path
import numba as nb
import numpy as np
import polars as pl

sys.path.insert(0, "<repo>/src")
from metrics import ndcg_per_feed  # noqa

ROOT = Path("<repo>/data")
FLAT = ROOT / "flat"
FEATS = ROOT / "feats"
CACHE = ROOT / "trd_cache"
N_CAT = 54
MSK = 3 * 3600
POOL_PER_DAY = 7200.0     # val/test density (val 7.2K/day, test 7.15K/day)
PUBLIC_TS = 1790497920    # test public/private boundary


def feeds_df(split):
    return pl.read_parquet(FLAT / split / "feeds.parquet")


def load_item_emb(split):
    d = ROOT / "competition" / split / "item_embeddings"
    meta = json.loads((d / "data.meta").read_text())
    return np.memmap(d / "data.memmap", dtype=np.int8, mode="r", shape=tuple(meta["shape"]))


def pool_mask(split, feeds):
    """Boolean mask over feeds: which feeds form the population pool."""
    n = len(feeds)
    if split != "train":
        return np.ones(n, bool)
    ts = feeds["timestamp"].to_numpy().astype(np.int64)
    day = (ts + MSK) // 86400
    fid = feeds["feed_id"].to_numpy().astype(np.uint64)
    # deterministic hash -> uniform [0,1)
    h = (fid * np.uint64(0x9E3779B97F4A7C15)) & np.uint64(0xFFFFFFFF)
    u = h.astype(np.float64) / 2.0 ** 32
    dcnt = np.bincount(day - day.min())
    rate = np.minimum(1.0, POOL_PER_DAY / np.maximum(dcnt[day - day.min()], 1))
    return u < rate


@nb.njit(parallel=True, cache=True)
def feed_cat_counts(sel, hist_off, H, hcat, het, hpos, kmax, et_lo, et_hi, out):
    """out[i, j, c] = # history events of feed sel[i] with hpos<kmax[j], et_lo[j]<=event_type<=et_hi[j], category c."""
    nj = len(kmax)
    for i in nb.prange(len(sel)):
        f = sel[i]
        ho = hist_off[f]
        for k in range(H[f]):
            c = hcat[ho + k]
            if c < 0 or c >= 54:
                continue
            e = het[ho + k]
            p = hpos[ho + k]
            for j in range(nj):
                if p < kmax[j] and e >= et_lo[j] and e <= et_hi[j]:
                    out[i, j, c] += 1.0


@nb.njit(parallel=True, cache=True)
def feed_cand_cat_counts(sel, cand_off, M, ccat, out):
    for i in nb.prange(len(sel)):
        f = sel[i]
        co = cand_off[f]
        for k in range(M[f]):
            out[i, ccat[co + k]] += 1.0


def window_bounds(pool_ts, ts, w_left, w_right):
    """indices [lo, hi) into time-sorted pool for window [t - w_left, t + w_right]."""
    lo = np.searchsorted(pool_ts, ts - w_left, side="left")
    hi = np.searchsorted(pool_ts, ts + w_right, side="right")
    return lo, hi


def single_ndcg(x, y, off, M, seed=0):
    """single-feature NDCG with random tie-break, NaN -> bottom; returns (asc, desc)."""
    rng = np.random.default_rng(seed)
    noise = rng.random(len(x)) * 1e-9
    x = np.asarray(x, np.float64)
    fin = np.isfinite(x)
    lo, hi = np.nanmin(x[fin]) if fin.any() else 0, np.nanmax(x[fin]) if fin.any() else 1
    sc = (x - lo) / (hi - lo + 1e-12)
    res = []
    for sgn in (1, -1):
        s = np.where(fin, sgn * sc, -10.0) + noise * (1 + 0)
        # random tie-break: add per-row uniform noise smaller than the smallest gap
        s = s + rng.random(len(x)) * 1e-7
        res.append(float(ndcg_per_feed(s, y, off, M).mean()))
    return res[0], res[1]


def psi(a, b, bins=10):
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return np.nan
    qs = np.unique(np.quantile(a, np.linspace(0, 1, bins + 1)))
    if len(qs) < 3:
        return 0.0
    qs[0], qs[-1] = -np.inf, np.inf
    pa = np.histogram(a, qs)[0] / len(a) + 1e-4
    pb = np.histogram(b, qs)[0] / len(b) + 1e-4
    return float(((pa - pb) * np.log(pa / pb)).sum())


def day_or_window_bounds(pool_ts, ts, wv):
    """wv = (w_left, w_right) seconds, or 'pday' (MSK day start .. t) / 'sday' (whole MSK day)."""
    if isinstance(wv, str):
        d0 = (ts + MSK) // 86400 * 86400 - MSK
        lo = np.searchsorted(pool_ts, d0, side="left")
        hi = np.searchsorted(pool_ts, ts if wv == "pday" else d0 + 86399, side="right")
        return lo, hi
    return window_bounds(pool_ts, ts, wv[0], wv[1])
