"""Shared horizon-matching helpers for the hz* feature families (hzcv, ixh, it_hist_hz).

Train/val/test parity idea (featrev_research RESEARCH-2 / F1): val/test rows see a dense source that ends at a fixed
cutoff (2026-09-20 00:00 MSK) and are 2-10 days stale; train rows built with "source strictly before own day" are 0-1
day stale. Train rows therefore get a per-feed LAG L in {0, 4, 8} days on their cutoff, plus a 1/37 hash-sampled
"sparse" window [c, t) that mirrors the exploration-density val/test histories between the cutoff and the feed.

  lag L      train feeds only: L = [0, 4, 8][((feed_id * 0x9E3779B97F4A7C15 mod 2^64) >> 40) % 3] days;
             forced to 0 for the 30K nn/train.py --holdout feeds (nn/train.py:153-162) so the holdout keeps the
             as-built (L = 0) horizon. Val/test: L = 0 (their cutoff is the fixed CUT_VT).
  sparse s   train feeds only: s = (((feed_id * 0xC2B2AE3D27D4EB4F mod 2^64) >> 40) % 37 == 0)  (~7.1K feeds/day,
             the val/test feed density of ~7.2K/day).
  cutoff c   train: 00:00 MSK of the latest full train day <= own day (feats_v2/seq_/sources.py feed_day_index ==
             feat_new/research_expo.py cutoff_per_feed) minus 86400 * L; val/test: CUT_VT = 1789851600.
  t_post     t_feed - 3600 * age_hours(f4) (life_ knots, research_expo.age_hours); f4 >= 63/64 -> -inf.

Everything here depends only on feed_id, the feed timestamp, f4 and fixed constants: no labels, no other feeds.
"""
from pathlib import Path
import numpy as np
import polars as pl

ROOT = Path("<repo>")
FLAT = ROOT / "data/flat"
FEATS = ROOT / "data/feats"
FEATS_NEW = ROOT / "data/feats_new"
MSK = 3 * 3600
DAY = 86400
CUT_VT = 1789851600  # 2026-09-20 00:00 MSK
LAGS = np.array([0, 4, 8], np.int64)
H_LAG = np.uint64(0x9E3779B97F4A7C15)
H_SP = np.uint64(0xC2B2AE3D27D4EB4F)
SP_MOD = 37
K_KNOTS = 64
F4_CAP = 63 / 64


def feeds(split):
    return pl.read_parquet(FLAT / split / "feeds.parquet")


def day_of(ts):
    return (np.asarray(ts, dtype=np.int64) + MSK) // DAY


def lag_raw(feed_id):
    """Hash lag in days {0,4,8} from feed_id only (before the holdout override)."""
    h = (np.asarray(feed_id).astype(np.uint64) * H_LAG) >> np.uint64(40)
    return LAGS[(h % np.uint64(3)).astype(np.int64)]


def sparse_flag(feed_id):
    h = (np.asarray(feed_id).astype(np.uint64) * H_SP) >> np.uint64(40)
    return (h % np.uint64(SP_MOD)) == 0


def holdout_feeds():
    """Exactly nn/train.py --holdout 30000: p_random>0.9 (M<50 -> 0), M>=50, rng 12345 (nn/train.py:148-158)."""
    f = feeds("train")
    M = f["M"].to_numpy().astype(np.int64)
    pr = np.load(ROOT / "eda/labels_position/out/train_prandom.npz")
    assert (pr["fidx"] == np.arange(len(f))).all()
    p = np.nan_to_num(pr["p_random"].astype(np.float32), nan=0.0)
    p[M < 50] = 0.0
    cand = np.where((p > 0.9) & (M >= 50))[0]
    return np.sort(np.random.default_rng(12345).choice(cand, min(30000, len(cand)), replace=False))


def p_random_raw():
    """Raw train p_random per feed (NaN -> 0, no M filter)."""
    pr = np.load(ROOT / "eda/labels_position/out/train_prandom.npz")
    return np.nan_to_num(pr["p_random"].astype(np.float32), nan=0.0)


def train_days():
    """Full train days (count > 1000), same rule as feats_v2/seq_/sources.py::train_days."""
    d = day_of(feeds("train")["timestamp"].to_numpy())
    u, c = np.unique(d, return_counts=True)
    return u[c > 1000]


def lag_per_feed(split):
    """Per-feed lag in days (int64). Train: hash lag with the 30K holdout forced to 0. Val/test: 0."""
    f = feeds(split)
    if split != "train":
        return np.zeros(len(f), np.int64)
    L = lag_raw(f["feed_id"].to_numpy())
    L[holdout_feeds()] = 0
    return L


def sparse_per_feed(split):
    f = feeds(split)
    if split != "train":
        return np.zeros(len(f), np.bool_)
    return sparse_flag(f["feed_id"].to_numpy())


def cutoff_per_feed(split, lag=True):
    """(ts int64, cutoff int64, L int64) per feed. lag=False -> the as-built seq_cv cutoff (L = 0 everywhere)."""
    f = feeds(split)
    ts = f["timestamp"].to_numpy().astype(np.int64)
    if split != "train":
        return ts, np.full(len(ts), CUT_VT, np.int64), np.zeros(len(ts), np.int64)
    d = day_of(ts)
    days = train_days()
    idx = np.searchsorted(days, d, side="right") - 1
    assert idx.min() >= 0
    L = lag_per_feed("train") if lag else np.zeros(len(ts), np.int64)
    return ts, days[idx] * DAY - MSK - DAY * L, L


def age_hours(x, knots):
    """Piecewise-linear f4 -> listing age in hours (copy of feat_new/research_expo.py::age_hours); NaN at the cap."""
    x = x.astype(np.float64)
    u = x * K_KNOTS
    k = np.clip(np.floor(u).astype(int), 1, K_KNOTS - 2)
    w = u - k
    a = knots[k] * (1 - w) + knots[k + 1] * w
    return np.where(x < F4_CAP, a, np.nan)


def knots():
    return np.load(FEATS / "life_f4_age_knots.npy")


def expo_hours(split, cut, ts, step=20_000_000):
    """E = (cutoff - t_post) in hours per candidate row (float32), 1e5 for capped f4. Identical arithmetic to
    feat_new/research_expo.py (so val/test equal data/feats_new/<split>/src_expo_h bit for bit)."""
    kn = knots()
    fi = np.load(FLAT / split / "cand_fidx.npy", mmap_mode="r")
    f4 = np.load(FLAT / split / "cand_f4.npy", mmap_mode="r")
    n = len(fi)
    E = np.empty(n, np.float32)
    for s in range(0, n, step):
        ff = np.asarray(fi[s:s + step])
        a = age_hours(np.asarray(f4[s:s + step]), kn)
        e = (cut[ff] - ts[ff]) / 3600.0 + a
        E[s:s + step] = np.where(np.isnan(e), 1e5, e)
    return E


def slog_days(E):
    """sign(E) * log1p(|E| / 24) in float32 (same ops as research_expo.py src_expo_slog)."""
    return (np.sign(E) * np.log1p(np.abs(E) / 24.0)).astype(np.float32)


def stale_d(ts, cut):
    """Feed-level staleness clip((t - c)/86400, 0, 9)/10 (float32)."""
    return (np.clip((ts - cut) / DAY, 0, 9) / 10.0).astype(np.float32)


def save_new(split, name, x):
    x = np.asarray(x, dtype=np.float32)
    n = len(np.load(FLAT / split / "cand_fidx.npy", mmap_mode="r"))
    assert len(x) == n, (split, name, len(x), n)
    (FEATS_NEW / split).mkdir(parents=True, exist_ok=True)
    tmp = FEATS_NEW / split / f"{name}.tmp.npy"
    np.save(tmp, x)
    tmp.replace(FEATS_NEW / split / f"{name}.npy")
