"""Source-sequence selection for learned sequence statistics.

A "source" is a set of train feeds whose histories are used as de-duplicated user sequences:
  for each non-zero user, the LATEST train feed with day < cutoff (its history = last <=256 events before it);
  every user_id 0 feed with day < cutoff is taken separately (pooled unknown users).
cutoff = +inf -> all of train (used for val/test rows). For train rows of day d, cutoff = d (expanding window:
the histories, which implicitly contain earlier clicks, only come from feeds strictly before day d).
"""
import numpy as np
import polars as pl
from seq_lib import feeds

MSK = 3 * 3600


def day_of(ts):
    return (np.asarray(ts, dtype=np.int64) + MSK) // 86400


def train_days():
    f = feeds("train")
    d = day_of(f["timestamp"].to_numpy())
    u, c = np.unique(d, return_counts=True)
    return u[c > 1000]


def source_fidx(cutoff_day=None):
    f = feeds("train").with_columns(day=((pl.col("timestamp").cast(pl.Int64) + MSK) // 86400))
    if cutoff_day is not None:
        f = f.filter(pl.col("day") < cutoff_day)
    nz = f.filter(pl.col("user_id") != 0).sort(["user_id", "timestamp", "fidx"]).group_by("user_id").agg(
        pl.col("fidx").last())["fidx"].to_numpy()
    z = f.filter(pl.col("user_id") == 0)["fidx"].to_numpy()
    return np.sort(np.concatenate([nz, z])).astype(np.int64)


def feed_day_index(split, days):
    """For each feed of split: index into `days` cutoff list (train rows) or len(days) (= all-train source)."""
    f = feeds(split)
    if split != "train":
        return np.full(len(f), len(days), np.int64)
    d = day_of(f["timestamp"].to_numpy())
    # largest cutoff <= own day -> source strictly from earlier days (Sep 15 / Sep 20 stragglers map to the
    # previous full day's cutoff, which is still strictly earlier)
    idx = np.searchsorted(days, d, side="right") - 1
    assert idx.min() >= 0
    return idx.astype(np.int64)
