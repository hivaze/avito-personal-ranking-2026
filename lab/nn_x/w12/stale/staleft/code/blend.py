"""Blend utilities: per-feed standardization of score vectors and evaluation of averages on val."""
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, "<repo>/src")
from metrics import ndcg  # noqa: E402

FLAT = Path("<repo>/data/flat")


def feed_meta(split):
    f = pl.read_parquet(FLAT / split / "feeds.parquet")
    return f, f["M"].to_numpy().astype(np.int64), f["cand_off"].to_numpy()


def feed_z(s, M):
    """Standardize scores within each feed (flat arrays in feed order)."""
    fid = np.repeat(np.arange(len(M)), M)
    mu = np.bincount(fid, s, len(M)) / M
    sd = np.sqrt(np.maximum(np.bincount(fid, s * s, len(M)) / M - mu ** 2, 0)) + 1e-6
    return (s - mu[fid]) / sd[fid]


if __name__ == "__main__":
    f, M, off = feed_meta("val")
    y = np.load(FLAT / "val/cand_target.npy")
    files = sys.argv[1:]
    P = [np.load(p).astype(np.float64) for p in files]
    for p, s in zip(files, P):
        print(f"{ndcg(s, y, off, M):.5f} {p}")
    Z = [feed_z(s, M) for s in P]
    print(f"{ndcg(np.mean(Z, 0), y, off, M):.5f} mean of per-feed z-scores")
