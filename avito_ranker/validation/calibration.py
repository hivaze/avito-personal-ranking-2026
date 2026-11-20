"""Predict the public and private score from validation by matching the feed-size mix.

Validation reads higher than the leaderboard, and feed size explains most of
the difference. NDCG falls from about 0.68 on feeds of up to 10 candidates to
about 0.28 on feeds of more than 200, and test feeds are longer: 96 candidates
on average on validation, 104 on the public part of test, 101 on the private
part. This module reweights the per-feed NDCG of validation so that the
distribution of feed size (ten bins of the number of candidates) matches the
public or the private part of test, and reports the weighted mean. For the
first final file validation reads 0.40271, the forecast for public is 0.3947
and the board gave 0.394892.

The public part is the test feeds before ``PUBLIC_END`` (8,562 feeds, the
first 30% by time); the private part is the other 20,033. No test label is
used: the weights depend on feed sizes and timestamps only.

The gate prints the forecast with a band of +-0.003 as information. No rule
of the gate depends on it.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from avito_ranker.ensemble.feeds import load_feed_index

PUBLIC_END = 1790497920  # Unix time; test feeds before it are scored on the public board
SIZE_BIN_EDGES = np.array([10, 20, 30, 50, 75, 100, 150, 200, 250])
BOARD_PARTS = ("public", "private")


def public_mask(test_timestamps: np.ndarray) -> np.ndarray:
    """True for test feeds that belong to the public part of the leaderboard."""
    return test_timestamps < PUBLIC_END


@lru_cache(maxsize=None)
def size_weights(part: str) -> np.ndarray:
    """Weight of every val feed (mean 1) that makes val's feed-size mix equal to ``part`` of test."""
    val, test = load_feed_index("val"), load_feed_index("test")
    public = public_mask(test.table["timestamp"].to_numpy())
    in_part = public if part == "public" else ~public
    n_bins = len(SIZE_BIN_EDGES) + 1
    val_bin = np.searchsorted(SIZE_BIN_EDGES, val.sizes, side="right")
    test_bin = np.searchsorted(SIZE_BIN_EDGES, test.sizes[in_part], side="right")
    val_share = np.bincount(val_bin, minlength=n_bins) / len(val_bin)
    test_share = np.bincount(test_bin, minlength=n_bins) / len(test_bin)
    weights = (test_share / np.maximum(val_share, 1e-12))[val_bin]
    return weights / weights.mean()


def predict_board(per_feed_ndcg: np.ndarray) -> dict[str, float]:
    """Return ``{"public": ..., "private": ...}``: val NDCG reweighted to each part of test."""
    return {part: float((size_weights(part) * per_feed_ndcg).mean()) for part in BOARD_PARTS}
