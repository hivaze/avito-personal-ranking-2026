"""NDCG as the competition evaluator computes it.

One feed is one ranked list. The gain of a candidate is its click label (0 or
1), the discount is ``1 / log2(position + 1)`` over the whole list, and the
score of a split is the plain mean over feeds. Two rules of the evaluator
matter in practice:

* Ties are scored pessimistically. Among candidates with equal scores the
  negatives are placed first, so a model cannot gain from producing ties.
* A feed without a click scores 1. Such feeds add a constant to every model.

Scores and labels are flat arrays in feed order: the candidates of feed ``i``
occupy rows ``offsets[i] : offsets[i] + sizes[i]``, with the feeds laid end to
end from row 0. For a subset of the feeds of a split, gather their rows first
and pass ``feed_offsets(sizes)`` (``avito_ranker.data.flatten``); offsets
taken from the whole split are refused.
"""
from __future__ import annotations

import numpy as np


def ndcg_per_feed(scores: np.ndarray, targets: np.ndarray, offsets: np.ndarray, sizes: np.ndarray) -> np.ndarray:
    """Return the NDCG of every feed as a float64 array of length ``len(sizes)``.

    ``scores`` and ``targets`` are flat arrays in feed order, ``offsets`` is
    the first row of each feed and ``sizes`` its number of candidates. Raises
    unless the feeds lie end to end from row 0 and cover every row.
    """
    scores = np.asarray(scores, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    sizes = np.asarray(sizes)
    if not np.array_equal(np.asarray(offsets), np.cumsum(sizes) - sizes):
        raise ValueError("offsets must lay the feeds end to end from row 0; use feed_offsets(sizes) for a subset")
    if len(scores) != sizes.sum() or len(targets) != len(scores):
        raise ValueError(f"{len(scores)} scores and {len(targets)} targets for {int(sizes.sum())} candidate rows")
    n_feeds = len(sizes)
    feed_of_row = np.repeat(np.arange(n_feeds), sizes)

    # Sort inside each feed by score, best first. Equal scores are ordered by
    # label ascending: this is the pessimistic tie rule.
    order = np.lexsort((targets, -scores, feed_of_row))
    sorted_targets = targets[order]
    position = np.arange(len(order)) - np.repeat(offsets, sizes)  # 0 = top of the feed
    discount = 1.0 / np.log2(position + 2.0)

    dcg = np.bincount(feed_of_row, weights=sorted_targets * discount, minlength=n_feeds)
    n_positive = np.bincount(feed_of_row, weights=targets, minlength=n_feeds)
    ideal_targets = (position < np.repeat(n_positive, sizes)).astype(np.float64)
    ideal_dcg = np.bincount(feed_of_row, weights=ideal_targets * discount, minlength=n_feeds)

    out = np.ones(n_feeds)
    has_positive = ideal_dcg > 0
    out[has_positive] = dcg[has_positive] / ideal_dcg[has_positive]
    return out


def ndcg(scores: np.ndarray, targets: np.ndarray, offsets: np.ndarray, sizes: np.ndarray) -> float:
    """Return the mean per-feed NDCG, the number the leaderboard reports."""
    return float(ndcg_per_feed(scores, targets, offsets, sizes).mean())
