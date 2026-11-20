"""The metric: per-feed NDCG with binary gain, pessimistic ties and 1 for feeds without a click."""
from __future__ import annotations

import numpy as np
import pytest

from avito_ranker.metrics import ndcg, ndcg_per_feed


def _layout(sizes: list[int]) -> tuple[np.ndarray, np.ndarray]:
    sizes = np.asarray(sizes, np.int64)
    return np.r_[0, np.cumsum(sizes)[:-1]].astype(np.int64), sizes


def _reference_ndcg(scores: np.ndarray, targets: np.ndarray) -> float:
    """One feed, written the slow way: sort by score, negatives first among equal scores."""
    if targets.sum() == 0:
        return 1.0
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], targets[i]))
    discount = [1.0 / np.log2(position + 2.0) for position in range(len(scores))]
    dcg = sum(targets[i] * discount[position] for position, i in enumerate(order))
    ideal = sum(discount[: int(targets.sum())])
    return dcg / ideal


def test_perfect_order_scores_one():
    offsets, sizes = _layout([4])
    assert ndcg([0.9, 0.1, 0.8, 0.2], [1, 0, 1, 0], offsets, sizes) == pytest.approx(1.0)


def test_hand_computed_feed():
    # One click, ranked third of four: DCG = 1 / log2(4), ideal DCG = 1 / log2(2).
    offsets, sizes = _layout([4])
    value = ndcg([0.4, 0.3, 0.2, 0.1], [0, 0, 1, 0], offsets, sizes)
    assert value == pytest.approx(0.5)


def test_two_clicks_hand_computed():
    # Clicks at positions 1 and 3 (1-based): DCG = 1 + 1 / log2(4) = 1.5, ideal = 1 + 1 / log2(3).
    offsets, sizes = _layout([3])
    value = ndcg([3.0, 2.0, 1.0], [1, 0, 1], offsets, sizes)
    assert value == pytest.approx(1.5 / (1.0 + 1.0 / np.log2(3.0)))


def test_tie_between_click_and_non_click_puts_the_non_click_first():
    offsets, sizes = _layout([2])
    # Equal scores: the click is treated as ranked second, 1 / log2(3) of the ideal.
    assert ndcg([0.5, 0.5], [1, 0], offsets, sizes) == pytest.approx(1.0 / np.log2(3.0))
    assert ndcg([0.5, 0.5], [0, 1], offsets, sizes) == pytest.approx(1.0 / np.log2(3.0))


def test_constant_scores_give_the_worst_order():
    offsets, sizes = _layout([5])
    targets = [1, 0, 0, 1, 0]
    constant = ndcg(np.zeros(5), targets, offsets, sizes)
    worst = ndcg([0.0, 1.0, 1.0, 0.0, 1.0], targets, offsets, sizes)
    assert constant == pytest.approx(worst)
    assert constant < 1.0


def test_feed_without_click_scores_one():
    offsets, sizes = _layout([3, 2])
    per_feed = ndcg_per_feed([0.1, 0.2, 0.3, 0.9, 0.1], [0, 0, 0, 0, 1], offsets, sizes)
    assert per_feed[0] == 1.0
    assert per_feed[1] == pytest.approx(1.0 / np.log2(3.0))


def test_mean_is_unweighted_over_feeds():
    # A long feed and a short feed count the same.
    offsets, sizes = _layout([6, 2])
    scores = [6, 5, 4, 3, 2, 1, 1, 2]
    targets = [0, 0, 0, 0, 0, 1, 0, 1]
    per_feed = ndcg_per_feed(scores, targets, offsets, sizes)
    assert per_feed[1] == pytest.approx(1.0)
    assert ndcg(scores, targets, offsets, sizes) == pytest.approx((per_feed[0] + 1.0) / 2.0)


def test_only_the_order_inside_a_feed_matters():
    rng = np.random.default_rng(0)
    offsets, sizes = _layout([7, 3, 12])
    scores = rng.normal(size=22)
    targets = (rng.random(22) < 0.3).astype(np.int8)
    shifted = scores.copy()
    shifted[:7] = 10.0 * shifted[:7] + 100.0   # a monotone change of one feed's scores
    assert np.array_equal(ndcg_per_feed(scores, targets, offsets, sizes),
                          ndcg_per_feed(shifted, targets, offsets, sizes))


def test_matches_slow_reference_with_ties_and_empty_feeds():
    rng = np.random.default_rng(1)
    sizes = rng.integers(1, 30, 200)
    offsets, sizes = _layout(list(sizes))
    n = int(sizes.sum())
    scores = rng.integers(0, 4, n).astype(np.float32)   # few distinct values: many ties
    targets = (rng.random(n) < 0.1).astype(np.int8)     # some feeds have no click
    fast = ndcg_per_feed(scores, targets, offsets, sizes)
    slow = [_reference_ndcg(scores[o:o + m], targets[o:o + m]) for o, m in zip(offsets, sizes)]
    assert (np.asarray(slow) == 1.0).sum() > 0
    np.testing.assert_allclose(fast, slow, rtol=1e-12)


def test_offsets_of_a_subset_are_refused():
    """Offsets taken from the whole split would group the rows wrongly; the metric refuses them."""
    scores, targets = [3.0, 2.0, 1.0, 0.5, 0.2], [1, 0, 0, 0, 1]
    offsets, sizes = _layout([3, 2])
    assert ndcg_per_feed(scores, targets, offsets, sizes) == pytest.approx([1.0, 1.0 / np.log2(3.0)])
    with pytest.raises(ValueError, match="end to end"):
        ndcg_per_feed(scores, targets, offsets + 100, sizes)
    with pytest.raises(ValueError, match="candidate rows"):
        ndcg_per_feed(scores[:4], targets[:4], offsets, sizes)
