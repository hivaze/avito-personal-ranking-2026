"""The private-mix view: validation reweighted to look like the private part of test.

Validation and the private test days differ in two things that can be measured
without labels: the size of a feed and its cold share, the fraction of its
candidates whose item never occurs in train (neither as a candidate nor in a
history). The mean cold share is 0.290 on validation, 0.371 on the public part
of test and 0.406 on the private part. A change that helps warm items and
hurts cold ones can look good on validation and lose on private.

This module weights each validation feed so that the joint distribution of
(feed size, cold share) over 6 x 4 cells equals that of the public or the
private test feeds. A paired difference between two rankings can then be read
three ways: plain validation, public mix, private mix.

How it was used: as a veto, declared before any candidate was read with it. A
change whose private-mix difference against its reference was below -1
standard error was not adopted, whatever plain validation said. The view never
promoted a change on its own, and its level is not a forecast of the board.

The cold share is computed from the item ids of train (two flat arrays, 3.3 GB
on disk) the first time it is needed in a process. That takes about 7 s.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from avito_ranker.data.flatten import read_flat
from avito_ranker.ensemble.feeds import load_feed_index
from avito_ranker.validation.calibration import BOARD_PARTS, public_mask

# Cells are (edge[i-1], edge[i]]: a feed of exactly 50 candidates is in the 26-50 bin.
SIZE_EDGES = (10, 25, 50, 100, 200)      # 6 bins of the number of candidates
COLD_SHARE_EDGES = (0.15, 0.30, 0.45)    # 4 bins of the cold share


@dataclass(frozen=True)
class Estimate:
    """A mean with its standard error."""

    mean: float
    se: float


@dataclass(frozen=True)
class MixComparison:
    """A paired difference (candidate minus reference) under the three weightings."""

    val: Estimate
    public: Estimate
    private: Estimate


_CHUNK_ROWS = 50_000_000


def _mark(seen: np.ndarray, item_ids: np.ndarray) -> None:
    """Set ``seen[i]`` for every id of a (memory-mapped) array, a chunk at a time."""
    for start in range(0, len(item_ids), _CHUNK_ROWS):
        seen[item_ids[start:start + _CHUNK_ROWS]] = True


@lru_cache(maxsize=None)
def _items_seen_in_train() -> np.ndarray:
    """Boolean table over item ids: does the item occur in train, as a candidate or in a history."""
    train_candidates = read_flat("train", "cand_item_id")
    train_history = read_flat("train", "hist_item_id")
    held_out_max = max(int(read_flat(split, "cand_item_id").max()) for split in ("val", "test"))
    n_items = max(int(train_candidates.max()), int(train_history.max()), held_out_max) + 1
    seen = np.zeros(n_items, bool)
    _mark(seen, train_candidates)
    _mark(seen, train_history)
    return seen


@lru_cache(maxsize=None)
def cold_share(split: str) -> np.ndarray:
    """Per feed of ``split``, the fraction of candidates whose item never occurs in train."""
    feeds = load_feed_index(split)
    is_cold = ~_items_seen_in_train()[read_flat(split, "cand_item_id", mmap=False)]
    return np.bincount(feeds.feed_of_row, is_cold, feeds.n_feeds) / feeds.sizes


def _cell(sizes: np.ndarray, cold: np.ndarray) -> np.ndarray:
    """Index of the (feed size, cold share) cell of every feed."""
    size_bin = np.searchsorted(np.asarray(SIZE_EDGES, np.float64), sizes.astype(np.float64), side="left")
    cold_bin = np.searchsorted(np.asarray(COLD_SHARE_EDGES, np.float64), cold, side="left")
    return size_bin * (len(COLD_SHARE_EDGES) + 1) + cold_bin


@lru_cache(maxsize=None)
def mix_weights() -> dict[str, np.ndarray]:
    """Weights over the val feeds (mean 1) for the public and the private mix.

    A val feed in cell ``c`` gets ``share of c among the test feeds of the part
    / share of c among val feeds``. Raises if the part has feeds in a cell that
    validation does not cover, because no weight can repair that.
    """
    val, test = load_feed_index("val"), load_feed_index("test")
    val_cell = _cell(val.sizes, cold_share("val"))
    test_cell = _cell(test.sizes, cold_share("test"))
    n_cells = (len(SIZE_EDGES) + 1) * (len(COLD_SHARE_EDGES) + 1)
    val_share = np.bincount(val_cell, minlength=n_cells) / len(val_cell)
    public = public_mask(test.table["timestamp"].to_numpy())
    weights = {}
    for part in BOARD_PARTS:
        part_cell = test_cell[public if part == "public" else ~public]
        part_share = np.bincount(part_cell, minlength=n_cells) / len(part_cell)
        if ((part_share > 0) & (val_share == 0)).any():
            raise ValueError(f"the {part} part of test has feeds in a cell without val support")
        w = np.zeros(len(val_cell))
        covered = val_share[val_cell] > 0
        w[covered] = part_share[val_cell[covered]] / val_share[val_cell[covered]]
        weights[part] = w / w.mean()
    return weights


def weighted_mean_se(x: np.ndarray, weights: np.ndarray | None = None) -> Estimate:
    """Mean of ``x`` and its standard error, plain or weighted.

    The weighted standard error uses the effective sample size
    ``(sum w)^2 / sum w^2``.
    """
    if weights is None:
        return Estimate(x.mean(), x.std() / np.sqrt(len(x)))
    mean = np.average(x, weights=weights)
    n_effective = weights.sum() ** 2 / (weights ** 2).sum()
    return Estimate(mean, np.sqrt(np.average((x - mean) ** 2, weights=weights) / n_effective))


def compare(candidate_ndcg: np.ndarray, reference_ndcg: np.ndarray) -> MixComparison:
    """Paired difference of two per-feed NDCG arrays on val under the three weightings."""
    weights = mix_weights()
    delta = candidate_ndcg - reference_ndcg
    return MixComparison(
        val=weighted_mean_se(delta),
        public=weighted_mean_se(delta, weights["public"]),
        private=weighted_mean_se(delta, weights["private"]),
    )
