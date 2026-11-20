"""The exposure input of the popularity-light view (``xp_imp_c50``, ``names.EXPOSURE_INPUT``).

What it measures. How many times the item was shown as a candidate in train
feeds, capped at 50 and passed through ``log1p``. The popularity-light
ensemble members drop the eight popularity inputs (co-visitation and
leave-user-out counts) and keep only this coarse count; why, see
``configs/recipes/model_popularity_light.toml``.

Why it is safe. It counts impressions, never clicks, so no label enters. For a
train row of day ``d`` only train days before ``d`` are counted (an expanding
window; rows of the first day get 0). Val and test rows count all train days.
A train row therefore sees what a test row sees: the train impressions that
came before it. Days are Moscow calendar days.

Run time: 35 seconds on one CPU process (a sort of the 246.7M train candidate
item ids).
"""
from __future__ import annotations

import numpy as np

from avito_ranker.data.flatten import moscow_day, read_feeds, read_flat

CAP = 50.0


def train_impressions_before() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Impressions of each train candidate's item on earlier train days.

    Returns ``(before, item_ids, totals)``: the per-row count for train, and
    the sorted distinct train item ids with their impressions over all of
    train (the lookup table for val and test rows).
    """
    feeds = read_feeds("train")
    day = moscow_day(feeds["timestamp"].to_numpy())[read_flat("train", "cand_fidx", mmap=False)]
    item = read_flat("train", "cand_item_id", mmap=False).astype(np.int64)
    item_ids, code = np.unique(item, return_inverse=True)

    running = np.zeros(len(item_ids), np.float64)
    before = np.empty(len(item), np.float32)
    order = np.argsort(day, kind="stable")
    days = np.unique(day)
    bounds = np.r_[np.searchsorted(day[order], days), len(order)]
    for i in range(len(days)):
        rows = order[bounds[i]:bounds[i + 1]]
        codes = code[rows]
        before[rows] = running[codes]
        running += np.bincount(codes, minlength=len(item_ids))
    return before, item_ids, running


def lookup_total(item_ids: np.ndarray, totals: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Train impressions of each queried item, 0 for items never shown in train."""
    at = np.minimum(np.searchsorted(item_ids, query), len(item_ids) - 1)
    return np.where(item_ids[at] == query, totals[at], 0.0)


def cap_and_log(impressions: np.ndarray) -> np.ndarray:
    """``log1p(min(impressions, 50))`` as float32."""
    return np.log1p(np.minimum(impressions.astype(np.float32), CAP)).astype(np.float32)


def build_exposure(splits: tuple[str, ...] = ("train", "val", "test")) -> dict[str, np.ndarray]:
    """The exposure input for the requested splits, keyed by split."""
    before, item_ids, totals = train_impressions_before()
    out = {}
    for split in splits:
        if split == "train":
            out[split] = cap_and_log(before)
        else:
            query = read_flat(split, "cand_item_id", mmap=False).astype(np.int64)
            out[split] = cap_and_log(lookup_total(item_ids, totals, query))
    return out
