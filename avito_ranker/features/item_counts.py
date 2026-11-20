"""Item-occurrence counts with the same user's later snapshots left out.

What it measures. Every feed carries a snapshot of its user's last 256 history
events. Counting how often an item occurs in the snapshots of all feeds of all
three splits gives a popularity signal that also covers the test days. One
count over all events (``it_hist_all_luo``) and one per event type
(``it_hist_et<e>_luo``).

Why the plain count is banned. About 92% of a feed's positives reappear in the
next history snapshot of the same user. A plain count over all snapshots would
therefore hand a train row its own label, and would do the same on test. So
for a candidate of user ``u`` at time ``t`` we subtract every occurrence that
comes from a snapshot of the same user (``user_id != 0``) taken after ``t``,
whatever split that snapshot is in. The rule is applied in the same way to
train, val and test rows. No labels are read.

What stays in: snapshots of other users (any time, any split) and the same
user's own and earlier snapshots. For a train row this includes other users'
snapshots from later train days, val and test, which a test row cannot see.
A causal rebuild lost 0.0018 to 0.0023 validation NDCG in two LightGBM
proxies, so the counts were kept.

Run time on 12 threads, for the three splits and all ten event types: 27
seconds for the shared tables, 68 seconds for the train rows (most of it the
join with the later history rows of the same user), about a second each for
val and test.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import polars as pl

from avito_ranker.data.flatten import N_EVENT_TYPES, SPLITS, n_item_ids, read_feeds, read_flat

ALL_EVENTS = 0  # key of the count over all event types


@dataclass
class CountTables:
    """What every split needs: global counts per item and the history rows that may have to be subtracted."""

    global_counts: dict[int, np.ndarray]  # event type (0 = all) -> occurrences per item id, over all splits
    later_history: pl.DataFrame  # history rows (user, item, th, et) of users that have two or more feeds
    multi_feed_users: np.ndarray  # those users


def count_name(event_type: int) -> str:
    """File name of the count for one event type (0 = all events)."""
    return "it_hist_all_luo" if event_type == ALL_EVENTS else f"it_hist_et{event_type}_luo"


def _global_counts(event_types: Sequence[int]) -> dict[int, np.ndarray]:
    """Occurrences of every item in the history snapshots of all feeds of all splits."""
    n_items = n_item_ids()
    counts = {e: np.zeros(n_items, np.int32) for e in event_types}
    for split in SPLITS:
        items = read_flat(split, "hist_item_id", mmap=False)
        types = read_flat(split, "hist_event_type", mmap=False)
        for e in event_types:
            chosen = items if e == ALL_EVENTS else items[types == e]
            counts[e] += np.bincount(chosen, minlength=n_items).astype(np.int32)
    return counts


def _multi_feed_users() -> np.ndarray:
    """Users with two or more feeds over all splits. Only they can have a later snapshot."""
    users = pl.concat([read_feeds(split).select("user_id") for split in SPLITS])
    counted = users.filter(pl.col("user_id") > 0).group_by("user_id").len()
    return counted.filter(pl.col("len") >= 2)["user_id"].to_numpy()


def _history_of(users: np.ndarray) -> pl.DataFrame:
    """History rows of the given users in all splits, each with the timestamp of the feed it belongs to."""
    parts = []
    for split in SPLITS:
        feeds = read_feeds(split)
        hist_feed = read_flat(split, "hist_fidx", mmap=False)
        owner = feeds["user_id"].to_numpy()[hist_feed]
        keep = np.isin(owner, users)
        parts.append(pl.DataFrame({
            "user": owner[keep].astype(np.int32),
            "item": read_flat(split, "hist_item_id", mmap=False)[keep],
            "th": feeds["timestamp"].to_numpy()[hist_feed][keep],
            "et": read_flat(split, "hist_event_type", mmap=False)[keep],
        }))
    return pl.concat(parts)


def load_count_tables(event_types: Sequence[int] = range(N_EVENT_TYPES + 1)) -> CountTables:
    """Build the shared tables once; ``leave_user_out_counts`` then serves each split from them."""
    users = _multi_feed_users()
    return CountTables(_global_counts(list(event_types)), _history_of(users), users)


def _later_same_user(split: str, tables: CountTables,
                     event_types: Sequence[int]) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """For the candidates that have any: how many occurrences sit in later snapshots of the same user.

    Returns the candidate rows and, per event type, the number to subtract.
    """
    feeds = read_feeds(split)
    cand_feed = read_flat(split, "cand_fidx", mmap=False)
    owner = feeds["user_id"].to_numpy()[cand_feed]
    rows = np.flatnonzero(np.isin(owner, tables.multi_feed_users))
    candidates = pl.DataFrame({
        "row": rows,
        "user": owner[rows].astype(np.int32),
        "item": read_flat(split, "cand_item_id", mmap=False)[rows],
        "tc": feeds["timestamp"].to_numpy()[cand_feed][rows],
    })
    per_type = [(pl.col("et") == e).sum().alias(f"n{e}") for e in event_types if e != ALL_EVENTS]
    joined = (
        candidates.lazy()
        .join(tables.later_history.lazy(), on=["user", "item"], how="inner")
        .filter(pl.col("th") > pl.col("tc"))
        .group_by("row")
        .agg([pl.len().alias("n0")] + per_type)
        .collect()
    )
    return joined["row"].to_numpy(), {e: joined[f"n{e}"].to_numpy() for e in event_types}


def leave_user_out_counts(split: str, tables: CountTables, event_types: Sequence[int]) -> dict[str, np.ndarray]:
    """The counts of one split, float32, keyed by file name (``it_hist_all_luo``, ``it_hist_et1_luo``, ...)."""
    items = read_flat(split, "cand_item_id", mmap=False)
    rows, later = _later_same_user(split, tables, event_types)
    out = {}
    for e in event_types:
        total = tables.global_counts[e][items].astype(np.float32)
        subtract = np.zeros(len(total), np.float32)
        subtract[rows] = later[e]
        out[count_name(e)] = total - subtract
    return out
