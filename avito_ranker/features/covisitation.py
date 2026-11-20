"""Co-visitation between a candidate and the items in the feed's history.

What it measures. Take other users' browsing sequences. If a candidate ``c``
tends to appear in them close to items that the present user has in the
history, ``c`` is a likely next step for this user. For every occurrence of
``c`` in a source sequence we look 20 positions to either side; each neighbour
``x`` that is also in the feed's history adds ``w / d``, where ``d`` is the
distance in the source sequence and ``w = 0.95 ** hpos`` is the recency of
``x`` in the feed's own history (its newest occurrence counts). Three inputs
come out:

``seq_cv_invd20``  the sum of ``w / d``
``seq_cv_pop``     the number of occurrences of ``c`` that were scanned (0 = the item is unknown)
``seq_cv_norm``    ``seq_cv_invd20 / sqrt(seq_cv_pop + 1)``

Only the 3,000 most recent occurrences of an item are scanned.

Source sequences and why they are safe. A source sequence is one history
snapshot, used as an ordered list of items.

* For val and test rows: the latest train snapshot of every user (every
  snapshot of the anonymous user 0, which is many people), plus every val and
  test snapshot taken strictly before the feed in question.
* For train rows of day ``d``: the same selection restricted to train feeds of
  days before ``d``. A train row sees only the past, like a test row.
* In both cases every sequence of the feed's own user is skipped, so neither
  the feed's own snapshot nor a later snapshot of the same user (which would
  contain the feed's clicks) can contribute. No labels are read.

Run time on 12 threads: 100 seconds for val and test together, 17 minutes
for train (one index per train day, each built from scratch).
"""
from __future__ import annotations

from collections.abc import Callable, Sequence

import numba as nb
import numpy as np
import polars as pl

from avito_ranker.data.flatten import (
    feed_layout,
    feed_offsets,
    moscow_day,
    read_feeds,
    read_flat,
)
from avito_ranker.features.store import n_candidates

NAMES = ("seq_cv_invd20", "seq_cv_pop", "seq_cv_norm")
MAX_OCCURRENCES = 3000  # most recent occurrences of a candidate that are scanned
WINDOW = 20  # neighbours this far on either side of an occurrence count
RECENCY_DECAY = 0.95  # weight of a history item per step of age in the feed's own history
MIN_FEEDS_PER_DAY = 1000  # a train day with fewer feeds is a boundary artefact, not a day of its own


def full_train_days() -> np.ndarray:
    """The train days that hold real traffic, as Moscow day numbers."""
    day = moscow_day(read_feeds("train")["timestamp"].to_numpy())
    days, counts = np.unique(day, return_counts=True)
    return days[counts > MIN_FEEDS_PER_DAY]


def source_feeds(before_day: int | None = None) -> np.ndarray:
    """Train feeds whose history snapshots serve as source sequences.

    Per known user the latest feed (its snapshot holds the last 256 events, so
    earlier ones add little), and every feed of user 0. ``before_day`` keeps
    only feeds of earlier days.
    """
    table = read_feeds("train")
    feeds = table.with_columns(day=pl.Series(moscow_day(table["timestamp"].to_numpy()), dtype=pl.Int64))
    if before_day is not None:
        feeds = feeds.filter(pl.col("day") < before_day)
    latest = (
        feeds.filter(pl.col("user_id") != 0)
        .sort(["user_id", "timestamp", "fidx"])
        .group_by("user_id")
        .agg(pl.col("fidx").last())["fidx"]
        .to_numpy()
    )
    anonymous = feeds.filter(pl.col("user_id") == 0)["fidx"].to_numpy()
    return np.sort(np.concatenate([latest, anonymous])).astype(np.int64)


@nb.njit(parallel=True, cache=True)
def _gather(sel, hist_off, n_hist, source, out_off, out):
    """Copy the history of each selected feed ``sel[q]`` to ``out[out_off[q]:]``, keeping its order."""
    for q in nb.prange(len(sel)):
        f = sel[q]
        for j in range(n_hist[f]):
            out[out_off[q] + j] = source[hist_off[f] + j]


class SequenceIndex:
    """Source sequences plus an inverted index from item to its occurrences.

    ``parts`` lists ``(split, feed indices, only_before)``. Sequences of a part
    with ``only_before`` set are used for a query only when their feed is
    strictly older than the querying feed.
    """

    def __init__(self, parts: Sequence[tuple[str, np.ndarray, bool]]):
        items, lengths, users, times, only_before = [], [], [], [], []
        for split, sel, constrained in parts:
            feeds = read_feeds(split)
            _, _, hist_off, n_hist = feed_layout(feeds)
            length = n_hist[sel]
            start = feed_offsets(length)
            part_items = np.empty(length.sum(), np.int32)
            _gather(sel, hist_off, n_hist, np.asarray(read_flat(split, "hist_item_id")), start, part_items)
            items.append(part_items)
            lengths.append(length)
            users.append(feeds["user_id"].to_numpy()[sel])
            times.append(feeds["timestamp"].to_numpy()[sel].astype(np.int64))
            only_before.append(np.full(len(sel), constrained, np.bool_))
        self.items = np.concatenate(items)
        self.lengths = np.concatenate(lengths).astype(np.int64)
        self.starts = feed_offsets(self.lengths)
        self.users = np.concatenate(users).astype(np.int64)
        self.times = np.concatenate(times)
        self.only_before = np.concatenate(only_before)

        sequence = np.repeat(np.arange(len(self.lengths), dtype=np.int32), self.lengths)
        position = (np.arange(len(self.items), dtype=np.int64) - np.repeat(self.starts, self.lengths)).astype(np.int16)
        order = np.argsort(self.items, kind="stable")
        self.item_ids, first, count = np.unique(self.items[order], return_index=True, return_counts=True)
        self.first = first.astype(np.int64)
        self.count = count.astype(np.int64)
        self.occ_sequence = sequence[order]
        self.occ_position = position[order]


@nb.njit(parallel=True, cache=True)
def _covisitation(sel, cand_off, n_cand, hist_off, n_hist, feed_user, feed_time, cand_item, hist_item, hist_pos,
                  items, starts, lengths, seq_user, seq_time, seq_only_before,
                  item_ids, first, count, occ_sequence, occ_position, out):
    """Fill ``out[row] = (invd20, pop, norm)`` for the candidates of the feeds in ``sel``."""
    for q in nb.prange(len(sel)):
        f = sel[q]
        h = n_hist[f]
        ho = hist_off[f]
        user = feed_user[f]
        time = feed_time[f]
        # distinct history items, sorted, each with the weight of its newest occurrence
        own = np.empty(h, np.int32)
        for j in range(h):
            own[j] = hist_item[ho + j]
        order = np.argsort(own)
        n_own = 0
        own_item = np.empty(h, np.int32)
        own_weight = np.empty(h, np.float64)
        for jj in range(h):
            j = order[jj]
            w = RECENCY_DECAY ** hist_pos[ho + j]
            if n_own > 0 and own_item[n_own - 1] == own[j]:
                if w > own_weight[n_own - 1]:
                    own_weight[n_own - 1] = w
            else:
                own_item[n_own] = own[j]
                own_weight[n_own] = w
                n_own += 1
        co = cand_off[f]
        for i in range(n_cand[f]):
            r = co + i
            out[r, 0] = 0.0
            out[r, 1] = 0.0
            out[r, 2] = 0.0
            c = cand_item[r]
            k = np.searchsorted(item_ids, c)
            if k >= len(item_ids) or item_ids[k] != c or n_own == 0:
                continue
            n_occ = count[k]
            occ_from = first[k] + max(0, n_occ - MAX_OCCURRENCES)
            occ_to = first[k] + n_occ
            scanned = 0.0
            for e in range(occ_from, occ_to):
                s = occ_sequence[e]
                if user != 0 and seq_user[s] == user:
                    continue
                if seq_only_before[s] and seq_time[s] >= time:
                    continue
                scanned += 1.0
                p = occ_position[e]
                start = starts[s]
                lo = max(0, p - WINDOW)
                hi = min(lengths[s], p + WINDOW + 1)
                for qq in range(lo, hi):
                    if qq == p:
                        continue
                    x = items[start + qq]
                    if x == c:
                        continue
                    kk = np.searchsorted(own_item[:n_own], x)
                    if kk < n_own and own_item[kk] == x:
                        out[r, 0] += own_weight[kk] / abs(qq - p)
            out[r, 1] = scanned
            out[r, 2] = out[r, 0] / np.sqrt(scanned + 1.0)


def _query(index: SequenceIndex, split: str, sel: np.ndarray, out: np.ndarray) -> None:
    """Fill the rows of ``out`` that belong to the feeds ``sel`` of ``split``, scanning the sequences of ``index``."""
    feeds = read_feeds(split)
    cand_off, n_cand, hist_off, n_hist = feed_layout(feeds)
    _covisitation(
        np.asarray(sel, np.int64), cand_off, n_cand, hist_off, n_hist,
        feeds["user_id"].to_numpy().astype(np.int64), feeds["timestamp"].to_numpy().astype(np.int64),
        np.asarray(read_flat(split, "cand_item_id")), np.asarray(read_flat(split, "hist_item_id")),
        np.asarray(read_flat(split, "hist_hpos")),
        index.items, index.starts, index.lengths, index.users, index.times, index.only_before,
        index.item_ids, index.first, index.count, index.occ_sequence, index.occ_position, out,
    )


def _columns(out: np.ndarray) -> dict[str, np.ndarray]:
    """The three columns of the kernel's output, keyed by file name."""
    return {name: out[:, i] for i, name in enumerate(NAMES)}


def build_eval(splits: Sequence[str] = ("val", "test")) -> dict[str, dict[str, np.ndarray]]:
    """The three inputs for val and/or test rows, keyed by split then by name.

    One index serves both splits: all of train, plus val and test snapshots
    under the strictly-earlier rule.
    """
    parts = [("train", source_feeds(), False)]
    parts += [(s, np.arange(len(read_feeds(s))), True) for s in ("val", "test")]
    index = SequenceIndex(parts)
    result = {}
    for split in splits:
        out = np.zeros((n_candidates(split), len(NAMES)), np.float32)
        _query(index, split, np.arange(len(read_feeds(split))), out)
        result[split] = _columns(out)
    return result


def train_feeds_by_day() -> list[tuple[int, np.ndarray]]:
    """``(day, feed indices)`` per full train day.

    The few feeds stamped on a boundary day join the previous full day, whose
    cut-off is still strictly earlier than they are.
    """
    day = moscow_day(read_feeds("train")["timestamp"].to_numpy())
    days = full_train_days()
    slot = np.searchsorted(days, day, side="right") - 1
    return [(int(d), np.nonzero(slot == i)[0]) for i, d in enumerate(days)]


def build_train(days: Sequence[int] | None = None,
                log: Callable[[str], None] | None = None) -> dict[str, np.ndarray]:
    """The three inputs for train rows, keyed by name.

    One index per train day, built from the feeds of earlier days. ``days``
    restricts the work to some days (rows of the others stay 0), which is what
    the tests use.
    """
    out = np.zeros((n_candidates("train"), len(NAMES)), np.float32)
    for day, sel in train_feeds_by_day():
        if days is not None and day not in days:
            continue
        source = source_feeds(before_day=day)
        if len(source) == 0 or len(sel) == 0:
            continue  # the first day has no past: its rows keep 0, the value of an unknown item
        index = SequenceIndex([("train", source, False)])
        _query(index, "train", sel, out)
        if log is not None:
            log(f"train day {day}: {len(sel):,} feeds against {len(index.lengths):,} source sequences")
        del index
    return _columns(out)
