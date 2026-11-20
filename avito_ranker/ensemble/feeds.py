"""Where each feed sits in the flat candidate arrays of a split.

Predictions, labels and item ids are stored as one long array per split with
the feeds laid end to end. Every within-feed operation of the ensemble and of
the validation tools needs the same two facts about a feed: its first row and
its number of candidates. ``FeedIndex`` carries them together with the feed
table they came from.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from avito_ranker.data.flatten import read_feeds


@dataclass(frozen=True)
class FeedIndex:
    """Layout of one split: feed ``i`` owns rows ``offsets[i] : offsets[i] + sizes[i]``."""

    table: pl.DataFrame   # one row per feed: feed_id, user_id, timestamp, M, cand_off, ...
    sizes: np.ndarray     # int64, number of candidates of each feed
    offsets: np.ndarray   # int64, first row of each feed

    @classmethod
    def from_table(cls, table: pl.DataFrame) -> FeedIndex:
        """Build the index from a feed table with the columns ``M`` and ``cand_off``."""
        return cls(table, table["M"].to_numpy().astype(np.int64), table["cand_off"].to_numpy().astype(np.int64))

    @property
    def n_feeds(self) -> int:
        """Number of feeds of the split."""
        return len(self.sizes)

    @property
    def n_rows(self) -> int:
        """Number of candidate rows of the split."""
        return int(self.sizes.sum())

    @property
    def feed_of_row(self) -> np.ndarray:
        """Index of the feed that owns each row."""
        return np.repeat(np.arange(self.n_feeds), self.sizes)

    @property
    def row_in_feed(self) -> np.ndarray:
        """Position of each row inside its feed, counted from 0."""
        return np.arange(self.n_rows) - np.repeat(self.offsets, self.sizes)


def load_feed_index(split: str) -> FeedIndex:
    """Read the feed table of ``split`` ("val" or "test") from the flat data folder."""
    return FeedIndex.from_table(read_feeds(split))
