"""Flatten the nested competition parquet into flat arrays that can be memory-mapped.

The organisers ship one row per feed with list columns (the candidates, the
user's last 256 history events). Every later stage wants the opposite layout:
one long array per field, plus an offset table that says where each feed
starts. This module does that conversion once per split.

Layout of ``<data.flat>/<split>/``:

``feeds.parquet``
    One row per feed, sorted by ``(timestamp, feed_id)``: ``fidx`` (the row
    number, used everywhere as the feed key), ``feed_id``, ``timestamp``,
    ``user_id``, ``M`` (number of candidates), ``H`` (number of history
    events), ``cand_off`` and ``hist_off`` (start of the feed in the flat
    candidate and history arrays).
``user_emb.npy``
    int8 ``[n_feeds, 128]``.
``cand_<field>.npy``
    One value per candidate, feeds in ``fidx`` order, candidates in the order
    of the parquet: ``fidx``, ``item_id``, ``category_id``, ``f1`` .. ``f5``,
    ``emb_idx`` (row of the split's item-embedding table, -1 when the item has
    none) and, where the split has them, ``pos`` (production position, train
    only) and ``target`` (train and val).
``hist_<field>.npy``
    One value per history event, newest first inside a feed: ``fidx``,
    ``hpos`` (0 = newest), ``item_id``, ``category_id``, ``event_type``,
    ``event_source``, ``emb_idx``.

Run time with 12 polars threads: 17 s for train (246.7M candidates, 19 GB of
arrays), under a second each for val and test.

Every input, checkpoint row and prediction downstream is aligned with these
arrays, so a stored array is never replaced by different values unless
``overwrite`` is set (``avito_ranker.io.save_array``). Against the stored
arrays a re-run therefore checks them.

The readers at the bottom (``read_feeds``, ``read_flat``,
``read_item_embeddings``) are what the feature builders use to get the arrays
back, next to a few helpers for the layout (``feed_offsets``,
``rows_of_feeds``) and the calendar (``moscow_day``).

Timestamps are Unix seconds. Wherever the pipeline needs a day (a train day, a
validation weekday), it is the calendar day in Moscow time, UTC+3.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

from avito_ranker.io import save_array, save_frame
from avito_ranker.paths import path

SPLITS = ("train", "val", "test")

CANDIDATE_COLUMNS = ("item_id", "category_id", "f1", "f2", "f3", "f4", "f5", "item_id__embedding_idx")
HISTORY_COLUMNS = (
    "history__item_id",
    "history__category_id",
    "history__event_type",
    "history__event_source",
    "history__item_id__embedding_idx",
)
OPTIONAL_CANDIDATE_COLUMNS = ("pos", "target")
USER_EMBEDDING_DIM = 128
MOSCOW_OFFSET_SECONDS = 3 * 3600

# Sizes fixed by the data; the model reads them through avito_ranker.model.config.
N_EVENT_TYPES = 10     # history event types are coded 1..10; code 0 does not occur
N_EVENT_SOURCES = 32   # event sources are coded 0..30
N_CATEGORIES = 54      # category codes 0..53
MAX_HISTORY = 256      # the data keeps at most 256 history events per feed


def feed_offsets(lengths: np.ndarray) -> np.ndarray:
    """First row of every feed when feeds of the given lengths are laid end to end (int64)."""
    return np.r_[0, np.cumsum(lengths)[:-1]].astype(np.int64)


def rows_of_feeds(first_rows: np.ndarray, sizes: np.ndarray) -> np.ndarray:
    """The flat rows of a set of feeds, feed after feed.

    ``first_rows`` and ``sizes`` are the first row and the length of each
    chosen feed in the flat arrays (``cand_off`` and ``M``, or ``hist_off``
    and ``H``). Row ``j`` of the result is the flat row of the ``j``-th value
    when the chosen feeds are laid end to end.
    """
    sizes = np.asarray(sizes)
    return np.repeat(np.asarray(first_rows) - feed_offsets(sizes), sizes) + np.arange(int(sizes.sum()))


def moscow_day(timestamp: np.ndarray) -> np.ndarray:
    """Calendar day number (days since 1970-01-01) of Unix timestamps in Moscow time."""
    return (np.asarray(timestamp, dtype=np.int64) + MOSCOW_OFFSET_SECONDS) // 86400


def _candidate_name(column: str) -> str:
    return "emb_idx" if column == "item_id__embedding_idx" else column


def _history_name(column: str) -> str:
    return column.replace("history__", "").replace("item_id__embedding_idx", "emb_idx")


def flatten_split(split: str, out_dir: Path | None = None, competition_dir: Path | None = None,
                  overwrite: bool = False) -> dict[str, int]:
    """Write the flat arrays of one split and return its row counts.

    Feeds are sorted by ``(timestamp, feed_id)`` first, so ``fidx`` order is
    time order. Everything downstream (expanding-window statistics, the day
    loop of the co-visitation builder) relies on that. A stored file with the
    same values is left alone; one with other values is replaced only with
    ``overwrite``, after it is unlinked.
    """
    out = (out_dir or path("data.flat")) / split
    out.mkdir(parents=True, exist_ok=True)
    source = (competition_dir or path("data.competition")) / split / "impressions"
    frame = pl.read_parquet(sorted(source.glob("*.parquet"))).sort(["timestamp", "feed_id"])

    n_feeds = len(frame)
    n_cand = frame["item_id"].list.len().to_numpy().astype(np.int64)
    n_hist = frame["history__item_id"].list.len().to_numpy().astype(np.int64)
    feed_index = np.arange(n_feeds, dtype=np.int32)

    feeds = pl.DataFrame({
        "fidx": feed_index,
        "feed_id": frame["feed_id"],
        "timestamp": frame["timestamp"],
        "user_id": frame["user_id"],
        "M": n_cand.astype(np.int32),
        "H": n_hist.astype(np.int32),
        "cand_off": feed_offsets(n_cand),
        "hist_off": feed_offsets(n_hist),
    })
    save_frame(out / "feeds.parquet", feeds, overwrite)

    user_emb = frame["user_embedding"].list.to_array(USER_EMBEDDING_DIM).to_numpy()
    save_array(out / "user_emb.npy", user_emb.astype(np.int8), overwrite)

    save_array(out / "cand_fidx.npy", np.repeat(feed_index, n_cand), overwrite)
    present = [c for c in OPTIONAL_CANDIDATE_COLUMNS if c in frame.columns]
    for column in (*CANDIDATE_COLUMNS, *present):
        values = frame[column].explode().to_numpy()
        if len(values) != n_cand.sum():
            raise ValueError(f"{split}: column {column} has {len(values)} values, expected {n_cand.sum()}")
        save_array(out / f"cand_{_candidate_name(column)}.npy", values, overwrite)

    save_array(out / "hist_fidx.npy", np.repeat(feed_index, n_hist), overwrite)
    position = np.arange(n_hist.sum()) - np.repeat(feed_offsets(n_hist), n_hist)
    save_array(out / "hist_hpos.npy", position.astype(np.int16), overwrite)
    for column in HISTORY_COLUMNS:
        # Exploding an empty list yields one null; a feed without history must
        # contribute no rows, so the nulls are dropped.
        values = frame[column].explode().drop_nulls().to_numpy()
        if len(values) != n_hist.sum():
            raise ValueError(f"{split}: column {column} has {len(values)} values, expected {n_hist.sum()}")
        save_array(out / f"hist_{_history_name(column)}.npy", values, overwrite)

    return {"feeds": n_feeds, "candidates": int(n_cand.sum()), "history_events": int(n_hist.sum())}


def read_feeds(split: str) -> pl.DataFrame:
    """The feed table of a split (one row per feed, in ``fidx`` order)."""
    return pl.read_parquet(path("data.flat") / split / "feeds.parquet")


def read_flat(split: str, name: str, mmap: bool = True) -> np.ndarray:
    """One flat array of a split, for example ``read_flat("val", "cand_item_id")``."""
    return np.load(path("data.flat") / split / f"{name}.npy", mmap_mode="r" if mmap else None)


def has_flat(split: str, name: str) -> bool:
    """Whether the split has this array (``cand_pos`` and ``cand_target`` are optional)."""
    return (path("data.flat") / split / f"{name}.npy").exists()


def n_item_ids() -> int:
    """One more than the largest item id that occurs anywhere, as a candidate or in a history, in any split."""
    top = 0
    for split in SPLITS:
        for name in ("hist_item_id", "cand_item_id"):
            top = max(top, int(read_flat(split, name).max()) + 1)
    return top


def read_item_embeddings(split: str) -> np.ndarray:
    """The split's item-embedding table, int8 ``[n_items, 128]``, memory-mapped.

    ``cand_emb_idx`` and ``hist_emb_idx`` index its rows. Each split has its
    own table, so an index is only meaningful together with its split.
    """
    folder = path("data.competition") / split / "item_embeddings"
    meta = json.loads((folder / "data.meta").read_text())
    return np.memmap(folder / "data.memmap", dtype=np.int8, mode="r", shape=tuple(meta["shape"]))


def feed_layout(feeds: pl.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(cand_off, M, hist_off, H)`` of a feed table as int64 arrays, the form the numba kernels take."""
    return (
        feeds["cand_off"].to_numpy().astype(np.int64),
        feeds["M"].to_numpy().astype(np.int64),
        feeds["hist_off"].to_numpy().astype(np.int64),
        feeds["H"].to_numpy().astype(np.int64),
    )
