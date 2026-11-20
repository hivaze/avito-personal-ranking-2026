"""Training corpora of the trajectory encoder, and the item content tables.

A corpus is a set of history snapshots, each stored oldest to newest as one
sequence, with two self-supervised targets per position ``t`` (computed inside
the 256-event snapshot):

``tgt_new[t]``  the item of the first later event whose item has not occurred up to ``t``
``tgt_fe[t]``   the same, restricted to events of type 1 from source 1: an item
                opened from a feed, the closest thing in the histories to the
                competition's click
``tgt_fe_multi[t]``  the first four distinct such items; training draws one of them

Which snapshots go in decides whether the encoder leaks. A snapshot taken
after a labelled feed of the same user contains that feed's clicks, so:

* The corpus ``all`` takes, per known user, the latest train snapshot and
  only the earliest snapshot over val and test together. Every snapshot of
  user 0 (many anonymous people) is its own sequence. A val or test feed thus
  never meets a later snapshot of its user. Train rows cannot be scored with
  this corpus: the latest train snapshot is later than most of the user's
  train feeds.
* For train rows there is one corpus per block of train days (a "window").
  It takes, per known user, the latest snapshot before the block (background)
  and the earliest snapshot inside the block. A train feed of the block is
  scored by an encoder that never saw a later snapshot of its user, the same
  situation as a val or test feed. Train feeds before the first block get no
  encoder inputs (they stay NaN).

The content table maps an item id to its int8 content embedding. Two exist.
The ``partial`` table has the items that occur in some history or as a val or
test candidate. The ``full`` table adds the train candidates. With the partial
table "has content" on a train candidate meant "someone interacted with it
later", which is label information; all train-row scores therefore use the
full table. The partial table is kept because the encoder scores among the
fine-tune extras (val and test rows only) were built with it.

Run time on 12 threads: about one minute for the ``all`` corpus and the two
content tables (42 GiB of output), about one minute for the three window
corpora (38 GiB).
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import numba as nb
import numpy as np
import polars as pl

from avito_ranker.data.flatten import (
    MOSCOW_OFFSET_SECONDS,
    SPLITS,
    feed_offsets,
    n_item_ids,
    read_feeds,
    read_flat,
    read_item_embeddings,
)
from avito_ranker.features.trajectory_encoder.files import content_files, corpus_dir
from avito_ranker.io import save_array

TOKEN_FIELDS = {"item_id": "tok_item", "category_id": "tok_cat", "event_type": "tok_et", "event_source": "tok_es"}
N_FEED_OPEN_TARGETS = 4
MOSCOW = dt.timezone(dt.timedelta(seconds=MOSCOW_OFFSET_SECONDS))
# Blocks of train days, [start, end) in Moscow time. There is no traffic on Sep 15-16, and the
# last block is open-ended because a few feeds are stamped Sep 20 00:00.
WINDOW_DAYS = {1: ("2026-09-09", "2026-09-12"), 2: ("2026-09-12", "2026-09-17"), 3: ("2026-09-17", None)}


def window_bounds(window: int) -> tuple[int, int]:
    """Unix timestamps ``[start, end)`` of a block of train days."""
    def midnight(day: str) -> int:
        """Unix time of 00:00 Moscow time on an ISO date."""
        return int(dt.datetime.fromisoformat(day + "T00:00:00").replace(tzinfo=MOSCOW).timestamp())
    start, end = WINDOW_DAYS[window]
    return midnight(start), midnight(end) if end else 2**31 - 1


# ----------------------------------------------------------------------------- kernels
@nb.njit(parallel=True, cache=True)
def reverse_gather(hist_off, n_hist, source, out_off, out):
    """Copy the history of each selected feed into ``out``, reversed (stored newest first, wanted oldest first)."""
    for s in nb.prange(len(n_hist)):
        o, h, d = hist_off[s], n_hist[s], out_off[s]
        for k in range(h):
            out[d + k] = source[o + h - 1 - k]


@nb.njit(parallel=True, cache=True)
def next_item_targets(seq_off, item, event_type, event_source, tgt_fe, tgt_new):
    """Fill the two single targets of every position (-1 where there is none)."""
    for s in nb.prange(len(seq_off) - 1):
        a, b = seq_off[s], seq_off[s + 1]
        L = b - a
        first = np.empty(L, np.int32)  # position of the first occurrence of the item at j
        for j in range(L):
            first[j] = j
            x = item[a + j]
            for i in range(j):
                if item[a + i] == x:
                    first[j] = i
                    break
        for t in range(L):
            tgt_fe[a + t] = -1
            tgt_new[a + t] = -1
            for j in range(t + 1, L):
                if first[j] == j:
                    tgt_new[a + t] = item[a + j]
                    break
            for j in range(t + 1, L):
                if event_type[a + j] == 1 and event_source[a + j] == 1 and first[j] > t:
                    tgt_fe[a + t] = item[a + j]
                    break


@nb.njit(parallel=True, cache=True)
def next_feed_opens(seq_off, item, event_type, event_source, out):
    """Fill ``out[t]`` with the first ``out.shape[1]`` distinct new items opened from a feed after ``t`` (-1 padded)."""
    for s in nb.prange(len(seq_off) - 1):
        a, b = seq_off[s], seq_off[s + 1]
        L = b - a
        first = np.empty(L, np.int32)
        for j in range(L):
            first[j] = j
            x = item[a + j]
            for i in range(j):
                if item[a + i] == x:
                    first[j] = i
                    break
        for t in range(L):
            n = 0
            for r in range(out.shape[1]):
                out[a + t, r] = -1
            for j in range(t + 1, L):
                if event_type[a + j] == 1 and event_source[a + j] == 1 and first[j] > t:
                    x = item[a + j]
                    dup = False
                    for r in range(n):
                        if out[a + t, r] == x:
                            dup = True
                    if not dup:
                        out[a + t, n] = x
                        n += 1
                        if n == out.shape[1]:
                            break


@nb.njit(cache=True)
def _first_embedding_row(items, emb_idx, has, source_row):
    """For items not seen yet, remember the first embedding row that describes them."""
    for k in range(len(items)):
        e = emb_idx[k]
        if e >= 0:
            it = items[k]
            if not has[it]:
                has[it] = True
                source_row[it] = e


# ----------------------------------------------------------------------------- writing a corpus
def _write_sequences(out: Path, groups: list[tuple[int, dict[str, np.ndarray], np.ndarray, np.ndarray]],
                     n_items: int, overwrite: bool) -> dict[str, np.ndarray]:
    """Gather the histories of the selected feeds, compute targets and counts, write the corpus arrays.

    ``groups`` lists ``(label, history arrays, hist_off, H)`` of the selected
    feeds; the label ends up in ``seq_split``. Feeds without history are
    dropped. Returns the token arrays and the sequence offsets.
    """
    parts = {field: [] for field in TOKEN_FIELDS}
    lengths, labels = [], []
    for label, history, hist_off, n_hist in groups:
        out_off = feed_offsets(n_hist)
        for field in TOKEN_FIELDS:
            gathered = np.empty(int(n_hist.sum()), history[field].dtype)
            reverse_gather(hist_off, n_hist, history[field], out_off, gathered)
            parts[field].append(gathered)
        lengths.append(n_hist)
        labels.append(np.full(len(n_hist), label, np.int8))
    seq_off = np.concatenate([[0], np.cumsum(np.concatenate(lengths))]).astype(np.int64)
    tokens = {field: np.concatenate(parts[field]) for field in TOKEN_FIELDS}

    item, et, es = tokens["item_id"], tokens["event_type"], tokens["event_source"]
    tgt_fe = np.empty(len(item), np.int32)
    tgt_new = np.empty(len(item), np.int32)
    next_item_targets(seq_off, item, et, es, tgt_fe, tgt_new)
    tgt_fe_multi = np.empty((len(item), N_FEED_OPEN_TARGETS), np.int32)
    next_feed_opens(seq_off, item, et, es, tgt_fe_multi)

    save_array(out / "seq_off.npy", seq_off, overwrite)
    save_array(out / "seq_split.npy", np.concatenate(labels), overwrite)
    for field, name in TOKEN_FIELDS.items():
        save_array(out / f"{name}.npy", tokens[field], overwrite)
    save_array(out / "tgt_fe.npy", tgt_fe, overwrite)
    save_array(out / "tgt_new.npy", tgt_new, overwrite)
    save_array(out / "tgt_fe_multi.npy", tgt_fe_multi, overwrite)
    counted = {"item_cnt": item, "item_cnt_fe": tgt_fe[tgt_fe >= 0], "item_cnt_new": tgt_new[tgt_new >= 0]}
    for name, occurrences in counted.items():
        save_array(out / f"{name}.npy", np.bincount(occurrences, minlength=n_items).astype(np.int32), overwrite)
    return {"seq_off": seq_off, **tokens}


def _selected_history(split: str, feed_ids: np.ndarray, history: dict[str, np.ndarray] | None = None):
    """``(feed ids, history arrays, hist_off, H)`` of the selected feeds that have a history."""
    feeds = read_feeds(split)
    hist_off = feeds["hist_off"].to_numpy()[feed_ids].astype(np.int64)
    n_hist = feeds["H"].to_numpy()[feed_ids].astype(np.int64)
    keep = n_hist > 0
    if history is None:
        history = {field: np.ascontiguousarray(read_flat(split, f"hist_{field}")) for field in TOKEN_FIELDS}
    return feed_ids[keep], history, hist_off[keep], n_hist[keep]


def select_snapshots() -> dict[str, np.ndarray]:
    """Feeds whose snapshots form the ``all`` corpus, per split (see the module docstring for the rule)."""
    feeds = {split: read_feeds(split) for split in SPLITS}
    train = feeds["train"]
    latest = train.filter(pl.col("user_id") != 0).sort("timestamp").group_by("user_id").agg(pl.col("fidx").last())
    selected = {"train": np.sort(np.concatenate([
        latest["fidx"].to_numpy(), train.filter(pl.col("user_id") == 0)["fidx"].to_numpy()]))}
    labelled = pl.concat([
        feeds[split].with_columns(pl.lit(code).alias("sp")) for code, split in enumerate(("val", "test"))
    ])
    earliest = (
        labelled.filter(pl.col("user_id") != 0)
        .sort(["timestamp", "sp", "fidx"])
        .group_by("user_id")
        .agg(pl.col("fidx").first(), pl.col("sp").first())
    )
    for code, split in enumerate(("val", "test")):
        known = earliest.filter(pl.col("sp") == code)["fidx"].to_numpy()
        anonymous = feeds[split].filter(pl.col("user_id") == 0)["fidx"].to_numpy()
        selected[split] = np.sort(np.concatenate([known, anonymous]))
    return selected


def build_partial_content_table(n_items: int) -> tuple[np.ndarray, np.ndarray]:
    """Content of every item that occurs in a history or as a val or test candidate.

    Newer splits are read first, so a val or test candidate keeps the row of
    its own split's embedding table.
    """
    has = np.zeros(n_items, np.bool_)
    table = np.zeros((n_items, 128), np.int8)
    for split in ("test", "val", "train"):
        embeddings = read_item_embeddings(split)
        source_row = np.full(n_items, -1, np.int64)
        seen = has.copy()
        for kind in ("cand", "hist"):
            if kind == "cand" and split == "train":
                continue
            _first_embedding_row(read_flat(split, f"{kind}_item_id", mmap=False),
                                 read_flat(split, f"{kind}_emb_idx", mmap=False), seen, source_row)
        new = np.where(seen & ~has)[0]
        table[new] = np.asarray(embeddings)[source_row[new]] if split != "train" else embeddings[source_row[new]]
        has = seen
    return table, has


def add_train_candidates(table: np.ndarray, has: np.ndarray) -> np.ndarray:
    """Turn the partial table into the full one, in place: add the train candidates it lacks.

    Existing rows are kept. Returns the mask of the full table.
    """
    embeddings = read_item_embeddings("train")
    source_row = np.full(len(has), -1, np.int64)
    seen = has.copy()
    _first_embedding_row(read_flat("train", "cand_item_id", mmap=False),
                         read_flat("train", "cand_emb_idx", mmap=False), seen, source_row)
    new = np.where(seen & ~has)[0]
    order = np.argsort(source_row[new])
    table[new[order]] = embeddings[source_row[new[order]]]
    return seen


def build_all_corpus(overwrite: bool = False) -> dict[str, int]:
    """Write the ``all`` corpus and both content tables; return the sequence count per split."""
    out = corpus_dir("all")
    selected = select_snapshots()
    groups, sequence_feed, counts = [], [], {}
    for code, split in enumerate(SPLITS):
        feed_ids, history, hist_off, n_hist = _selected_history(split, selected[split])
        groups.append((code, history, hist_off, n_hist))
        sequence_feed.append(feed_ids.astype(np.int32))
        counts[split] = len(feed_ids)
    n_items = n_item_ids()
    _write_sequences(out, groups, n_items, overwrite)
    save_array(out / "seq_fidx.npy", np.concatenate(sequence_feed), overwrite)

    table, has = build_partial_content_table(n_items)
    table_file, mask_file = content_files("partial")
    save_array(table_file, table, overwrite)
    save_array(mask_file, has, overwrite)
    has = add_train_candidates(table, has)
    table_file, mask_file = content_files("full")
    save_array(table_file, table, overwrite)
    save_array(mask_file, has, overwrite)
    return counts


def build_window_corpus(window: int, overwrite: bool = False) -> dict:
    """Write the corpus of one block of train days and the list of train feeds it will score.

    Returns the counts of feeds, sequences and tokens.
    """
    out = corpus_dir(f"window{window}")
    start, end = window_bounds(window)
    feeds = read_feeds("train").select("fidx", "timestamp", "user_id", "H", "hist_off")
    known = pl.col("user_id") != 0

    def per_known_user(frame: pl.DataFrame, pick: pl.Expr) -> np.ndarray:
        return frame.filter(known).sort(["timestamp", "fidx"]).group_by("user_id").agg(pick)["fidx"].to_numpy()

    before = feeds.filter(pl.col("timestamp") < start)
    background = np.concatenate([
        per_known_user(before, pl.col("fidx").last()), before.filter(~known)["fidx"].to_numpy()])
    inside = feeds.filter((pl.col("timestamp") >= start) & (pl.col("timestamp") < end))
    in_window = np.concatenate([
        per_known_user(inside, pl.col("fidx").first()), inside.filter(~known)["fidx"].to_numpy()])
    save_array(out / "window_feeds.npy", np.sort(inside["fidx"].to_numpy()).astype(np.int64), overwrite)

    history = {field: read_flat("train", f"hist_{field}", mmap=False) for field in TOKEN_FIELDS}
    groups = []
    for label, feed_ids in enumerate((np.sort(background), np.sort(in_window))):
        _, _, hist_off, n_hist = _selected_history("train", feed_ids, history)
        groups.append((label, history, hist_off, n_hist))
    n_items = len(np.load(content_files("partial")[1], mmap_mode="r"))
    written = _write_sequences(out, groups, n_items, overwrite)

    return {"window": window, "start": start, "end": end, "n_window_feeds": len(inside),
            "n_background_sequences": len(groups[0][3]), "n_window_sequences": len(groups[1][3]),
            "tokens": int(written["seq_off"][-1])}
