"""Write a submission file from test scores.

The evaluator only looks at the order of candidates inside a feed, and it
punishes ties (negatives are placed first among equal scores). The file
therefore does not hold the model score. It holds the within-feed rank,
0 for the worst candidate up to M - 1 for the best: exact in float32, and
free of ties by construction.

Exact ties of the score do occur after the float32 arithmetic of the
ensemble: 3 to 7 of the 28,595 test feeds in the files this repository
rebuilds. They are broken by a random key drawn from a fixed seed, so the same
scores always give the same file.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from avito_ranker.data.flatten import read_flat
from avito_ranker.ensemble.feeds import FeedIndex, load_feed_index
from avito_ranker.paths import path

TIE_KEY_SEED = 0


def tie_free_ranks(scores: np.ndarray, feed_of_row: np.ndarray, feeds: FeedIndex) -> np.ndarray:
    """Rank every row inside its feed, 0 = lowest score, ties broken by the fixed key.

    ``feed_of_row`` is the feed index of each row and must be non-decreasing
    (the flat arrays are stored feed by feed).
    """
    scores = np.asarray(scores, dtype=np.float64)
    tie_key = np.random.default_rng(TIE_KEY_SEED).random(len(scores))
    order = np.lexsort((tie_key, scores, feed_of_row))
    ranks = np.empty(len(scores), np.float64)
    ranks[order] = feeds.row_in_feed
    return ranks


def submission_frame(scores: np.ndarray, feed_of_row: np.ndarray, item_ids: np.ndarray,
                     feeds: FeedIndex) -> pl.DataFrame:
    """Return the table that is written to disk: ``feed_id``, ``item_id``, ``score``."""
    scores = np.asarray(scores, dtype=np.float64)
    if len(scores) != len(item_ids):
        raise ValueError(f"{len(scores)} scores for {len(item_ids)} test candidates")
    if not np.isfinite(scores).all():
        raise ValueError("scores contain NaN or infinity")
    ranks = tie_free_ranks(scores, feed_of_row, feeds)
    feed_ids = feeds.table["feed_id"].to_numpy()[feed_of_row]
    frame = pl.DataFrame({
        "feed_id": feed_ids.astype(np.int32),
        "item_id": item_ids.astype(np.int32),
        "score": ranks.astype(np.float32),
    })
    if frame.select("feed_id", "score").is_duplicated().any():
        raise ValueError("two candidates of one feed share a rank")
    return frame


def check_against_sample(frame: pl.DataFrame, sample: pl.DataFrame) -> None:
    """Raise unless ``frame`` has exactly the (feed_id, item_id) keys of the sample submission."""
    joined = sample.select("feed_id", "item_id").join(frame, on=["feed_id", "item_id"], how="left")
    if joined["score"].null_count() != 0:
        raise ValueError("the sample submission has keys that the scores do not cover")
    if len(frame) != len(sample):
        raise ValueError(f"{len(frame)} rows, the sample submission has {len(sample)}")
    if not frame.select("feed_id", "item_id").is_unique().all():
        raise ValueError("duplicate (feed_id, item_id) keys")


def checked_submission_frame(scores: np.ndarray) -> pl.DataFrame:
    """The table of the submission for ``scores`` given in flat test row order, checked against the sample.

    Raises if the key set differs from the organisers' sample submission.
    Nothing is written.
    """
    feeds = load_feed_index("test")
    feed_of_row = read_flat("test", "cand_fidx", mmap=False)
    item_ids = read_flat("test", "cand_item_id", mmap=False)
    frame = submission_frame(scores, feed_of_row, item_ids, feeds)
    check_against_sample(frame, pl.read_parquet(path("data.raw") / "sample_submission.parquet"))
    return frame


def write_submission(scores: np.ndarray, out_path: str | Path) -> Path:
    """Write the submission parquet for ``scores`` given in flat test row order.

    The key set is checked against the organisers' sample submission before
    anything is written.
    """
    frame = checked_submission_frame(scores)
    out_path = Path(out_path)
    frame.write_parquet(out_path)
    return out_path
