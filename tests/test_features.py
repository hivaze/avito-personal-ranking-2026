"""The pre-computed inputs: flattening, the builders, their leak rules, and the stored arrays.

The first part runs anywhere. It builds a world of seven feeds whose expected
values can be worked out by hand, and checks each builder against them; the
worked examples double as documentation of the leak rules.

The second part (marker ``artefacts``) rebuilds inputs of the validation split
from the competition data and compares them with the stored arrays, bit for
bit. It takes about five minutes. One of these reads the training arguments
stored in the ten encoder checkpoints and compares them with the encoder
recipe. One test (marker ``gpu``) loads a stored encoder checkpoint and
re-scores the validation split.
"""
from __future__ import annotations

import datetime as dt
import json
import tomllib
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from avito_ranker import paths
from avito_ranker.data.flatten import flatten_split, moscow_day, read_feeds, read_flat
from avito_ranker.features import (
    EXPOSURE_INPUT,
    FINETUNE_EXTRAS,
    POPULARITY_INPUTS,
    RANKER_INPUTS,
    covisitation,
    exploration,
    exposure,
    feed_context,
    item_counts,
)
from avito_ranker.features.store import load_input, save_input
from avito_ranker.features.trajectory_encoder import corpus as encoder_corpus
from avito_ranker.features.trajectory_encoder import inputs as encoder_inputs
from avito_ranker.features.trajectory_encoder.files import (
    SCORE_KEYS,
    VARIANTS,
    checkpoint_file,
    content_files,
    corpus_dir,
    score_file,
)
from avito_ranker.features.within_feed import feed_max, rank_zscore_gap
from support import CONFIGS_TABLE, RECIPE_DIR

MOSCOW = dt.timezone(dt.timedelta(hours=3))


def moscow_midnight(day: str) -> int:
    """Unix time of 00:00 Moscow time on an ISO date."""
    return int(dt.datetime.fromisoformat(day + "T00:00:00").replace(tzinfo=MOSCOW).timestamp())


# ----------------------------------------------------------------------------- a world small enough to check by hand
A, B, C, D, E, X = 1, 2, 3, 4, 5, 6  # item ids; X has no content embedding
DAY1, DAY2, EVAL_DAY = moscow_midnight("2026-09-08"), moscow_midnight("2026-09-09"), moscow_midnight("2026-09-22")
ANONYMOUS = 0

# feed id -> (user, time, candidates, history newest first as (item, event type))
TRAIN_FEEDS = {
    100: (9, DAY1 + 100, [A, B], [(B, 1), (C, 1), (D, 2)]),
    101: (ANONYMOUS, DAY1 + 200, [A, C], [(C, 1), (A, 1)]),
    102: (9, DAY2 + 100, [A, D, X], [(A, 1), (B, 1), (C, 1)]),
    103: (8, DAY2 + 200, [B], [(E, 1), (A, 2)]),
}
VAL_FEEDS = {
    200: (7, EVAL_DAY + 100, [A, C, E], [(B, 1)]),
    201: (9, EVAL_DAY + 200, [C], [(B, 1)]),
}
TEST_FEEDS = {
    300: (7, EVAL_DAY + 300, [B], [(A, 1), (A, 2), (C, 1)]),  # a later snapshot of user 7: it holds A twice
}
TINY_PATHS = """
[data]
flat = "{root}/flat"
competition = "{root}/competition"
feature_cache = "{root}/feature_cache"
[encoder.corpus]
all = "{root}/encoder/all"
window1 = "{root}/encoder/window1"
[encoder.content]
partial = "{root}/encoder/content_partial.npy"
partial_mask = "{root}/encoder/content_partial_mask.npy"
full = "{root}/encoder/content_full.npy"
full_mask = "{root}/encoder/content_full_mask.npy"
"""


def _write_raw_split(root: Path, split: str, feeds: dict, table: np.ndarray, rng: np.random.Generator) -> None:
    """Write one split in the layout of the competition: a parquet with list columns and an embedding table."""
    def emb(item: int) -> int:
        return -1 if item == X else item

    rows = []
    for feed_id, (user, time, cands, history) in feeds.items():
        row = {
            "feed_id": feed_id, "timestamp": time, "user_id": user,
            "user_embedding": rng.integers(-50, 50, 128).tolist(),
            "item_id": cands, "category_id": [i % 5 for i in cands],
            "item_id__embedding_idx": [emb(i) for i in cands],
            "history__item_id": [i for i, _ in history], "history__category_id": [i % 5 for i, _ in history],
            "history__event_type": [e for _, e in history], "history__event_source": [1] * len(history),
            "history__item_id__embedding_idx": [emb(i) for i, _ in history],
        }
        for k in range(1, 6):
            row[f"f{k}"] = rng.random(len(cands)).tolist()
        if split == "train":
            row["pos"] = list(range(len(cands)))
        if split != "test":
            row["target"] = [1] + [0] * (len(cands) - 1)
        rows.append(row)
    lists = {"user_embedding": pl.List(pl.Int8), "item_id": pl.List(pl.Int32), "category_id": pl.List(pl.Int8),
             "item_id__embedding_idx": pl.List(pl.Int32), "history__item_id": pl.List(pl.Int32),
             "history__category_id": pl.List(pl.Int8), "history__event_type": pl.List(pl.Int8),
             "history__event_source": pl.List(pl.Int8), "history__item_id__embedding_idx": pl.List(pl.Int32),
             "pos": pl.List(pl.Int16), "target": pl.List(pl.Int8),
             **{f"f{k}": pl.List(pl.Float32) for k in range(1, 6)}}
    schema = {"feed_id": pl.Int32, "timestamp": pl.Int32, "user_id": pl.Int32,
              **{name: dtype for name, dtype in lists.items() if name in rows[0]}}
    folder = root / "competition" / split
    (folder / "impressions").mkdir(parents=True)
    shuffled = pl.DataFrame(rows, schema=schema).sample(fraction=1.0, shuffle=True, seed=1)
    shuffled.write_parquet(folder / "impressions" / "part.parquet")
    (folder / "item_embeddings").mkdir()
    table.tofile(folder / "item_embeddings" / "data.memmap")
    meta = {"dtype": "int8", "shape": list(table.shape), "order": "C"}
    (folder / "item_embeddings" / "data.meta").write_text(json.dumps(meta))


@pytest.fixture
def tiny_world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[np.ndarray]:
    """Seven feeds in the competition layout, flattened, with ``avito_ranker.paths`` pointing at them.

    Yields the item-embedding table (row = item id).
    """
    rng = np.random.default_rng(3)
    table = rng.integers(-100, 100, (8, 128)).astype(np.int8)
    for split, feeds in (("train", TRAIN_FEEDS), ("val", VAL_FEEDS), ("test", TEST_FEEDS)):
        _write_raw_split(tmp_path, split, feeds, table, rng)
    config = tmp_path / "paths.toml"
    config.write_text(TINY_PATHS.format(root=tmp_path) + CONFIGS_TABLE)
    monkeypatch.setenv("AVITO_PATHS", str(config))
    paths._table.cache_clear()
    for split in ("train", "val", "test"):
        flatten_split(split)
    yield table
    paths._table.cache_clear()


def _rows(split: str, feed_id: int) -> slice:
    """Candidate rows of a feed, looked up by its feed id."""
    feed = read_feeds(split).filter(pl.col("feed_id") == feed_id).row(0, named=True)
    return slice(feed["cand_off"], feed["cand_off"] + feed["M"])


def _unit(table: np.ndarray, item: int) -> np.ndarray:
    v = table[item].astype(np.float64)
    return v / np.linalg.norm(v)


# ----------------------------------------------------------------------------- flatten
def test_flatten_orders_feeds_by_time_and_keeps_history_newest_first(tiny_world):
    feeds = read_feeds("train")
    assert feeds["feed_id"].to_list() == [100, 101, 102, 103]  # the parquet was shuffled; time order is restored
    assert feeds["fidx"].to_list() == [0, 1, 2, 3]
    assert feeds["M"].to_list() == [2, 2, 3, 1] and feeds["H"].to_list() == [3, 2, 3, 2]
    assert feeds["cand_off"].to_list() == [0, 2, 4, 7] and feeds["hist_off"].to_list() == [0, 3, 5, 8]
    assert read_flat("train", "cand_item_id").tolist() == [A, B, A, C, A, D, X, B]
    assert read_flat("train", "cand_fidx").tolist() == [0, 0, 1, 1, 2, 2, 2, 3]
    assert read_flat("train", "cand_emb_idx").tolist() == [A, B, A, C, A, D, -1, B]
    assert read_flat("train", "hist_item_id").tolist() == [B, C, D, C, A, A, B, C, E, A]
    assert read_flat("train", "hist_hpos").tolist() == [0, 1, 2, 0, 1, 0, 1, 2, 0, 1]
    assert read_flat("train", "hist_event_type").tolist() == [1, 1, 2, 1, 1, 1, 1, 1, 1, 2]
    assert read_flat("train", "user_emb").shape == (4, 128)


def test_flatten_writes_labels_and_positions_only_where_they_exist(tiny_world):
    flat = paths.path("data.flat")
    assert (flat / "train" / "cand_pos.npy").exists() and (flat / "train" / "cand_target.npy").exists()
    assert not (flat / "val" / "cand_pos.npy").exists() and (flat / "val" / "cand_target.npy").exists()
    assert not (flat / "test" / "cand_pos.npy").exists() and not (flat / "test" / "cand_target.npy").exists()
    names = ("cand_item_id", "cand_category_id", "cand_f1", "cand_pos", "hist_hpos")
    dtypes = {name: read_flat("train", name).dtype for name in names}
    assert dtypes == {"cand_item_id": np.int32, "cand_category_id": np.int8, "cand_f1": np.float32,
                      "cand_pos": np.int16, "hist_hpos": np.int16}


def test_flatten_checks_the_stored_arrays_and_replaces_them_only_when_asked(tiny_world):
    folder = paths.path("data.flat") / "val"
    inode = (folder / "cand_item_id.npy").stat().st_ino
    flatten_split("val")                                     # the same input again: every file left alone
    assert (folder / "cand_item_id.npy").stat().st_ino == inode
    stored = np.load(folder / "cand_item_id.npy")
    np.save(folder / "cand_item_id.npy", stored + 1)          # a stored array that disagrees with the input
    with pytest.raises(FileExistsError, match="cand_item_id"):
        flatten_split("val")
    feeds = read_feeds("val")
    feeds.with_columns(feeds["M"] + 1).write_parquet(folder / "feeds.parquet")
    with pytest.raises(FileExistsError, match="feeds.parquet"):
        flatten_split("val")
    flatten_split("val", overwrite=True)
    assert np.array_equal(np.load(folder / "cand_item_id.npy"), stored)


# ----------------------------------------------------------------------------- item counts
def test_item_counts_leave_out_later_snapshots_of_the_same_user(tiny_world):
    tables = item_counts.load_count_tables()
    # A occurs 5 times in all snapshots: feeds 101, 102, 103 once each and twice in test feed 300.
    assert tables.global_counts[item_counts.ALL_EVENTS][A] == 5

    val = item_counts.leave_user_out_counts("val", tables, range(3))
    feed_200 = _rows("val", 200)  # user 7; the test feed 300 is a later snapshot of user 7 and holds A twice
    assert val["it_hist_all_luo"][feed_200].tolist() == [3.0, 3.0, 1.0]  # A: 5 - 2; C: 4 - 1; E: 1
    assert val["it_hist_et1_luo"][feed_200].tolist() == [2.0, 3.0, 1.0]  # A as type 1: 101, 102 (300 removed)
    assert val["it_hist_et2_luo"][feed_200].tolist() == [1.0, 0.0, 0.0]  # A as type 2: 103 (300 removed)

    test = item_counts.leave_user_out_counts("test", tables, [item_counts.ALL_EVENTS])
    # B occurs in 100, 102, 200, 201. Feed 200 is an earlier snapshot of the same user: it stays.
    assert test["it_hist_all_luo"].tolist() == [4.0]

    train = item_counts.leave_user_out_counts("train", tables, [item_counts.ALL_EVENTS])["it_hist_all_luo"]
    # Feed 100 of user 9: the later snapshots of user 9 are 102 (A, B) and val feed 201 (B).
    assert train[_rows("train", 100)].tolist() == [4.0, 2.0]
    # The anonymous user 0 is many people: nothing is subtracted.
    assert train[_rows("train", 101)].tolist() == [5.0, 4.0]


# ----------------------------------------------------------------------------- co-visitation
def test_covisitation_counts_neighbours_in_other_users_sequences(tiny_world):
    result = covisitation.build_eval(("val", "test"))
    val, test = result["val"], result["test"]
    # Sources: the latest train snapshot per user (102 for user 9, 103 for user 8), the anonymous 101,
    # and val / test snapshots that are strictly earlier than the querying feed.
    feed_200 = _rows("val", 200)  # user 7, history [B]
    # A sits next to B in sequence 102 ([A, B, C]): 1 / distance 1. It also occurs in 103 and 101.
    # The test snapshot 300 holds A too but is later (and of the same user): not scanned.
    np.testing.assert_allclose(val["seq_cv_invd20"][feed_200], [1.0, 1.0, 0.0])
    np.testing.assert_allclose(val["seq_cv_pop"][feed_200], [3.0, 2.0, 1.0])
    np.testing.assert_allclose(val["seq_cv_norm"][feed_200], [1 / np.sqrt(4), 1 / np.sqrt(3), 0.0], rtol=1e-6)

    # Feed 201 belongs to user 9. Sequence 102 is user 9's own and is skipped, although C sits next to B there.
    feed_201 = _rows("val", 201)
    assert val["seq_cv_invd20"][feed_201].tolist() == [0.0] and val["seq_cv_pop"][feed_201].tolist() == [1.0]

    # Feed 300: history [A, A, C]. In sequence 102, B has A and C at distance 1; the weights are
    # 0.95 ** 0 for A (its newest occurrence counts) and 0.95 ** 2 for C. Val feed 201 is earlier: scanned.
    np.testing.assert_allclose(test["seq_cv_invd20"], [1.0 + 0.95 ** 2], rtol=1e-6)
    assert test["seq_cv_pop"].tolist() == [2.0]


def test_covisitation_sources_for_a_train_day_come_from_earlier_days(tiny_world):
    day2 = int(moscow_day(np.array([DAY2]))[0])
    assert covisitation.source_feeds(before_day=day2).tolist() == [0, 1]  # feeds 100 and 101
    assert covisitation.source_feeds().tolist() == [1, 2, 3]  # latest of user 9, user 8, and the anonymous feed
    assert len(covisitation.source_feeds(before_day=day2 - 1)) == 0  # the first day has no past


# ----------------------------------------------------------------------------- exposure
def test_exposure_counts_only_earlier_train_days(tiny_world):
    built = exposure.build_exposure()
    # Day 1 shows A twice, B once, C once. Day-1 rows see nothing; day-2 rows see day 1.
    expected_train = np.log1p(np.array([0, 0, 0, 0, 2, 0, 0, 1], np.float32))
    np.testing.assert_array_equal(built["train"], expected_train)
    # Val and test rows see all of train: A 3, B 2, C 1, D 1, E 0.
    np.testing.assert_array_equal(built["val"], np.log1p(np.array([3, 1, 0, 1], np.float32)))
    np.testing.assert_array_equal(built["test"], np.log1p(np.array([2], np.float32)))
    assert built["train"].dtype == np.float32


def test_exposure_is_capped_at_fifty():
    np.testing.assert_array_equal(exposure.cap_and_log(np.array([0.0, 49.0, 50.0, 1e6])),
                                  np.log1p(np.array([0, 49, 50, 50], np.float32)))


# ----------------------------------------------------------------------------- feed context
def test_feed_context_reads_one_feed_only(tiny_world):
    table = tiny_world
    columns = feed_context.build_feed_context("val")
    assert set(columns) == {n for n in RANKER_INPUTS if n.startswith("ctx_")}
    feed_200 = _rows("val", 200)
    units = [_unit(table, item) for item in (A, C, E)]
    cosine = np.array([[u @ v for v in units] for u in units])
    np.fill_diagonal(cosine, -np.inf)
    np.testing.assert_allclose(columns["ctx_nd_max"][feed_200], cosine.max(1), atol=1e-5)
    # The soft competition hands out one unit of attention per history item; times M, it sums to M.
    np.testing.assert_allclose(columns["ctx_hs_t10"][feed_200].sum(), 3.0, rtol=1e-5)
    # The only history item is B. Its nearest candidate gets the whole hard count.
    nearest = int(np.argmax([u @ _unit(table, B) for u in units]))
    assert columns["ctx_hn_top1w"][feed_200].tolist() == [float(i == nearest) for i in range(3)]


def test_feed_context_is_missing_for_a_candidate_without_embedding(tiny_world):
    columns = feed_context.build_feed_context("train")
    feed_102 = _rows("train", 102)  # candidates A, D, X; X has no embedding
    for name, values in columns.items():
        assert np.isnan(values[feed_102][2]), name
        assert not np.isnan(values[feed_102][:2]).any(), name


def test_base_similarity_matches_numpy(tiny_world):
    table = tiny_world
    base = feed_context.base_similarity("val")
    feed_200 = _rows("val", 200)
    user = read_flat("val", "user_emb")[0].astype(np.float64)
    expected = [_unit(table, item) @ user / np.linalg.norm(user) for item in (A, C, E)]
    np.testing.assert_allclose(base["cos_user"][feed_200], expected, atol=1e-5)
    to_history = [_unit(table, i) @ _unit(table, B) for i in (A, C, E)]  # the history is the single item B
    np.testing.assert_allclose(base["max_history"][feed_200], to_history, atol=1e-5)


# ----------------------------------------------------------------------------- within-feed transforms, storage, names
def test_rank_zscore_gap_matches_numpy():
    rng = np.random.default_rng(0)
    sizes = rng.integers(1, 40, 200).astype(np.int64)
    offsets = np.r_[0, np.cumsum(sizes)[:-1]].astype(np.int64)
    x = rng.integers(0, 6, int(sizes.sum())).astype(np.float32)  # many ties
    out = np.empty((len(x), 3), np.float32)
    rank_zscore_gap(offsets, sizes, x[:, None], out)
    for o, m in zip(offsets, sizes):
        v = x[o:o + m].astype(np.float64)
        below, equal = (v[None, :] < v[:, None]).sum(1), (v[None, :] == v[:, None]).sum(1)
        rank = (below + (equal - 1) / 2) / (m - 1) if m > 1 else np.full(m, 0.5)
        np.testing.assert_allclose(out[o:o + m, 0], rank, atol=1e-6)
        np.testing.assert_allclose(out[o:o + m, 1], (v - v.mean()) / (v.std() + 1e-6), atol=1e-4)
        np.testing.assert_allclose(out[o:o + m, 2], v - v.max(), atol=1e-6)
    np.testing.assert_array_equal(feed_max(offsets, sizes, x), np.repeat(np.maximum.reduceat(x, offsets), sizes))


def test_input_names():
    assert len(RANKER_INPUTS) == len(set(RANKER_INPUTS)) == 34
    assert len(FINETUNE_EXTRAS) == len(set(FINETUNE_EXTRAS)) == 42
    assert len(POPULARITY_INPUTS) == 8 and set(POPULARITY_INPUTS) < set(RANKER_INPUTS)
    assert EXPOSURE_INPUT not in RANKER_INPUTS


def test_save_input_never_replaces_a_stored_array_by_other_values(tiny_world, tmp_path):
    values = np.array([0.0, 1.0, np.nan, 3.0])
    save_input(tmp_path / "out", "val", "demo", values)
    stored = load_input("val", "demo", folder=tmp_path / "out")
    assert stored.dtype == np.float32 and stored[[0, 1, 3]].tolist() == [0.0, 1.0, 3.0] and np.isnan(stored[2])
    save_input(tmp_path / "out", "val", "demo", values)  # the same values again: nothing to do
    with pytest.raises(FileExistsError):
        save_input(tmp_path / "out", "val", "demo", values + 1)
    save_input(tmp_path / "out", "val", "demo", values + 1, overwrite=True)
    assert load_input("val", "demo", folder=tmp_path / "out")[0] == 1.0
    with pytest.raises(ValueError):
        save_input(tmp_path / "out", "val", "short", values[:3])


# ----------------------------------------------------------------------------- exploration probability
def test_position_gaps_and_feed_spearman():
    position = np.array([0, 1, 2, 0, 2, 5], np.int64)
    offsets, sizes = np.array([0, 3], np.int64), np.array([3, 3], np.int64)
    assert exploration.position_gaps(position, offsets, sizes).tolist() == [0.0, 3.0]  # second feed misses 1, 3, 4
    predicted = np.array([0.1, 0.2, 0.3, 0.9, 0.5, 0.1])
    rho = exploration.feed_spearman(position.astype(np.float64), predicted, offsets, sizes)
    np.testing.assert_allclose(rho, [1.0, -1.0])
    constant = np.zeros(6)
    assert np.isnan(exploration.feed_spearman(position.astype(np.float64), constant, offsets, sizes)).all()


def test_mixture_separates_random_order_from_production_order():
    rng = np.random.default_rng(1)
    n_random, n_production = 3000, 7000
    z = np.r_[rng.normal(0, 1, n_random), rng.normal(4, 1, n_production)]
    log_gap = np.r_[rng.normal(-4, 0.3, n_random), rng.normal(-2.5, 0.3, n_production)]
    posterior = exploration._random_posterior(z, log_gap)
    assert posterior[:n_random].mean() > 0.97 and posterior[n_random:].mean() < 0.03


# ----------------------------------------------------------------------------- trajectory encoder
def test_encoder_targets_on_a_hand_made_sequence():
    # items oldest to newest; the last event is not a feed open
    item = np.array([5, 6, 5, 7, 8], np.int32)
    event_type = np.array([1, 1, 1, 1, 2], np.int8)
    event_source = np.ones(5, np.int8)
    seq_off = np.array([0, 5], np.int64)
    tgt_fe, tgt_new = np.empty(5, np.int32), np.empty(5, np.int32)
    encoder_corpus.next_item_targets(seq_off, item, event_type, event_source, tgt_fe, tgt_new)
    assert tgt_new.tolist() == [6, 7, 7, 8, -1]  # the repeat of 5 is not a new item
    assert tgt_fe.tolist() == [6, 7, 7, -1, -1]  # 8 is new but not opened from a feed
    several = np.empty((5, 4), np.int32)
    encoder_corpus.next_feed_opens(seq_off, item, event_type, event_source, several)
    assert several.tolist() == [[6, 7, -1, -1], [7, -1, -1, -1], [7, -1, -1, -1], [-1] * 4, [-1] * 4]
    assert (several[:, 0] == tgt_fe).all()


def test_encoder_corpus_takes_no_later_snapshot_of_a_labelled_user(tiny_world):
    selected = encoder_corpus.select_snapshots()
    # train: the anonymous feed 101, the latest snapshot of user 9 (102) and of user 8 (103)
    assert selected["train"].tolist() == [1, 2, 3]
    assert selected["val"].tolist() == [0, 1]  # feeds 200 and 201: the earliest snapshots of users 7 and 9
    assert selected["test"].tolist() == []  # feed 300 is a later snapshot of user 7, who has a labelled val feed

    counts = encoder_corpus.build_all_corpus()
    assert counts == {"train": 3, "val": 2, "test": 0}
    folder = corpus_dir("all")
    seq_off = np.load(folder / "seq_off.npy")
    tokens = np.load(folder / "tok_item.npy")
    assert seq_off.tolist() == [0, 2, 5, 7, 8, 9]
    assert tokens[seq_off[1]:seq_off[2]].tolist() == [C, B, A]  # feed 102, oldest first
    assert np.load(folder / "seq_split.npy").tolist() == [0, 0, 0, 1, 1]

    partial, full = np.load(content_files("partial")[1]), np.load(content_files("full")[1])
    # D occurs in a history, so both tables have it; X has no embedding anywhere.
    assert partial[[A, B, C, D, E]].all() and not partial[X] and not full[X]
    np.testing.assert_array_equal(np.load(content_files("full")[0])[A], tiny_world[A])


def test_encoder_window_corpus_holds_the_past_and_the_first_snapshot_inside(tiny_world):
    encoder_corpus.build_all_corpus()
    info = encoder_corpus.build_window_corpus(1)  # block 1 starts on Sep 9: feeds 102 and 103 are inside
    folder = corpus_dir("window1")
    assert np.load(folder / "window_feeds.npy").tolist() == [2, 3]
    assert info["n_background_sequences"] == 2 and info["n_window_sequences"] == 2  # 100 and 101; 102 and 103
    assert np.load(folder / "seq_split.npy").tolist() == [0, 0, 1, 1]
    tokens, seq_off = np.load(folder / "tok_item.npy"), np.load(folder / "seq_off.npy")
    assert tokens[seq_off[0]:seq_off[1]].tolist() == [D, C, B]  # feed 100: user 9's latest snapshot before the block


def test_alignment_matches_moments_and_keeps_the_sentinel():
    rng = np.random.default_rng(2)
    x = rng.normal(10, 2, 5000).astype(np.float32)
    reference = rng.normal(15, 4, 5000).astype(np.float32)
    x[:100] = encoder_inputs.SENTINEL
    line = encoder_inputs.alignment(x, reference)
    aligned = encoder_inputs._apply(x, line)
    assert (aligned[:100] == encoder_inputs.SENTINEL).all()
    assert abs(aligned[100:].mean() - reference.mean()) < 1e-3 and abs(aligned[100:].std() - reference.std()) < 1e-3


def test_clip_safe_keeps_missing_values_and_refuses_values_at_the_bounds():
    out = encoder_inputs.clip_safe(np.array([np.nan, -8.0, 16.0], np.float32))
    assert np.isnan(out[0]) and out[1:].tolist() == [2.0, 6.0] and out.dtype == np.float32
    with pytest.raises(ValueError):
        encoder_inputs.clip_safe(np.array([40.0], np.float32))


# ----------------------------------------------------------------------------- against the stored arrays
def _stored(split: str, name: str) -> Path:
    return paths.path("data.feature_cache") / split / f"{name}.npy"


def _assert_equals_stored(split: str, columns: dict[str, np.ndarray]) -> None:
    for name, values in columns.items():
        stored = np.load(_stored(split, name))
        assert values.dtype == stored.dtype == np.float32, name
        np.testing.assert_array_equal(values, stored, err_msg=f"{split}/{name}")


@pytest.mark.artefacts
def test_stored_item_counts_of_val(need):
    names = [n for n in POPULARITY_INPUTS if n.startswith("it_hist")]
    need(*(_stored("val", n) for n in names), paths.path("data.flat") / "train" / "hist_item_id.npy")
    tables = item_counts.load_count_tables(range(5))
    _assert_equals_stored("val", item_counts.leave_user_out_counts("val", tables, range(5)))


@pytest.mark.artefacts
def test_stored_covisitation_of_val_and_test(need):
    need(*(_stored(s, n) for s in ("val", "test") for n in covisitation.NAMES),
         paths.path("data.flat") / "train" / "hist_item_id.npy")
    for split, columns in covisitation.build_eval(("val", "test")).items():
        _assert_equals_stored(split, columns)


@pytest.mark.artefacts
def test_stored_feed_context_of_val(need):
    need(*(_stored("val", n) for n in RANKER_INPUTS if n.startswith("ctx_")),
         paths.path("data.competition") / "val" / "item_embeddings" / "data.memmap")
    _assert_equals_stored("val", feed_context.build_feed_context("val"))


@pytest.mark.artefacts
def test_stored_exposure(need):
    need(*(_stored(s, EXPOSURE_INPUT) for s in ("train", "val", "test")),
         paths.path("data.flat") / "train" / "cand_item_id.npy")
    for split, values in exposure.build_exposure().items():
        _assert_equals_stored(split, {EXPOSURE_INPUT: values})


@pytest.mark.artefacts
def test_stored_exploration_probability_from_its_stored_statistics(need):
    need(paths.path("exploration.probability"), paths.path("data.flat") / "train" / "feeds.parquet")
    stored = np.load(paths.path("exploration.probability"))
    posterior = exploration.mixture_posterior(stored["z"].astype(np.float64), stored["gap_share"].astype(np.float64))
    np.testing.assert_array_equal(posterior.astype(np.float32), stored["p_random"])
    # the 400,000 feeds the position proxy was fitted on have no probability
    used = exploration.proxy_feed_sample()[:exploration.ENCODING_FEEDS + exploration.TREE_FEEDS]
    assert np.isnan(stored["p_random"][used]).all()


def _encoder_score_files(content: str, target: str, models: tuple[str, ...]) -> list[Path]:
    return [score_file(content, target, model, key) for model in models for key in SCORE_KEYS]


@pytest.mark.artefacts
def test_stored_encoder_inputs_of_val_from_the_stored_raw_scores(need):
    models = ("no_test_id64", "no_test_id128_bias", "all_id64", "all_id128_bias")
    need(*_encoder_score_files("full", "val", models),
         *(_stored("val", n) for n in RANKER_INPUTS if n.startswith("pre2_")))
    _assert_equals_stored("val", encoder_inputs.ranker_inputs("val", sample_feeds=60000))


@pytest.mark.artefacts
def test_stored_encoder_checkpoints_were_trained_with_the_recipe(need):
    import torch

    from avito_ranker.features.trajectory_encoder.score import split_model_name

    recipe = tomllib.loads((RECIPE_DIR / "encoder.toml").read_text())
    names = [f"{corpus}_{variant}" for corpus in recipe["corpora"] for variant in VARIANTS]
    need(*(checkpoint_file(name) for name in names))
    model, training = recipe["model"], recipe["training"]
    for name in names:
        corpus, variant = split_model_name(name)
        # the arguments of the original training script, read without loading the weights
        args = torch.load(checkpoint_file(name), map_location="cpu", mmap=True, weights_only=False)["args"]
        expected = {
            "d": model["d"], "nl": model["layers"], "nh": model["heads"], "drop": model["dropout"],
            "min_cnt": model["min_item_count"], "epochs": training["epochs"], "budget": training["token_budget"],
            "lr": training["lr"], "id_lr": training["id_lr"], "id_drop": training["id_dropout"],
            "w_new": training["new_item_weight"], "K1": training["negatives_feed_open"],
            "K2": training["negatives_new_item"], "K3": training["negatives_uniform"],
            "max_pos": training["max_positions"], "multi": training["feed_open_targets"], "seed": training["seed"],
            "d_id": recipe["variants"][variant]["d_id"], "vt_rep": recipe["corpora"][corpus]["eval_oversampling"],
        }
        assert {key: args[key] for key in expected} == expected, name
        assert bool(args.get("id_bias")) == recipe["variants"][variant]["id_bias"], name
        # checkpoints made before the script had --item_tab were trained with its default, the partial table
        table = {"item_tab": "partial", "item_tab2": "full"}[args.get("item_tab") or "item_tab"]
        assert table == recipe["corpora"][corpus]["content_table"], name


@pytest.mark.artefacts
@pytest.mark.gpu
def test_stored_encoder_checkpoint_reproduces_its_stored_val_scores(need):
    import torch

    from avito_ranker.features.trajectory_encoder import score

    model_name = "no_test_id64"
    need(checkpoint_file(model_name), *content_files("full"), *_encoder_score_files("full", "val", (model_name,)))
    recipe = tomllib.loads((RECIPE_DIR / "encoder.toml").read_text())
    model, items = score.load_encoder(model_name, "full", recipe, torch.device("cuda:0"))
    split, feeds = score.target_feeds("val", recipe["scoring"]["sample_feeds"])
    scores = score.score_feeds(model, items, split, feeds, recipe["scoring"]["chunk_feeds_eval"])
    for key in SCORE_KEYS:
        stored = np.load(score_file("full", "val", model_name, key))
        # Measured after the port on the same GPU model: every value identical. The tolerance
        # allows for bf16 arithmetic on other hardware.
        np.testing.assert_allclose(scores[key], stored, atol=2e-2, err_msg=key)
        assert np.corrcoef(scores[key], stored)[0, 1] > 0.99999, key


def test_encoder_negative_pool_keeps_its_size_when_a_source_is_empty():
    """A batch without a new-item target used to stop the encoder's training; its share is drawn from the corpus."""
    import types

    import torch

    from avito_ranker.features.trajectory_encoder.train import _negative_pool

    items = types.SimpleNamespace(corpus_items=torch.arange(100, 110))
    cfg = {"negatives_feed_open": 4, "negatives_new_item": 3, "negatives_uniform": 5}
    torch.manual_seed(0)
    pool = _negative_pool(torch.tensor([1, 2]), torch.tensor([], dtype=torch.long), items, cfg)
    assert len(pool) == 12
    assert set(pool[:4].tolist()) <= {1, 2} and set(pool[4:].tolist()) <= set(range(100, 110))
