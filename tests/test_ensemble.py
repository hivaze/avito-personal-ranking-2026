"""The ensemble: the z-score kernel, pooling and mixing, the submission writer, the recipes."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from avito_ranker.ensemble.build import (
    Branches,
    build_ensemble,
    final_score,
    missing_member_files,
    mix_views,
    pool_members,
    popularity_light_weight,
    write_ensemble,
)
from avito_ranker.ensemble.config import (
    EnsembleConfig,
    Record,
    load_ensemble_config,
    member_file,
    pool_file,
    submission_file,
)
from avito_ranker.ensemble.feeds import FeedIndex, load_feed_index
from avito_ranker.ensemble.submission import (
    check_against_sample,
    submission_frame,
    tie_free_ranks,
    write_submission,
)
from avito_ranker.ensemble.zscore import feed_rank_z, feed_zscore
from avito_ranker.paths import path
from avito_ranker.validation.freeze import file_sha256
from support import ENSEMBLE_DIR


def make_feeds(sizes: list[int]) -> FeedIndex:
    """A feed index for synthetic feeds of the given sizes."""
    sizes = np.asarray(sizes, np.int64)
    offsets = np.r_[0, np.cumsum(sizes)[:-1]].astype(np.int64)
    table = pl.DataFrame({"feed_id": 100 + np.arange(len(sizes)), "M": sizes, "cand_off": offsets})
    return FeedIndex.from_table(table)


# ---------------------------------------------------------------- z-score kernel

def test_kernel_matches_numpy_feed_by_feed():
    rng = np.random.default_rng(0)
    feeds = make_feeds(list(rng.integers(2, 60, 300)))
    x = rng.normal(size=feeds.n_rows).astype(np.float32)
    out = feed_rank_z(feeds.offsets, feeds.sizes, x)
    assert out.dtype == np.float32 and out.shape == (feeds.n_rows, 2)
    for o, m in zip(feeds.offsets, feeds.sizes):
        v = x[o:o + m].astype(np.float64)
        np.testing.assert_array_equal(out[o:o + m, 0], (np.argsort(np.argsort(v)) / (m - 1)).astype(np.float32))
        np.testing.assert_allclose(out[o:o + m, 1], (v - v.mean()) / (v.std() + 1e-9), rtol=2e-4, atol=2e-5)


def test_kernel_on_constant_and_single_candidate_feeds():
    feeds = make_feeds([1, 4, 3])
    x = np.array([7.0, 2.0, 2.0, 2.0, 2.0, 1.0, 2.0, 3.0], np.float32)
    out = feed_rank_z(feeds.offsets, feeds.sizes, x)
    assert out[0, 0] == 0.5 and out[0, 1] == 0.0            # one candidate: rank 0.5, z 0
    np.testing.assert_array_equal(out[1:5, 1], np.zeros(4))   # constant feed: std + 1e-9 keeps z finite
    np.testing.assert_array_equal(out[5:, 0], [0.0, 0.5, 1.0])


def test_zscore_is_standardised_inside_each_feed_and_float32():
    rng = np.random.default_rng(1)
    feeds = make_feeds([5, 40, 17])
    x = rng.normal(loc=[3.0] * 5 + [-8.0] * 40 + [0.5] * 17, scale=2.0)   # float64 input, different feed levels
    z = feed_zscore(feeds, x)
    assert z.dtype == np.float32
    for o, m in zip(feeds.offsets, feeds.sizes):
        assert abs(z[o:o + m].mean()) < 1e-5
        assert z[o:o + m].std() == pytest.approx(1.0, abs=1e-4)


def test_zscore_does_not_depend_on_the_number_of_threads():
    import numba

    rng = np.random.default_rng(2)
    feeds = make_feeds(list(rng.integers(1, 80, 500)))
    x = rng.normal(size=feeds.n_rows).astype(np.float32)
    before = numba.get_num_threads()
    try:
        numba.set_num_threads(1)
        single = feed_zscore(feeds, x)
    finally:
        numba.set_num_threads(before)
    np.testing.assert_array_equal(single, feed_zscore(feeds, x))


# ---------------------------------------------------------------- pooling and mixing

def test_mix_views_and_final_score_follow_the_formulas():
    rng = np.random.default_rng(3)
    feeds = make_feeds([6, 9, 4])
    columns = [rng.normal(size=feeds.n_rows).astype(np.float32) for _ in range(4)]
    full, light = Branches(columns[0], columns[1]), Branches(columns[2], columns[3])
    mixed = mix_views(full, light, 0.2, feeds)
    z = lambda x: feed_zscore(feeds, x)   # noqa: E731
    np.testing.assert_array_equal(mixed.finetuned, 0.8 * z(full.finetuned) + 0.2 * z(light.finetuned))
    np.testing.assert_array_equal(mixed.train_only, 0.8 * z(full.train_only) + 0.2 * z(light.train_only))
    assert mixed.finetuned.dtype == np.float32
    final = final_score(mixed, feeds, 0.5)
    np.testing.assert_array_equal(final, 0.5 * z(mixed.finetuned) + 0.5 * z(mixed.train_only))
    assert final.dtype == np.float32


# ---------------------------------------------------------------- the feed-size rule

def rule_config(short: float | None, below: int | None, weight: float = 0.2) -> EnsembleConfig:
    """A recipe without members, for the weight functions: the feed-size rule when ``below`` is given."""
    return EnsembleConfig(
        name="RULE", full_view=("a",), popularity_light=("b",), popularity_light_weight=weight,
        finetuned_weight=0.5, finetune_tag="ft", pool_tag="RULE", submission="rule.parquet",
        reference_pool_tag="REF", record=Record("unset", 0.5, 0.4, "gate_RULE.txt"),
        popularity_light_weight_short=short, short_feed_below=below,
    )


def with_feed_size_rule(tiny_repo, short: float, below: int, long: float = 0.2) -> Path:
    """A copy of the tiny recipe whose popularity-light weight follows the feed-size rule."""
    text = tiny_repo.config_path.read_text()
    assert "popularity_light = 0.2\n" in text
    text = text.replace("popularity_light = 0.2\n", f"popularity_light = {long}\n"
                        f"popularity_light_short = {short}\nshort_feed_below = {below}\n")
    text = text.replace('pool_tag = "TINY"', 'pool_tag = "TINYRULE"').replace('"tiny.parquet"', '"tiny_rule.parquet"')
    rule_path = tiny_repo.root / "tiny_rule.toml"
    rule_path.write_text(text)
    return rule_path


def test_feed_size_rule_gives_one_float32_weight_per_row():
    feeds = make_feeds([3, 49, 50, 1, 120])
    weight = popularity_light_weight(rule_config(0.4, 50), feeds)
    assert weight.dtype == np.float32 and weight.shape == (feeds.n_rows,)
    expected = np.repeat(np.array([0.4, 0.4, 0.2, 0.4, 0.2], np.float32), feeds.sizes)   # short: fewer than 50
    np.testing.assert_array_equal(weight, expected)
    assert popularity_light_weight(rule_config(None, None), feeds) == 0.2   # no rule: the recipe's constant


def test_mix_views_with_the_feed_size_rule_mixes_each_feed_at_its_weight():
    rng = np.random.default_rng(8)
    feeds = make_feeds([6, 60, 4, 55, 49])
    columns = [rng.normal(size=feeds.n_rows).astype(np.float32) for _ in range(4)]
    full, light = Branches(columns[0], columns[1]), Branches(columns[2], columns[3])
    weight = popularity_light_weight(rule_config(0.4, 50), feeds)
    mixed = mix_views(full, light, weight, feeds)
    z = lambda x: feed_zscore(feeds, x)   # noqa: E731
    # The lab builder's expression, with the same float32 weight array.
    np.testing.assert_array_equal(mixed.finetuned, (1 - weight) * z(full.finetuned) + weight * z(light.finetuned))
    assert mixed.finetuned.dtype == np.float32 and mixed.train_only.dtype == np.float32
    # Feed by feed, the rows equal a plain mix at that feed's weight, bit for bit.
    at = {w: mix_views(full, light, w, feeds) for w in (0.2, 0.4)}
    for o, m in zip(feeds.offsets, feeds.sizes):
        plain = at[0.4 if m < 50 else 0.2]
        np.testing.assert_array_equal(mixed.finetuned[o:o + m], plain.finetuned[o:o + m])
        np.testing.assert_array_equal(mixed.train_only[o:o + m], plain.train_only[o:o + m])


def test_feed_size_rule_with_one_weight_everywhere_equals_the_plain_recipe(tiny_repo):
    """A rule that gives 0.2 to short feeds too builds the same branches and the same file as 0.2 alone."""
    plain = load_ensemble_config(tiny_repo.config_path)
    same = load_ensemble_config(with_feed_size_rule(tiny_repo, short=0.2, below=50))
    assert same.has_feed_size_rule and not plain.has_feed_size_rule
    built_plain, built_same = build_ensemble(plain), build_ensemble(same)
    for split in ("val", "test"):
        np.testing.assert_array_equal(built_same[split].finetuned, built_plain[split].finetuned)
        np.testing.assert_array_equal(built_same[split].train_only, built_plain[split].train_only)
    rebuild = tiny_repo.root / "rebuild"
    rebuild.mkdir()
    written_plain = write_ensemble(plain, built_plain, rebuild)
    written_same = write_ensemble(same, built_same, rebuild)
    assert file_sha256(written_plain[-1]) == file_sha256(written_same[-1])


def test_feed_size_rule_changes_only_the_short_feeds(tiny_repo):
    plain = build_ensemble(load_ensemble_config(tiny_repo.config_path))
    rule = build_ensemble(load_ensemble_config(with_feed_size_rule(tiny_repo, short=0.4, below=50)))
    for split in ("val", "test"):
        feeds = load_feed_index(split)
        short_rows = np.repeat(feeds.sizes < 50, feeds.sizes)
        assert 0 < short_rows.sum() < feeds.n_rows   # the tiny tree has feeds on both sides of 50
        for branch in ("finetuned", "train_only"):
            a, b = getattr(rule[split], branch), getattr(plain[split], branch)
            np.testing.assert_array_equal(a[~short_rows], b[~short_rows])
            assert not np.array_equal(a[short_rows], b[short_rows])


def test_recipe_reads_the_feed_size_rule(tiny_repo):
    config = load_ensemble_config(with_feed_size_rule(tiny_repo, short=0.4, below=50))
    assert (config.popularity_light_weight, config.popularity_light_weight_short, config.short_feed_below) == \
        (0.2, 0.4, 50)
    assert config.popularity_light_weight_text() == "0.4 on feeds of fewer than 50 candidates, 0.2 on the others"
    plain = load_ensemble_config(tiny_repo.config_path)
    assert plain.popularity_light_weight_short is None and plain.short_feed_below is None
    assert plain.popularity_light_weight_text() == "0.2"


@pytest.mark.parametrize("lines, message", [
    ("popularity_light_short = 0.4\n", "needs both"),
    ("short_feed_below = 50\n", "needs both"),
    ("popularity_light_short = 0.4\nshort_feed_below = 49.5\n", "integer"),
    ("popularity_light_short = 1.4\nshort_feed_below = 50\n", "outside"),
    ("popularity_light_shrot = 0.4\n", "unknown keys"),
])
def test_recipe_refuses_an_incomplete_or_misspelt_rule(tiny_repo, lines: str, message: str):
    text = tiny_repo.config_path.read_text().replace("popularity_light = 0.2\n", "popularity_light = 0.2\n" + lines)
    broken = tiny_repo.root / "broken.toml"
    broken.write_text(text)
    with pytest.raises(ValueError, match=message):
        load_ensemble_config(broken)


def test_pool_is_the_member_mean_of_z_scores(tiny_repo):
    config = load_ensemble_config(tiny_repo.config_path)
    feeds = load_feed_index("val")
    pooled = pool_members(config, config.full_view, "val", feeds)
    by_hand = np.mean(
        [feed_zscore(feeds, np.load(member_file(config, m, "train_only", "val"))) for m in config.full_view], 0)
    np.testing.assert_array_equal(pooled.train_only, by_hand)
    assert pooled.finetuned.dtype == np.float32 and len(pooled.finetuned) == feeds.n_rows


def test_build_reports_missing_member_files(tiny_repo):
    config = load_ensemble_config(tiny_repo.config_path)
    assert missing_member_files(config) == []
    member_file(config, "light_b", "finetuned", "test").unlink()
    assert [p.name for p in missing_member_files(config)] == ["nn_light_b_ft1_test.npy"]
    with pytest.raises(FileNotFoundError, match=r"nn_light_b_ft1_test\.npy"):
        build_ensemble(config)


def test_write_ensemble_stores_five_files_and_never_overwrites(tiny_repo):
    config = load_ensemble_config(tiny_repo.config_path)
    branches = build_ensemble(config)
    written = write_ensemble(config, branches)
    assert [p.name for p in written] == ["ens_TINY_ft_val.npy", "ens_TINY_to_val.npy", "ens_TINY_ft_test.npy",
                                         "ens_TINY_to_test.npy", "tiny.parquet"]
    assert all(p.exists() for p in written)
    np.testing.assert_array_equal(np.load(pool_file("TINY", "finetuned", "val")), branches["val"].finetuned)
    with pytest.raises(FileExistsError):
        write_ensemble(config, branches)
    # A rebuild into another folder gives the same bytes.
    other = tiny_repo.root / "rebuild"
    other.mkdir()
    write_ensemble(config, build_ensemble(config), other)
    assert file_sha256(other / "tiny.parquet") == file_sha256(submission_file(config))
    assert sorted(p.name for p in other.iterdir()) == sorted(p.name for p in written)   # no temporary file left


def test_write_ensemble_checks_the_submission_before_writing_anything(tiny_repo):
    config = load_ensemble_config(tiny_repo.config_path)
    branches = build_ensemble(config)
    sample = tiny_repo.root / "raw" / "sample_submission.parquet"
    full_sample = pl.read_parquet(sample)
    full_sample.head(len(full_sample) - 1).write_parquet(sample)       # a key the scores cover is missing
    rebuild = tiny_repo.root / "rebuild"
    rebuild.mkdir()
    with pytest.raises(ValueError, match="rows"):
        write_ensemble(config, branches, rebuild)
    assert list(rebuild.iterdir()) == []                               # no pool file was left behind
    full_sample.write_parquet(sample)
    assert len(write_ensemble(config, branches, rebuild)) == 5         # so the retry is not refused


def test_members_are_read_from_the_given_folder(tiny_repo):
    """Step 8 reads members from outputs.predictions unless told otherwise; runs write elsewhere."""
    config = load_ensemble_config(tiny_repo.config_path)
    runs = tiny_repo.root / "runs" / "predictions"
    runs.mkdir(parents=True)
    assert len(missing_member_files(config, runs)) == 4 * len(config.members)
    for file in tiny_repo.predictions.glob("nn_*.npy"):
        np.save(runs / file.name, -np.load(file))                      # a different set of member predictions
    stored, rebuilt = build_ensemble(config), build_ensemble(config, runs)
    np.testing.assert_allclose(rebuilt["val"].finetuned, -stored["val"].finetuned, atol=1e-6)
    assert member_file(config, "full_a", "train_only", "val", runs).parent == runs


# ---------------------------------------------------------------- submission writer

def test_ranks_are_a_permutation_of_each_feed_in_score_order():
    rng = np.random.default_rng(4)
    feeds = make_feeds([3, 10, 1, 25])
    scores = rng.normal(size=feeds.n_rows)
    ranks = tie_free_ranks(scores, feeds.feed_of_row, feeds)
    for o, m in zip(feeds.offsets, feeds.sizes):
        np.testing.assert_array_equal(np.sort(ranks[o:o + m]), np.arange(m))
        np.testing.assert_array_equal(ranks[o:o + m], np.argsort(np.argsort(scores[o:o + m])))


def test_exact_ties_are_broken_by_the_fixed_key():
    # All scores equal: the order is that of numpy's default_rng(0) uniform draws,
    # 0.637, 0.270, 0.041, 0.017, 0.813, 0.913, 0.607, 0.729. Changing the seed or
    # the generator changes the stored submissions.
    feeds = make_feeds([8])
    ranks = tie_free_ranks(np.zeros(8), feeds.feed_of_row, feeds)
    np.testing.assert_array_equal(ranks, [4, 2, 1, 0, 6, 7, 3, 5])
    np.testing.assert_array_equal(ranks, tie_free_ranks(np.zeros(8), feeds.feed_of_row, feeds))


def test_tie_key_only_decides_among_equal_scores():
    feeds = make_feeds([4, 4])
    scores = np.array([0.5, 0.5, 0.9, 0.1, 2.0, 2.0, 2.0, -1.0])
    ranks = tie_free_ranks(scores, feeds.feed_of_row, feeds)
    assert ranks[2] == 3 and ranks[3] == 0 and set(ranks[:2]) == {1, 2}
    assert ranks[7] == 0 and set(ranks[4:7]) == {1, 2, 3}


def test_submission_frame_has_the_file_schema_and_rejects_bad_scores():
    feeds = make_feeds([2, 3])
    items = np.array([11, 12, 21, 22, 23])
    frame = submission_frame([0.1, 0.2, 5.0, 4.0, 3.0], feeds.feed_of_row, items, feeds)
    assert frame.schema == {"feed_id": pl.Int32, "item_id": pl.Int32, "score": pl.Float32}
    assert frame["feed_id"].to_list() == [100, 100, 101, 101, 101]
    assert frame["score"].to_list() == [0.0, 1.0, 2.0, 1.0, 0.0]
    with pytest.raises(ValueError, match="NaN or infinity"):
        submission_frame([0.1, np.nan, 5.0, 4.0, 3.0], feeds.feed_of_row, items, feeds)
    with pytest.raises(ValueError, match="4 scores for 5"):
        submission_frame([0.1, 0.2, 5.0, 4.0], feeds.feed_of_row, items, feeds)


def test_check_against_sample_catches_missing_and_extra_keys():
    frame = pl.DataFrame({"feed_id": [1, 1, 2], "item_id": [10, 11, 20], "score": [0.0, 1.0, 0.0]})
    check_against_sample(frame, frame.select("feed_id", "item_id").sample(fraction=1.0, shuffle=True, seed=0))
    with pytest.raises(ValueError, match=r"does not cover|do not cover"):
        check_against_sample(frame.head(2), frame.select("feed_id", "item_id"))
    with pytest.raises(ValueError, match="rows"):
        check_against_sample(frame, frame.select("feed_id", "item_id").head(2))


def test_write_submission_writes_ranks_in_flat_row_order(tiny_repo):
    feeds = load_feed_index("test")
    scores = np.random.default_rng(5).normal(size=feeds.n_rows).astype(np.float32)
    out = write_submission(scores, tiny_repo.submissions / "one.parquet")
    frame = pl.read_parquet(out)
    assert frame.columns == ["feed_id", "item_id", "score"]
    np.testing.assert_array_equal(frame["item_id"].to_numpy(), np.load(tiny_repo.flat / "test/cand_item_id.npy"))
    for o, m in zip(feeds.offsets, feeds.sizes):
        np.testing.assert_array_equal(frame["score"].to_numpy()[o:o + m], np.argsort(np.argsort(scores[o:o + m])))


# ---------------------------------------------------------------- the shipped recipes

SHIPPED_RECIPES = ("fd16p20_m04_02", "fd16p20", "fd16p20vf", "w8p20")


@pytest.mark.parametrize("recipe, n_full, n_light", [("fd16p20_m04_02", 16, 4), ("fd16p20", 16, 4),
                                                     ("fd16p20vf", 16, 8), ("w8p20", 8, 4)])
def test_shipped_recipes_parse(recipe: str, n_full: int, n_light: int):
    config = load_ensemble_config(ENSEMBLE_DIR / f"{recipe}.toml")
    assert (len(config.full_view), len(config.popularity_light)) == (n_full, n_light)
    assert len(set(config.members)) == n_full + n_light
    assert config.popularity_light_weight == 0.2 and config.finetuned_weight == 0.5
    assert config.has_feed_size_rule == (recipe == "fd16p20_m04_02")
    assert len(config.record.sha256) == 64
    assert config.submission.endswith(".parquet")


def test_first_pick_is_fd16p20_with_the_feed_size_rule():
    """FD16P20@m0.4_0.2 has the members of FD16P20 in the same order; only the weight of short feeds differs."""
    pick, base = (load_ensemble_config(ENSEMBLE_DIR / f"{name}.toml") for name in ("fd16p20_m04_02", "fd16p20"))
    assert pick.full_view == base.full_view and pick.popularity_light == base.popularity_light
    assert (pick.popularity_light_weight_short, pick.short_feed_below) == (0.4, 50)
    assert (pick.finetune_tag, pick.reference_pool_tag) == (base.finetune_tag, base.reference_pool_tag)
    assert pick.pool_tag == "FD16P20_m04_02kd" and pick.submission == "cand_FD16P20_m04_02kd_r4.parquet"
    assert pick.record.sha256 == "3cf6faccbad1357ae1a1b11999fef8168835d77407684832c7578c832c6e71b3"
    assert (pick.record.val_ndcg, pick.record.public_ndcg) == (0.40293, 0.3952092)


@pytest.mark.artefacts
@pytest.mark.parametrize("recipe", SHIPPED_RECIPES)
def test_rebuild_reproduces_the_stored_submission(recipe: str, tmp_path: Path, need):
    """Pool the stored member predictions again and compare the file byte for byte."""
    config = load_ensemble_config(ENSEMBLE_DIR / f"{recipe}.toml")
    need(submission_file(config), path("data.raw") / "sample_submission.parquet", *missing_member_files(config))
    written = write_ensemble(config, build_ensemble(config), tmp_path)
    assert file_sha256(written[-1]) == config.record.sha256
    assert file_sha256(written[-1]) == file_sha256(submission_file(config))
    for rebuilt in written[:-1]:
        stored = path("outputs.predictions") / rebuilt.name
        need(stored)
        assert rebuilt.read_bytes() == stored.read_bytes()
