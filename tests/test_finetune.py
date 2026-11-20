"""The fine-tune on the validation days: recipe, folds, schedule, loss, weight average, leak discipline.

Most tests run on the CPU, the end-to-end ones on a small synthetic val and
test written to a temporary folder. The last test loads a stored member and
the real validation split; it needs the artefacts and a GPU.
"""
from __future__ import annotations

import dataclasses
import json
import math
import tomllib
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from avito_ranker import paths
from avito_ranker.data.loader import SplitTensors
from avito_ranker.ensemble.config import load_ensemble_config
from avito_ranker.features.names import FINETUNE_EXTRAS
from avito_ranker.model import Checkpoint, InputColumn, InputView, Ranker, RankerConfig, build_ranker, read_checkpoint
from avito_ranker.paths import path
from avito_ranker.training.finetune import (
    FinetuneRecipe,
    check_folds,
    finetune,
    learning_rate,
    load_finetune_recipe,
    load_finetune_splits,
    user_folds,
)
from avito_ranker.training.loss import mean_listwise_loss
from avito_ranker.training.predict import predict
from support import CONFIGS_TABLE, ENSEMBLE_DIR, RECIPE_DIR

N_FEEDS, N_ITEMS = 60, 300
VIEW = InputView((InputColumn("count_a", "log_count", rank_and_z=True), InputColumn("value_b", "clip_10")))
EXTRAS = ("extra_a", "extra_b", "extra_c")
TINY_RECIPE = FinetuneRecipe(file_tag="tiny", n_folds=3, seeds=2, learning_rate=1e-3,
                             extras_learning_rate_multiplier=10.0, weight_decay=0.02, gradient_clip=1.0, epochs=2,
                             warmup_steps=3, ema_decay=0.9, ema_start_epoch=0.5, max_feeds=8, max_candidates=256)


def _quiet(line: str) -> None:
    pass


# ----------------------------------------------------------------------------- a synthetic val and test
def _write_split(root: Path, split: str, rng: np.random.Generator) -> None:
    """Flat arrays, an item table, the two inputs of VIEW and the three EXTRAS for 60 small feeds."""
    sizes = rng.integers(5, 45, N_FEEDS)
    lengths = rng.integers(0, 90, N_FEEDS)
    users = rng.integers(1, 15, N_FEEDS)
    users[rng.random(N_FEEDS) < 0.15] = 0      # not-logged-in sessions
    n_cand, n_hist = int(sizes.sum()), int(lengths.sum())
    flat = root / "flat" / split
    flat.mkdir(parents=True)
    pl.DataFrame({
        "user_id": users.astype(np.int32),
        "M": sizes.astype(np.int32),
        "H": lengths.astype(np.int32),
        "cand_off": np.r_[0, np.cumsum(sizes)[:-1]].astype(np.int64),
        "hist_off": np.r_[0, np.cumsum(lengths)[:-1]].astype(np.int64),
    }).write_parquet(flat / "feeds.parquet")
    arrays = {
        "user_emb": rng.integers(-60, 60, (N_FEEDS, 128)).astype(np.int8),
        "cand_emb_idx": rng.integers(-1, N_ITEMS, n_cand).astype(np.int32),
        "cand_category_id": rng.integers(2, 54, n_cand).astype(np.int8),
        "hist_emb_idx": rng.integers(-1, N_ITEMS, n_hist).astype(np.int32),
        "hist_category_id": rng.integers(2, 54, n_hist).astype(np.int8),
        "hist_event_type": rng.integers(1, 11, n_hist).astype(np.int8),
        "hist_event_source": rng.integers(0, 31, n_hist).astype(np.int8),
        **{f"cand_f{k}": rng.random(n_cand).astype(np.float32) for k in range(1, 6)},
    }
    if split == "val":
        targets = np.zeros(n_cand, np.int8)
        clicked = rng.random(N_FEEDS) < 0.8    # some feeds without a click
        targets[(np.r_[0, np.cumsum(sizes)[:-1]] + rng.integers(0, sizes))[clicked]] = 1
        arrays["cand_target"] = targets
    for name, values in arrays.items():
        np.save(flat / f"{name}.npy", values)

    table = root / "competition" / split / "item_embeddings"
    table.mkdir(parents=True)
    rng.integers(-60, 60, (N_ITEMS, 128)).astype(np.int8).tofile(table / "data.memmap")
    (table / "data.meta").write_text(json.dumps({"dtype": "int8", "shape": [N_ITEMS, 128], "order": "C"}))

    cache = root / "feature_cache" / split
    cache.mkdir(parents=True)
    np.save(cache / "count_a.npy", rng.gamma(1.0, 30.0, n_cand).astype(np.float32))
    np.save(cache / "value_b.npy", rng.normal(0, 4, n_cand).astype(np.float32))
    for name in EXTRAS:
        np.save(cache / f"{name}.npy", rng.gamma(1.0, 40.0, n_cand).astype(np.float32))


@pytest.fixture
def tiny_splits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[SplitTensors, SplitTensors]]:
    """Synthetic val and test on the CPU, with the fine-tune extras attached."""
    root = tmp_path / "data"
    rng = np.random.default_rng(11)
    for split in ("val", "test"):
        _write_split(root, split, rng)
    (root / "paths.toml").write_text(
        f'[data]\nflat = "{root / "flat"}"\ncompetition = "{root / "competition"}"\n'
        f'feature_cache = "{root / "feature_cache"}"\n' + CONFIGS_TABLE)
    monkeypatch.setenv("AVITO_PATHS", str(root / "paths.toml"))
    paths._table.cache_clear()
    yield load_finetune_splits(VIEW, "cpu", EXTRAS)
    paths._table.cache_clear()


def _member() -> Checkpoint:
    """A small trained-looking member: random weights of width 32 without the fine-tune branch."""
    torch.manual_seed(5)
    model = Ranker(RankerConfig(view=VIEW, d_model=32))
    return Checkpoint(model.config, model.state_dict())


def _rows(split: SplitTensors, feeds: np.ndarray) -> np.ndarray:
    """Candidate rows of the given feeds."""
    return np.concatenate([np.arange(split.candidate_offsets[f], split.candidate_offsets[f] + split.feed_sizes[f])
                           for f in feeds])


# ----------------------------------------------------------------------------- recipe, folds, schedule, loss
def test_recipe_holds_the_production_values():
    recipe = load_finetune_recipe()
    ensembles = ("fd16p20", "fd16p20vf", "w8p20")
    tags = {load_ensemble_config(ENSEMBLE_DIR / f"{name}.toml").finetune_tag for name in ensembles}
    assert tags == {recipe.file_tag}            # the ensembles find the fine-tuned predictions by this tag
    assert dataclasses.replace(recipe, file_tag="") == FinetuneRecipe(
        file_tag="", n_folds=5, seeds=3, learning_rate=5e-5, extras_learning_rate_multiplier=10.0,
        weight_decay=0.02, gradient_clip=1.0, epochs=4, warmup_steps=30, ema_decay=0.99, ema_start_epoch=0.5,
        max_feeds=128, max_candidates=8192)


def test_folds_hold_whole_users_and_spread_the_anonymous_feeds():
    rng = np.random.default_rng(0)
    users = rng.integers(1, 300, 3000)
    users[rng.random(3000) < 0.05] = 0
    folds = user_folds(users, 5)
    check_folds(users, folds)
    for training, held_out in folds:
        assert abs(len(held_out) - 600) < 30
        assert (users[held_out] == 0).sum() > 15
        assert not set(users[training][users[training] > 0]) & set(users[held_out][users[held_out] > 0])


def test_check_folds_refuses_a_shared_user_and_a_feed_held_out_twice():
    users = np.array([1, 1, 2, 2, 0, 0])
    check_folds(users, [(np.array([2, 3, 5]), np.array([0, 1, 4])), (np.array([0, 1, 4]), np.array([2, 3, 5]))])
    with pytest.raises(ValueError, match="logged-in users"):
        check_folds(users, [(np.array([1, 3, 5]), np.array([0, 2, 4])), (np.array([0, 2, 4]), np.array([1, 3, 5]))])
    with pytest.raises(ValueError, match="exactly once"):
        check_folds(users, [(np.array([2, 3, 5]), np.array([0, 1, 4])), (np.array([2, 3, 5]), np.array([0, 1, 4]))])


def test_learning_rate_warms_up_then_falls_on_a_cosine_to_zero():
    recipe = load_finetune_recipe()
    total = 400
    rates = np.array([learning_rate(step, total, recipe) for step in range(total + 1)])
    assert rates[0] == pytest.approx(5e-5 / 30)
    assert np.all(np.diff(rates[:30]) > 0) and np.all(np.diff(rates[29:]) < 0)
    assert rates[total] == 0.0
    for step in (0, 7, 29, 30, 123, 399):  # the formula of the stored runs, in the same order of operations
        stored = 5e-5 * min(1.0, (step + 1) / 30)
        stored *= 0.5 * (1 + math.cos(math.pi * min(step, total) / total))
        assert learning_rate(step, total, recipe) == stored


def test_fine_tune_loss_averages_the_feeds_with_a_click():
    scores = torch.tensor([[2.0, 0.5, -1.0, 9.0], [0.3, 0.1, 0.2, 0.0], [1.0, 2.0, 3.0, 4.0]])
    mask = torch.tensor([[True, True, True, False], [True, True, True, True], [True, True, False, False]])
    targets = torch.tensor([[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    first = -torch.log_softmax(scores[0, :3], 0)[1]
    second = -(torch.log_softmax(scores[1], 0)[[0, 2]]).mean()
    loss = mean_listwise_loss(scores, targets, mask)
    assert loss.item() == pytest.approx(((first + second) / 2).item(), rel=1e-6)
    padded = scores.clone()
    padded[0, 3], padded[2, 2:] = -50.0, 100.0
    assert mean_listwise_loss(padded, targets, mask).item() == loss.item()
    assert mean_listwise_loss(scores, torch.zeros_like(targets), mask).item() == 0.0


def test_weight_average_is_the_update_of_the_stored_runs():
    """torch's AveragedModel with an EMA function: a copy at the start, then the same lerp the stored runs applied."""
    torch.manual_seed(0)
    model = torch.nn.Linear(5, 3)
    average, reference = None, None
    for _ in range(6):
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p))
        if average is None:
            average = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.99))
            reference = [p.detach().clone() for p in model.parameters()]
        else:
            torch._foreach_lerp_(reference, [p.detach() for p in model.parameters()], 1.0 - 0.99)
        average.update_parameters(model)
    for averaged, expected in zip(average.module.parameters(), reference, strict=True):
        assert torch.equal(averaged, expected)


# ----------------------------------------------------------------------------- end to end on the synthetic splits
def test_finetune_scores_every_row_and_reports_each_fold(tiny_splits):
    val, test = tiny_splits
    result = finetune(_member(), val, test, TINY_RECIPE, _quiet)
    assert result.val_scores.dtype == np.float32 and result.val_scores.shape == (val.n_candidates,)
    assert result.test_scores.dtype == np.float32 and result.test_scores.shape == (test.n_candidates,)
    assert np.all(result.val_scores != 0) and np.all(result.test_scores != 0)
    assert [r.fold for r in result.folds] == [0, 1, 2]
    assert sum(r.n_held_out_feeds for r in result.folds) == val.n_feeds
    assert all(len(r.held_out_ndcg_per_seed) == TINY_RECIPE.seeds for r in result.folds)


def test_out_of_fold_scores_do_not_depend_on_their_own_labels(tiny_splits):
    """Change the labels of one fold: its own rows keep their scores bit for bit, everything else moves."""
    val, test = tiny_splits
    member = _member()
    first = finetune(member, val, test, TINY_RECIPE, _quiet)

    held_out = user_folds(val.user_ids, TINY_RECIPE.n_folds)[0][1]
    rows = _rows(val, held_out)
    labels = val.targets.copy()
    labels[rows] = 1 - labels[rows]
    relabelled = dataclasses.replace(val, targets=labels, candidate_target=torch.from_numpy(labels.astype(np.int8)))
    second = finetune(member, relabelled, test, TINY_RECIPE, _quiet)

    others = np.setdiff1d(np.arange(val.n_candidates), rows)
    assert np.array_equal(first.val_scores[rows], second.val_scores[rows])
    assert not np.array_equal(first.val_scores[others], second.val_scores[others])
    assert not np.array_equal(first.test_scores, second.test_scores)


def test_finetune_refuses_a_member_with_a_branch_and_splits_without_extras(tiny_splits):
    val, test = tiny_splits
    member = _member()
    with_branch = Checkpoint(member.config.with_finetune_extras(3), build_ranker(member, 3).state_dict())
    with pytest.raises(ValueError, match="branch already"):
        finetune(with_branch, val, test, TINY_RECIPE, _quiet)
    with pytest.raises(ValueError, match="extras"):
        finetune(member, dataclasses.replace(val, finetune_extras=None), test, TINY_RECIPE, _quiet)


# ----------------------------------------------------------------------------- the stored member on the real split
def _first_wide_student() -> str:
    """The first stored run of student_wide.toml, the first full-view member of FD16P20."""
    with (RECIPE_DIR / "student_wide.toml").open("rb") as handle:
        return tomllib.load(handle)["record"]["members"][0]


# Held-out feeds per fold of the 14,423 validation feeds, 658 of them from not-logged-in sessions.
FOLD_SIZES = [2885, 2885, 2885, 2884, 2884]


@pytest.mark.artefacts
@pytest.mark.gpu
def test_finetune_starts_from_the_member_as_it_left_training(need):
    """Before the first step, the fine-tune model scores val exactly like the stored train-only predictions."""
    member = _first_wide_student()
    checkpoint_file = path("members.checkpoints") / f"{member}.pt"
    stored = path("outputs.predictions") / f"nn_{member}_trainonly_val.npy"
    need(checkpoint_file, stored)
    checkpoint = read_checkpoint(checkpoint_file)
    val, _ = load_finetune_splits(checkpoint.config.view, "cuda")
    torch.manual_seed(0)
    model = build_ranker(checkpoint, n_finetune_extras=len(FINETUNE_EXTRAS)).to("cuda")
    model.use_validation_domain()
    assert np.array_equal(predict(model, val), np.load(stored))

    folds = user_folds(val.user_ids, 5)
    check_folds(val.user_ids, folds)
    assert int((val.user_ids == 0).sum()) == 658
    assert [len(held_out) for _, held_out in folds] == FOLD_SIZES
