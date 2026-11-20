"""The trainer, its losses, schedule and optimisers, and the teacher files.

Most tests run on the CPU against small synthetic train, val and test splits
written to a temporary folder. Two tests read stored artefacts: the recipes
against the arguments logged by the stored members, and the teacher files
against the stored dumps.

The comparison of seeded runs of this trainer with the research trainer it
replaces needs the full train split on a GPU (about 40 GB) and is not a unit
test; ``docs/10-reproduce.md`` describes it.
"""
from __future__ import annotations

import ast
import json
import math
import tomllib
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch

from avito_ranker import paths
from avito_ranker.data.flatten import read_feeds
from avito_ranker.data.loader import load_split
from avito_ranker.model import Ranker, load_ranker, load_recipe, read_checkpoint
from avito_ranker.paths import path
from avito_ranker.training.cooldown import ExplorationBatches, exploration_pool, position_weight
from avito_ranker.training.loss import distillation_target, listwise_loss, production_order_loss
from avito_ranker.training.optim import Schedule, build_optimizers, muon_parameter_names
from avito_ranker.training.predict import predict
from avito_ranker.training.teacher import (
    TeacherRecipe,
    average_dumps,
    build_teacher,
    centre_per_feed,
    count_file,
    dump_train_scores,
    read_dump,
)
from avito_ranker.training.train import (
    TrainingRecipe,
    draw_holdout,
    input_names,
    plan_run,
    read_exploration_probability,
    run_length,
    train_member,
)
from avito_ranker.validation.gate import MemberLogs
from support import CONFIGS_TABLE, RECIPE_DIR

N_ITEMS = 300
SPLIT_FEEDS = {"train": 64, "val": 24, "test": 24}


# ----------------------------------------------------------------------------- synthetic splits
def _write_split(root: Path, split: str, rng: np.random.Generator) -> None:
    """Flat arrays, an item table and every input of both views for a small split."""
    n_feeds = SPLIT_FEEDS[split]
    sizes = rng.integers(5, 120, n_feeds)
    sizes[:12] = rng.integers(50, 120, 12)      # enough long feeds for a holdout and an exploration pool
    lengths = rng.integers(0, 300, n_feeds)
    n_cand, n_hist = int(sizes.sum()), int(lengths.sum())
    flat = root / "flat" / split
    flat.mkdir(parents=True)
    pl.DataFrame({
        "fidx": np.arange(n_feeds, dtype=np.int32),
        "user_id": rng.integers(0, 20, n_feeds).astype(np.int32),
        "M": sizes.astype(np.int32),
        "H": lengths.astype(np.int32),
        "cand_off": np.r_[0, np.cumsum(sizes)[:-1]].astype(np.int64),
        "hist_off": np.r_[0, np.cumsum(lengths)[:-1]].astype(np.int64),
    }).write_parquet(flat / "feeds.parquet")
    targets = np.zeros(n_cand, np.int8)
    clicked = rng.random(n_feeds) < 0.8          # some feeds without a click
    targets[(np.r_[0, np.cumsum(sizes)[:-1]] + rng.integers(0, sizes))[clicked]] = 1
    arrays = {
        "user_emb": rng.integers(-60, 60, (n_feeds, 128)).astype(np.int8),
        "cand_emb_idx": rng.integers(-1, N_ITEMS, n_cand).astype(np.int32),
        "cand_category_id": rng.integers(2, 54, n_cand).astype(np.int8),
        "hist_emb_idx": rng.integers(-1, N_ITEMS, n_hist).astype(np.int32),
        "hist_category_id": rng.integers(2, 54, n_hist).astype(np.int8),
        "hist_event_type": rng.integers(1, 11, n_hist).astype(np.int8),
        "hist_event_source": rng.integers(0, 31, n_hist).astype(np.int8),
        **{f"cand_f{k}": rng.random(n_cand).astype(np.float32) for k in range(1, 6)},
    }
    if split != "test":
        arrays["cand_target"] = targets
    if split == "train":
        arrays["cand_pos"] = np.concatenate([rng.permutation(m) for m in sizes]).astype(np.int16)
    for name, values in arrays.items():
        np.save(flat / f"{name}.npy", values)
    table = root / "competition" / split / "item_embeddings"
    table.mkdir(parents=True)
    rng.integers(-60, 60, (N_ITEMS, 128)).astype(np.int8).tofile(table / "data.memmap")
    (table / "data.meta").write_text(json.dumps({"dtype": "int8", "shape": [N_ITEMS, 128], "order": "C"}))
    cache = root / "feature_cache" / split
    cache.mkdir(parents=True)
    for file in sorted({c.file for name in ("full", "popularity_light") for c in load_recipe(name).view.columns}):
        np.save(cache / f"{file}.npy", rng.normal(0, 3, n_cand).astype(np.float32))


@pytest.fixture
def tiny_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Synthetic train, val and test splits, exploration probabilities and a teacher, with paths pointed at them."""
    root = tmp_path / "data"
    rng = np.random.default_rng(7)
    for split in SPLIT_FEEDS:
        _write_split(root, split, rng)
    sizes = pl.read_parquet(root / "flat" / "train" / "feeds.parquet")["M"].to_numpy()
    probability = rng.random(len(sizes)).astype(np.float32) * 0.4
    probability[:12] = 0.95                      # exploration-like long feeds
    probability[20] = np.nan                     # a feed without an estimate
    np.savez(root / "exploration.npz", fidx=np.arange(len(sizes)), p_random=probability)
    np.save(root / "teacher.npy", rng.normal(-100, 3, int(sizes.sum())).astype(np.float32))
    (root / "paths.toml").write_text(
        f'[data]\nflat = "{root / "flat"}"\ncompetition = "{root / "competition"}"\n'
        f'feature_cache = "{root / "feature_cache"}"\n'
        f'[outputs]\nruns = "{root / "runs"}"\n'
        f'[exploration]\nprobability = "{root / "exploration.npz"}"\n'
        f'[teachers]\ndumps = "{root / "dumps"}"\nmean11 = "{root / "teacher.npy"}"\n' + CONFIGS_TABLE)
    monkeypatch.setenv("AVITO_PATHS", str(root / "paths.toml"))
    paths._table.cache_clear()
    yield root
    paths._table.cache_clear()


def _tiny_recipe(base: str = "student_wide", **changes) -> TrainingRecipe:
    """A production recipe shrunk to a width-32 model, small batches and a few dozen steps."""
    recipe = TrainingRecipe.read(RECIPE_DIR / f"{base}.toml")
    small = dict(model=replace(recipe.model, d_model=32), epochs=3, warmup_steps=3, max_candidates=512, max_feeds=8,
                 cooldown_fraction=0.5, exploration_probability=0.5, eval_every=5)
    return replace(recipe, **{**small, **changes})


# ----------------------------------------------------------------------------- recipes
def _training_settings(recipe: TrainingRecipe) -> dict:
    """The settings that do not name the model, the teacher or the holdout."""
    skip = {"recipe", "width", "model_recipe", "teacher", "holdout_feeds", "select_from_step"}
    return {key: value for key, value in recipe.settings().items() if key not in skip}


def test_recipes_hold_the_production_settings():
    recipes = {name: TrainingRecipe.read(RECIPE_DIR / f"{name}.toml") for name in
               ("student_wide", "student_wide_t25", "student_popularity_light", "teacher_hard_label")}
    held_out = {name: TrainingRecipe.read(RECIPE_DIR / f"{name}_holdout.toml") for name in
                ("student_wide", "student_wide_t25", "student_popularity_light_narrow_t25")}
    for name, recipe in held_out.items():
        assert (recipe.holdout_feeds, recipe.select_from_step) == (30000, 12000)
        base = recipes.get(name, recipes["student_popularity_light"])
        assert _training_settings(recipe) == _training_settings(base)
    narrow = held_out["student_popularity_light_narrow_t25"]
    assert (narrow.model, narrow.teacher) == (load_recipe("popularity_light_narrow"), "mean25")
    for recipe in recipes.values():
        assert (recipe.learning_rate, recipe.weight_decay, recipe.warmup_steps, recipe.epochs) == (2e-3, 0.02, 500, 10)
        assert (recipe.cooldown_fraction, recipe.exploration_probability) == (0.2, 0.02)
        assert (recipe.position_weight, recipe.position_weight_end, recipe.position_temperature) == (0.5, 0.1, 10.0)
        assert (recipe.max_candidates, recipe.max_feeds, recipe.eval_every) == (32768, 512, 2500)
    assert [recipes[n].teacher for n in recipes] == ["mean11", "mean25", "mean11", None]
    assert [recipes[n].teacher_weight for n in recipes] == [0.5, 0.5, 0.5, 0.0]
    assert [recipes[n].model.d_model for n in recipes] == [192, 192, 192, 128]
    assert [recipes[n].model.view for n in recipes] == [load_recipe("full").view] * 2 + [
        load_recipe("popularity_light").view, load_recipe("full").view]
    assert [recipes[n].holdout_feeds for n in recipes] == [0, 0, 0, 30000]
    assert recipes["teacher_hard_label"].select_from_step == 12000


def test_a_teacher_needs_a_weight_and_a_weight_needs_a_teacher(tmp_path: Path):
    text = (RECIPE_DIR / "student_wide.toml").read_text().replace("teacher_weight = 0.5", "teacher_weight = 0.0")
    (tmp_path / "bad.toml").write_text(text)
    with pytest.raises(ValueError, match="teacher"):
        TrainingRecipe.read(tmp_path / "bad.toml")


# Every recipe of a training run: the distilled students and the hard-label teacher members.
TRAINING_RECIPES = sorted(p.stem for p in RECIPE_DIR.glob("*.toml") if p.stem.startswith(("student_", "teacher_hard")))


def stored_members(recipe: str) -> list[str]:
    """The stored runs a training recipe lists under ``[record]``."""
    with (RECIPE_DIR / f"{recipe}.toml").open("rb") as handle:
        return tomllib.load(handle)["record"]["members"]


def test_every_training_recipe_is_readable_and_lists_its_stored_runs():
    assert len(TRAINING_RECIPES) == 9
    for recipe in TRAINING_RECIPES:
        TrainingRecipe.read(RECIPE_DIR / f"{recipe}.toml")
        assert stored_members(recipe)


@pytest.mark.artefacts
@pytest.mark.parametrize("recipe", TRAINING_RECIPES)
def test_recipes_match_the_arguments_logged_by_the_stored_members(need, recipe: str):
    """The settings of a recipe are the arguments its first stored run logged, and so are its inputs."""
    member = stored_members(recipe)[0]
    log = path("outputs.member_logs") / f"{member}.log"
    need(log)
    lines = log.read_text().splitlines()
    line = next(line for line in lines if "args: {" in line)
    args = ast.literal_eval(line.split("args: ", 1)[1])
    assert args["variant"] == member
    r = TrainingRecipe.read(RECIPE_DIR / f"{recipe}.toml")
    assert (args["lr"], args["wd"], args["warmup"], args["epochs"]) == (r.learning_rate, r.weight_decay,
                                                                        r.warmup_steps, r.epochs)
    assert (args["sched"], args["cooldown_frac"], args["cooldown_mix"]) == ("wsd", r.cooldown_fraction,
                                                                            r.exploration_probability)
    assert (args["aux_pos"], args["aux_anneal"], args["aux_tau"], args["aux_mode"]) == (
        r.position_weight, r.position_weight_end, r.position_temperature, "listwise")
    assert (args["cand_budget"], args["max_feeds"], args["loss_norm"]) == (r.max_candidates, r.max_feeds, "feed")
    assert (args["d"], args["muon"], args["eval_every"]) == (r.model.d_model, True, r.eval_every)
    assert args.get("kd_file", "").endswith(f"teacher_{r.teacher}.npy" if r.teacher else "")
    if r.teacher:
        assert args["kd_w"] == r.teacher_weight and args["kd_T"] == 1.0
    assert bool(args.get("holdout_in_train")) == (r.holdout_feeds == 0)
    inputs_line = next(x for x in lines[lines.index(line):] if x.startswith("train-stage extra features"))
    assert ast.literal_eval(inputs_line.split(": ", 1)[1]) == input_names(r.model)
    if r.holdout_feeds:
        assert (args["holdout"], args["select_on"], args["select_min_step"]) == (r.holdout_feeds, "holdout",
                                                                                 r.select_from_step)


# ----------------------------------------------------------------------------- schedule and optimisers
def test_schedule_of_the_final_runs():
    schedule = Schedule.from_fraction(99840, 2e-3, 500, 0.2)
    assert schedule.cooldown_start == 79872                      # "cooldown mix: ... from step 79872" in the logs
    assert schedule.learning_rate(0) == 2e-3 / 500
    assert schedule.learning_rate(499) == 2e-3
    assert schedule.learning_rate(500) == schedule.learning_rate(79871) == schedule.learning_rate(79872) == 2e-3
    rates = [schedule.learning_rate(s) for s in range(79872, 99840)]
    assert all(a >= b for a, b in zip(rates, rates[1:]))
    assert rates[-1] == pytest.approx(2e-3 * (0.03 + 0.97 * (1 - math.sqrt(19967 / 19968))))
    assert schedule.decay_left(79872) == pytest.approx(1.0)
    assert schedule.decay_left(99839) == pytest.approx(1 - math.sqrt(19967 / 19968))


def test_production_order_weight_follows_the_learning_rate():
    schedule = Schedule.from_fraction(1000, 2e-3, 10, 0.2)
    assert position_weight(schedule, 0, 0.5, 0.1) == position_weight(schedule, 799, 0.5, 0.1) == 0.5
    assert position_weight(schedule, 800, 0.5, 0.1) == pytest.approx(0.5)
    for step in (850, 900, 999):
        expected = 0.1 + 0.4 * (schedule.learning_rate(step) / 2e-3 - 0.03) / 0.97
        assert position_weight(schedule, step, 0.5, 0.1) == pytest.approx(expected)
    assert position_weight(schedule, 999, 0.5, 0.1) < 0.15


def test_muon_takes_the_hidden_matrices_and_adamw_the_rest():
    torch.manual_seed(0)
    model = Ranker(load_recipe("full"))
    adamw, muon = build_optimizers(model, 2e-3, 0.02)
    to_muon = muon.param_groups[0]["params"]
    decay, no_decay = adamw.param_groups[0]["params"], adamw.param_groups[1]["params"]
    assert len(to_muon) == 46 and sum(p.numel() for p in to_muon) == 2_022_400
    assert all(p.ndim == 2 and min(p.shape) > 1 for p in to_muon)
    assert all(p.ndim == 2 for p in decay) and all(p.ndim < 2 for p in no_decay)
    assert adamw.param_groups[0]["weight_decay"] == 0.02 and adamw.param_groups[1]["weight_decay"] == 0.0
    assert adamw.defaults["betas"] == (0.9, 0.98)
    ids = [id(p) for p in to_muon + decay + no_decay]
    assert sorted(ids) == sorted(id(p) for p in model.parameters())
    names = muon_parameter_names(model)
    assert "category.weight" not in names and "wide.weight" not in names and "head.3.weight" not in names
    assert any(p is model.category.weight for p in decay)       # embedding tables get weight decay


# ----------------------------------------------------------------------------- losses
def _feeds() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Three feeds of sizes 4, 3 and 2 padded to 4: scores, clicks (the last feed has none) and the mask."""
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.bool)
    scores = torch.tensor([[0.5, -1.0, 2.0, 0.0], [1.0, 0.0, -0.5, 9.0], [0.3, 0.1, 5.0, 5.0]])
    clicks = torch.tensor([[0.0, 1.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    return scores, clicks, mask


def test_listwise_loss_is_the_feed_cross_entropy_over_max_feeds():
    scores, clicks, mask = _feeds()
    loss = listwise_loss(scores, clicks, mask, max_feeds=512)
    first = -(torch.log_softmax(scores[0], 0)[1] + torch.log_softmax(scores[0], 0)[2]) / 2
    second = -torch.log_softmax(scores[1, :3], 0)[0]
    assert loss.item() == pytest.approx(((first + second) / 512).item(), rel=1e-6)
    padded = scores.clone()
    padded[1, 3], padded[2, 2:] = -50.0, 7.0                    # padding and the click-free feed do not count
    assert listwise_loss(padded, clicks, mask, 512).item() == pytest.approx(loss.item(), rel=1e-6)


def test_distillation_target_mixes_clicks_and_the_teacher_softmax():
    _, clicks, mask = _feeds()
    teacher = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 99.0], [0.0, 0.0, 0.0, 0.0]])
    target = distillation_target(clicks, teacher, mask, 0.5)
    assert torch.allclose(target.sum(1), torch.tensor([1.0, 1.0, 0.0]))
    assert target[1, 3] == 0 and torch.all(target[2] == 0)
    expected = 0.5 * clicks[0] / 2 + 0.5 * torch.softmax(teacher[0], 0)
    assert torch.allclose(target[0], expected)
    assert torch.allclose(distillation_target(clicks, teacher, mask, 1e-9)[0], clicks[0] / 2, atol=1e-6)


def test_production_order_loss_weighs_feeds():
    scores, _, mask = _feeds()
    positions = torch.tensor([[0, 1, 2, 3], [2, 0, 1, 0], [1, 0, 0, 0]])
    weight = torch.tensor([1.0, 0.0, 1.0])
    loss = production_order_loss(scores, positions, mask, weight, 10.0, 0.5)
    other = scores.clone()
    other[1] = torch.tensor([-3.0, 4.0, 1.0, 0.0])               # a feed of weight 0 does not count
    assert production_order_loss(other, positions, mask, weight, 10.0, 0.5).item() == pytest.approx(loss.item())
    assert production_order_loss(scores, positions, mask, weight, 10.0, 0.25).item() == pytest.approx(loss.item() / 2)
    target = torch.softmax(-positions[0].float() / 10.0, 0)
    first = -(target * torch.log_softmax(scores[0], 0)).sum()
    third = -(torch.softmax(-positions[2, :2].float() / 10.0, 0) * torch.log_softmax(scores[2, :2], 0)).sum()
    assert loss.item() == pytest.approx((0.5 * (first + third) / 2).item(), rel=1e-6)


# ----------------------------------------------------------------------------- feeds of a run
def test_exploration_probability_holdout_and_pool(tiny_tree: Path):
    sizes = read_feeds("train")["M"].to_numpy()
    p = read_exploration_probability(sizes)
    assert p.dtype == np.float32 and p[20] == 0.0 and np.all(p[sizes < 50] == 0.0)
    holdout = draw_holdout(p, sizes, 5)
    assert len(holdout) == 5 and np.all(np.diff(holdout) > 0)
    assert np.all(p[holdout] > 0.9) and np.all(sizes[holdout] >= 50)
    assert np.array_equal(holdout, draw_holdout(p, sizes, 5))
    feeds = np.setdiff1d(np.arange(len(sizes)), holdout)
    pool = exploration_pool(feeds, p)
    assert set(pool) == {f for f in feeds if p[f] > 0.5} and not set(pool) & set(holdout)


def test_exploration_batches_cycle_through_the_pool(tiny_tree: Path):
    split = load_split("train", "full", device="cpu")
    pool = np.arange(12)
    explorer = ExplorationBatches(split, pool, 0.3, seed=5, max_candidates=512, max_feeds=4)
    n_batches = len(explorer.plan)
    first_pass = np.concatenate([explorer.next() for _ in range(n_batches)])
    assert sorted(first_pass) == list(pool)
    second = explorer.next()                                  # used up: a new shuffled plan
    assert explorer.position == 1 and explorer.used == n_batches + 1 and len(second) > 0
    draws = [ExplorationBatches(split, pool, 0.3, 5, 512, 4).replaces() for _ in range(3)]
    assert len(set(draws)) == 1                               # the same seed draws the same way


# ----------------------------------------------------------------------------- a run end to end
def test_a_run_writes_its_outputs_and_the_gate_reads_its_inputs(tiny_tree: Path):
    recipe = _tiny_recipe()
    out = tiny_tree / "runs"
    files = train_member(recipe, seed=1, name="tiny_student", out_dir=out, device="cpu")
    for split in ("val", "test"):
        scores = np.load(files.predictions[split])
        assert scores.dtype == np.float32 and len(scores) == read_feeds(split)["M"].sum()
    checkpoint = read_checkpoint(files.checkpoint)
    assert checkpoint.config == recipe.model
    assert checkpoint.info["kept_step"] == checkpoint.info["total_steps"]
    model = load_ranker(files.checkpoint)
    val = load_split("val", recipe.model.view, device="cpu")
    assert np.array_equal(predict(model, val), np.load(files.predictions["val"]))
    assert MemberLogs(out / "logs").training_inputs("tiny_student") == input_names(recipe.model)
    assert input_names(recipe.model)[-1] == "has:pre2_fe"
    with pytest.raises(FileExistsError):
        train_member(recipe, seed=1, name="tiny_student", out_dir=out, device="cpu")


def test_the_seed_fixes_the_run(tiny_tree: Path):
    recipe = _tiny_recipe(epochs=1)
    out = tiny_tree / "runs"
    scores = [np.load(train_member(recipe, seed, name, out, "cpu").predictions["val"])
              for seed, name in ((3, "a"), (3, "b"), (4, "c"))]
    assert np.array_equal(scores[0], scores[1])
    assert not np.array_equal(scores[0], scores[2])


def test_a_selection_step_past_the_run_fails_before_the_split_is_loaded(tiny_tree: Path, monkeypatch):
    recipe = _tiny_recipe("teacher_hard_label", holdout_feeds=5, select_from_step=10**6, eval_every=5)
    total = run_length(recipe)
    assert total == plan_run(replace(recipe, select_from_step=1), seed=2).total
    monkeypatch.setattr("avito_ranker.training.train.load_split", lambda *a, **k: pytest.fail("split loaded"))
    with pytest.raises(ValueError, match=f"must lie in 1..{total}"):
        train_member(recipe, seed=2, name="too_late", out_dir=tiny_tree / "runs", device="cpu")


def test_a_holdout_run_keeps_the_best_checkpoint_on_the_holdout(tiny_tree: Path):
    recipe = _tiny_recipe("teacher_hard_label", holdout_feeds=5, select_from_step=5, eval_every=5)
    files = train_member(recipe, seed=2, name="tiny_member", out_dir=tiny_tree / "runs", device="cpu")
    info = read_checkpoint(files.checkpoint).info
    assert info["kept_step"] % 5 == 0 or info["kept_step"] == info["total_steps"]
    assert 0.0 <= info["holdout_ndcg"] <= 1.0
    log = files.log.read_text()
    assert "holdout: 5 exploration-like train feeds kept out of training" in log
    assert f"holdout_ndcg {info['holdout_ndcg']:.5f}" in log


# ----------------------------------------------------------------------------- teacher files
def test_dump_and_average(tiny_tree: Path):
    files = train_member(_tiny_recipe(epochs=1), seed=1, name="tiny", out_dir=tiny_tree / "runs", device="cpu")
    dumps = tiny_tree / "dumps"
    dump_train_scores([files.checkpoint], dumps / "one.npy", device="cpu", log=lambda line: None)
    dump_train_scores([files.checkpoint, files.checkpoint], dumps / "two.npy", device="cpu", log=lambda line: None)
    one, n_one = read_dump(dumps / "one.npy")
    two, n_two = read_dump(dumps / "two.npy")
    assert (n_one, n_two) == (1, 2) and count_file(dumps / "one.npy").read_text() == "1"

    split = load_split("train", "full", device="cpu")
    split.set_domain_mix(read_exploration_probability(split.feed_sizes))
    direct = predict(load_ranker(files.checkpoint), split, tf32=False)
    assert np.array_equal(one, direct) and np.array_equal(two, direct + direct)

    sizes, offsets = split.feed_sizes, split.candidate_offsets
    teacher = average_dumps([dumps / "one.npy", dumps / "two.npy"], offsets, sizes)
    expected = (centre_per_feed(3 * direct.astype(np.float64), offsets, sizes) / 3).astype(np.float32)
    assert teacher.dtype == np.float32 and np.allclose(teacher, expected, atol=1e-5)
    assert np.abs(np.add.reduceat(teacher.astype(np.float64), offsets) / sizes).max() < 1e-5

    (tiny_tree / "teacher.toml").write_text(
        'name = "mean11"\n[[dumps]]\nfile = "one"\nmembers = ["a"]\n[[dumps]]\nfile = "two"\nmembers = ["a", "a"]\n')
    recipe = TeacherRecipe.read(tiny_tree / "teacher.toml")
    built = build_teacher(recipe, tiny_tree / "built.npy")
    assert np.array_equal(np.load(built), teacher) and recipe.n_members == 3
    with pytest.raises(FileExistsError):
        build_teacher(recipe)                                   # the [teachers] mean11 file exists
    with pytest.raises(FileExistsError):
        dump_train_scores([files.checkpoint], dumps / "one.npy", device="cpu")


@pytest.mark.artefacts
@pytest.mark.parametrize("name, tolerance", [("mean11", 4e-5), ("mean25", 1e-6)])
def test_stored_teachers_are_the_centred_mean_of_the_stored_dumps(need, name: str, tolerance: float):
    recipe = TeacherRecipe.read(RECIPE_DIR / f"teacher_{name}.toml")
    files = recipe.dump_files()
    need(path(f"teachers.{name}"), path("data.flat") / "train" / "feeds.parquet", *files,
         *(count_file(f) for f in files))
    feeds = read_feeds("train").head(30_000)
    sizes, offsets = feeds["M"].to_numpy().astype(np.int64), feeds["cand_off"].to_numpy().astype(np.int64)
    n_rows = int(offsets[-1] + sizes[-1])
    total, n_members = np.zeros(n_rows), 0
    for file in files:
        scores, count = read_dump(file)
        total += scores[:n_rows]
        n_members += count
    assert n_members == recipe.n_members
    rebuilt = (centre_per_feed(total, offsets, sizes) / n_members).astype(np.float32)
    stored = np.load(path(f"teachers.{name}"), mmap_mode="r")[:n_rows]
    assert np.abs(rebuilt.astype(np.float64) - stored).max() <= tolerance
