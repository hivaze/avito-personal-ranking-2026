"""Training one ensemble member on the train split.

This is the training loop of every member of the final ensembles and of the
click-label (hard-label) members behind their teacher; the recipes in
``configs/recipes/student_*.toml`` and ``teacher_hard_label*.toml`` hold the
settings. One run of a distilled student takes about four hours on one
GPU (99,840 steps of at most 512 feeds).

A step:

1. Take the next batch of the shuffled epoch plan. In the cooldown, a step
   trains with probability 0.02 on a batch of random-order feeds instead
   (``avito_ranker.training.cooldown``).
2. Compute the dense block in float32, then run the model in bfloat16 mixed
   precision; it returns the scores and the scores of the production-order
   head.
3. Loss = listwise cross-entropy of the scores toward the clicks, or toward
   ``0.5 * clicks + 0.5 * softmax(teacher)`` for a distilled student, plus
   the production-order loss (``avito_ranker.training.loss``).
4. Clip the gradient norm to 1 and step AdamW and Muon at the scheduled
   learning rate (``avito_ranker.training.optim``).

Every train feed carries its probability ``p`` of having been shown in random
order (``exploration.probability`` in ``configs/paths.toml``; 0 for feeds
under 50 candidates, where the estimate is unreliable). The model's domain
embedding mixes the production row and the random-order row with weights
``1 - p`` and ``p``, and the production-order loss weighs the feed by
``1 - p``.

Two ways to use the train split:

* ``holdout.feeds = 0``: train on every feed and keep the last step. The
  students of the two all-data final files (``configs/ensembles/fd16p20.toml``
  and ``fd16p20vf.toml``) were trained so; validation is scored in the log
  and never used to choose anything.
* ``holdout.feeds = 30000``: keep 30,000 train feeds that look like
  random-order traffic (``p > 0.9``, at least 50 candidates, drawn with a
  fixed seed) out of training, score them every ``eval_every`` steps and keep
  the checkpoint with the best holdout NDCG from ``select_from_step`` on. The
  click-label teacher members and the students of the best public file
  (``configs/ensembles/w8p20.toml``) were trained so.

The length of a run is known from the train feed table alone
(:func:`plan_run`), so a ``select_from_step`` beyond the last step is refused
before the train split is loaded.

A seeded run draws the same random numbers as the run that trained the stored
member of that seed: model initialisation and dropout from
``torch.manual_seed(seed)``, the epoch plans from ``default_rng(seed)``, the
random-order batches of the cooldown from ``default_rng(seed + 1)``, the
holdout from ``default_rng(12345)`` and the dense-block statistics from
``default_rng(0)``. The GPU backward pass is not bit-reproducible, so two
runs of the same seed drift apart slowly: over the first 250 steps their
losses differ by at most 4e-5 relative.

Outputs, under ``--out-dir`` (default ``outputs.runs``): the checkpoint
``checkpoints/<name>.pt``, the log ``logs/<name>.log`` and the train-only
predictions ``predictions/nn_<name>_trainonly_{val,test}.npy`` that the
ensemble reads. The checkpoint and the predictions are never replaced; the
log is appended to. Each run in the log starts with the two lines the
pre-submission gate reads to audit a member's inputs: ``args: {'variant':
'<name>', ...}`` and ``train-stage extra features (N): [...]``; the gate
audits the last run of a name.
"""
from __future__ import annotations

import copy
import math
import time
import tomllib
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from avito_ranker.data.flatten import feed_offsets, read_feeds, rows_of_feeds
from avito_ranker.data.loader import SplitTensors, load_split, plan_batches
from avito_ranker.io import refuse_existing
from avito_ranker.metrics import ndcg
from avito_ranker.model import Ranker, RankerConfig, dense_statistics, load_recipe, save_checkpoint
from avito_ranker.model.batch import Batch
from avito_ranker.paths import path
from avito_ranker.training.cooldown import ExplorationBatches, exploration_pool, position_weight
from avito_ranker.training.loss import distillation_target, listwise_loss, production_order_loss
from avito_ranker.training.optim import Schedule, build_optimizers, set_learning_rate
from avito_ranker.training.predict import predict, save_scores, tf32_matmul

MIN_FEED_SIZE_FOR_EXPLORATION = 50  # shorter feeds get p = 0
HOLDOUT_MIN_PROBABILITY = 0.9       # holdout feeds look like random-order traffic ...
HOLDOUT_MIN_FEED_SIZE = 50          # ... and have at least this many candidates
HOLDOUT_SEED = 12345
LOSS_SAMPLE_EVERY = 50              # the logged loss is the mean over every 50th step (avoids a sync per step)
GRADIENT_CLIP = 1.0


@dataclass(frozen=True)
class TrainingRecipe:
    """The settings of a member's training run, read from ``configs/recipes/<name>.toml``."""

    name: str
    model_recipe: str              # name of the model recipe, e.g. "full"
    model: RankerConfig
    teacher: str | None            # key under [teachers] in configs/paths.toml; None = clicks alone
    teacher_weight: float
    position_weight: float         # production-order loss weight before the cooldown ...
    position_weight_end: float     # ... and at the end of the cooldown
    position_temperature: float
    epochs: float
    learning_rate: float
    weight_decay: float
    warmup_steps: int
    cooldown_fraction: float
    exploration_probability: float  # chance that a cooldown step trains on random-order feeds
    max_candidates: int
    max_feeds: int
    holdout_feeds: int             # 0 = train on every feed and keep the last step
    select_from_step: int
    eval_every: int

    @classmethod
    def read(cls, file: str | Path) -> TrainingRecipe:
        """Read a training recipe; its ``model`` entry names a model recipe, ``width`` may override its width."""
        file = Path(file)
        with file.open("rb") as handle:
            table = tomllib.load(handle)
        model = load_recipe(table["model"])
        if "width" in table:
            model = replace(model, d_model=table["width"])
        targets, schedule, batches, holdout = table["targets"], table["schedule"], table["batches"], table["holdout"]
        recipe = cls(
            name=file.stem,
            model_recipe=table["model"],
            model=model,
            teacher=targets.get("teacher"),
            teacher_weight=targets["teacher_weight"],
            position_weight=targets["position_weight"],
            position_weight_end=targets["position_weight_end"],
            position_temperature=targets["position_temperature"],
            epochs=schedule["epochs"],
            learning_rate=schedule["learning_rate"],
            weight_decay=schedule["weight_decay"],
            warmup_steps=schedule["warmup_steps"],
            cooldown_fraction=schedule["cooldown_fraction"],
            exploration_probability=schedule["exploration_probability"],
            max_candidates=batches["max_candidates"],
            max_feeds=batches["max_feeds"],
            holdout_feeds=holdout["feeds"],
            select_from_step=holdout.get("select_from_step", 0),
            eval_every=holdout["eval_every"],
        )
        if (recipe.teacher is None) != (recipe.teacher_weight == 0):
            raise ValueError(f"{file}: a teacher needs a positive teacher_weight, "
                             "and a positive weight needs a teacher")
        return recipe

    def settings(self) -> dict:
        """The recipe as plain values, for the log and the checkpoint."""
        values = {key: value for key, value in asdict(self).items() if key not in ("name", "model")}
        return {"recipe": self.name, "width": self.model.d_model, **values}


# ----------------------------------------------------------------------------- inputs of a run
def read_exploration_probability(feed_sizes: np.ndarray) -> np.ndarray:
    """Probability that each train feed was shown in random order, float32; 0 where unknown or under 50 candidates.

    Feeds used to fit the position proxy have no estimate (NaN) and count as
    production feeds.
    """
    stored = np.load(path("exploration.probability"))
    if not np.array_equal(stored["fidx"], np.arange(len(feed_sizes))):
        raise ValueError("the exploration probabilities are not in train feed order")
    probability = np.nan_to_num(stored["p_random"].astype(np.float32), nan=0.0)
    probability[feed_sizes < MIN_FEED_SIZE_FOR_EXPLORATION] = 0.0
    return probability


def draw_holdout(exploration_probability: np.ndarray, feed_sizes: np.ndarray, n_feeds: int) -> np.ndarray:
    """The sorted holdout: ``n_feeds`` train feeds that look like random-order traffic, drawn with a fixed seed."""
    eligible = np.where((exploration_probability > HOLDOUT_MIN_PROBABILITY)
                        & (feed_sizes >= HOLDOUT_MIN_FEED_SIZE))[0]
    chosen = np.random.default_rng(HOLDOUT_SEED).choice(eligible, min(n_feeds, len(eligible)), replace=False)
    return np.sort(chosen)


def load_teacher(name: str, n_candidates: int, device: torch.device) -> Tensor:
    """The teacher's score of every train candidate, float32 on ``device``."""
    scores = np.load(path(f"teachers.{name}"))
    if scores.shape != (n_candidates,):
        raise ValueError(f"teacher {name}: {scores.shape[0]:,} scores for {n_candidates:,} train candidates")
    return torch.from_numpy(scores.astype(np.float32)).to(device)


def input_names(config: RankerConfig) -> list[str]:
    """The model's inputs as the gate reads them: presence flags as ``has:<file>``."""
    return [f"has:{c.file}" if c.scaling == "presence" else c.file for c in config.view.columns]


class RunLog:
    """Prints a line and appends it to the run's log file."""

    def __init__(self, file: Path):
        file.parent.mkdir(parents=True, exist_ok=True)
        self.file = file

    def __call__(self, line: str) -> None:
        print(line, flush=True)
        with self.file.open("a") as handle:
            handle.write(line + "\n")


@dataclass(frozen=True)
class RunFiles:
    """Where a run writes its outputs."""

    checkpoint: Path
    log: Path
    predictions: dict[str, Path]

    @classmethod
    def under(cls, out_dir: Path, name: str) -> RunFiles:
        """The files of the run ``name`` under ``out_dir``: checkpoints/, logs/ and predictions/."""
        return cls(checkpoint=out_dir / "checkpoints" / f"{name}.pt",
                   log=out_dir / "logs" / f"{name}.log",
                   predictions={split: out_dir / "predictions" / f"nn_{name}_trainonly_{split}.npy"
                                for split in ("val", "test")})

    def refuse_existing(self) -> None:
        """Raise if the checkpoint or a prediction file exists already; the log may exist (it is appended to)."""
        refuse_existing(self.checkpoint, *self.predictions.values())


# ----------------------------------------------------------------------------- one step
def training_loss(model: Ranker, batch: Batch, recipe: TrainingRecipe, teacher: Tensor | None,
                  exploration: Tensor, position_loss_weight: float) -> Tensor:
    """The loss of one batch: the main listwise loss plus the weighted production-order loss."""
    dense = model.dense_features(batch)
    with torch.autocast(batch.device.type, dtype=torch.bfloat16):
        scores, position_scores = model(batch, dense, with_position_scores=True)
    mask = batch.candidate_mask
    target = batch.target
    if teacher is not None:
        target = distillation_target(target, teacher[batch.candidate_row], mask, recipe.teacher_weight)
    loss = listwise_loss(scores, target, mask, recipe.max_feeds)
    feed_weight = 1 - exploration[batch.feed_index]
    return loss + production_order_loss(position_scores, batch.position, mask, feed_weight,
                                        recipe.position_temperature, position_loss_weight)


def holdout_ndcg(model: Ranker, train: SplitTensors, holdout: np.ndarray, holdout_rows: np.ndarray) -> float:
    """NDCG of the model on the held-out train feeds (``holdout_rows`` are their candidate rows)."""
    scores = predict(model, train, holdout)[holdout_rows]
    sizes = train.feed_sizes[holdout]
    return ndcg(scores, train.targets[holdout_rows], feed_offsets(sizes), sizes)


def validation_ndcg(model: Ranker, val: SplitTensors) -> tuple[np.ndarray, float]:
    """The model's scores on val and their NDCG."""
    scores = predict(model, val)
    return scores, ndcg(scores, val.targets, val.candidate_offsets, val.feed_sizes)


# ----------------------------------------------------------------------------- the run
@dataclass(frozen=True)
class RunPlan:
    """What a run does with the train split, known from the feed table before the split is loaded."""

    exploration: np.ndarray        # float32 per train feed, see read_exploration_probability
    feeds: np.ndarray              # the training feeds
    holdout: np.ndarray | None     # the held-out feeds, sorted; None without a holdout
    steps_per_epoch: int
    plan: list[np.ndarray]         # the batches of the run in order, ``total`` of them

    @property
    def total(self) -> int:
        """Number of optimiser steps."""
        return len(self.plan)


def plan_run(recipe: TrainingRecipe, seed: int, feed_sizes: np.ndarray | None = None) -> RunPlan:
    """The holdout, the training feeds and the epoch plans of a run, from the train feed table.

    Raises if a holdout run's ``select_from_step`` lies outside the run, so a
    wrong trial setting fails before the train split is loaded.
    """
    if feed_sizes is None:
        feed_sizes = read_feeds("train")["M"].to_numpy().astype(np.int64)
    exploration = read_exploration_probability(feed_sizes)
    feeds = np.arange(len(feed_sizes))
    holdout = None
    if recipe.holdout_feeds:
        holdout = draw_holdout(exploration, feed_sizes, recipe.holdout_feeds)
        feeds = np.setdiff1d(feeds, holdout)
    rng = np.random.default_rng(seed)
    epoch_plans = [plan_batches(feed_sizes, feeds, recipe.max_candidates, recipe.max_feeds, shuffle=True, rng=rng)
                   for _ in range(math.ceil(recipe.epochs))]
    total = int(len(epoch_plans[0]) * recipe.epochs)
    plan = [batch for epoch in epoch_plans for batch in epoch][:total]
    if holdout is not None and not 0 < recipe.select_from_step <= total:
        raise ValueError(f"select_from_step {recipe.select_from_step} must lie in 1..{total}, the steps of this run "
                         f"({len(epoch_plans[0])} per epoch, {recipe.epochs} epochs)")
    return RunPlan(exploration, feeds, holdout, len(epoch_plans[0]), plan)


def run_length(recipe: TrainingRecipe) -> int:
    """Number of optimiser steps of a run of ``recipe``, from the train feed table alone (no shuffling)."""
    feed_sizes = read_feeds("train")["M"].to_numpy().astype(np.int64)
    feeds = np.arange(len(feed_sizes))
    if recipe.holdout_feeds:
        holdout = draw_holdout(read_exploration_probability(feed_sizes), feed_sizes, recipe.holdout_feeds)
        feeds = np.setdiff1d(feeds, holdout)
    return int(len(plan_batches(feed_sizes, feeds, recipe.max_candidates, recipe.max_feeds)) * recipe.epochs)


@dataclass
class Run:
    """Everything a run needs once its splits are on the device."""

    model: Ranker
    train: SplitTensors
    val: SplitTensors
    plan: RunPlan
    holdout_rows: np.ndarray | None  # candidate rows of the held-out feeds
    exploration: Tensor              # per train feed, on the device
    optimizers: list
    schedule: Schedule
    teacher: Tensor | None
    explorer: ExplorationBatches


def _prepare_run(recipe: TrainingRecipe, seed: int, device: torch.device, log: RunLog) -> Run:
    """Plan the run, load the splits, build the model and its optimisers.

    The order of the random draws is the order of the runs that trained the
    stored members.
    """
    torch.manual_seed(seed)
    config = recipe.model
    run_plan = plan_run(recipe, seed)             # numpy only; raises before anything is loaded
    train = load_split("train", config.view, device)
    val = load_split("val", config.view, device)
    log(f"[data] {train.summary()}")
    log(f"[data] {val.summary()}")
    log(f"train-stage extra features ({config.view.n_inputs}): {input_names(config)}")

    holdout, feeds, exploration_np = run_plan.holdout, run_plan.feeds, run_plan.exploration
    holdout_rows = None
    if holdout is not None:
        holdout_rows = rows_of_feeds(train.candidate_offsets[holdout], train.feed_sizes[holdout])
        log(f"holdout: {len(holdout)} exploration-like train feeds kept out of training")
    log(f"exploration probability: mean {exploration_np.mean():.4f}, feeds above 0.5 "
        f"{(exploration_np > 0.5).sum()}, training feeds {len(feeds)}")

    model = Ranker(config).to(device)
    model.set_dense_scaling(*dense_statistics(train, feeds, config.view))
    train.set_domain_mix(exploration_np)
    exploration = torch.from_numpy(exploration_np).to(device)
    optimizers = build_optimizers(model, recipe.learning_rate, recipe.weight_decay)
    n_parameters = sum(p.numel() for p in model.parameters())
    to_muon = optimizers[1].param_groups[0]["params"]
    log(f"params: {n_parameters / 1e6:.2f}M ({n_parameters:,})")
    log(f"muon: {len(to_muon)} matrices, {sum(p.numel() for p in to_muon) / 1e6:.3f}M params")

    total = run_plan.total
    schedule = Schedule.from_fraction(total, recipe.learning_rate, recipe.warmup_steps, recipe.cooldown_fraction)
    log(f"train feeds {len(feeds)}, steps/epoch {run_plan.steps_per_epoch}, total steps {total}")

    teacher = None
    if recipe.teacher:
        teacher = load_teacher(recipe.teacher, train.n_candidates, device)
        log(f"teacher {recipe.teacher} ({path(f'teachers.{recipe.teacher}').name}), weight {recipe.teacher_weight}")
    pool = exploration_pool(feeds, exploration_np)
    explorer = ExplorationBatches(train, pool, recipe.exploration_probability, seed + 1,
                                  recipe.max_candidates, recipe.max_feeds)
    log(f"cooldown: from step {schedule.cooldown_start}, exploration batches with probability "
        f"{recipe.exploration_probability}, exploration pool {len(pool)} feeds")
    return Run(model, train, val, run_plan, holdout_rows, exploration, optimizers, schedule, teacher, explorer)


def _run_steps(run: Run, recipe: TrainingRecipe, log: RunLog) -> tuple[Ranker, np.ndarray, dict]:
    """The step loop, the evaluations and the choice of the kept weights.

    Returns the model with the kept weights loaded, its validation scores and
    a summary of the run.
    """
    model, train, val, schedule, explorer = run.model, run.train, run.val, run.schedule, run.explorer
    holdout, total = run.plan.holdout, run.plan.total
    best_score, best_state, kept_step = -math.inf, None, total
    loss_sum, loss_n = 0.0, 0
    started = time.time()
    model.train()
    for step, batch_feeds in enumerate(run.plan.plan):
        set_learning_rate(run.optimizers, schedule.learning_rate(step))
        if schedule.in_cooldown(step) and explorer.replaces():
            batch_feeds = explorer.next()
        batch = train.gather(batch_feeds, with_positions=True)
        weight = position_weight(schedule, step, recipe.position_weight, recipe.position_weight_end)
        loss = training_loss(model, batch, recipe, run.teacher, run.exploration, weight)
        for optimizer in run.optimizers:
            optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
        for optimizer in run.optimizers:
            optimizer.step()
        if step % LOSS_SAMPLE_EVERY == 0:
            loss_sum, loss_n = loss_sum + loss.item(), loss_n + 1

        if (step + 1) % recipe.eval_every == 0 or step == total - 1:
            _, val_score = validation_ndcg(model, val)
            held = holdout_ndcg(model, train, holdout, run.holdout_rows) if holdout is not None else float("nan")
            mean_loss = loss_sum / loss_n if loss_n else float("nan")
            log(f"step {step + 1}/{total} loss {mean_loss:.4f} val_ndcg {val_score:.5f} "
                f"holdout_ndcg {held:.5f} lr {schedule.learning_rate(step):.2e} "
                f"{(time.time() - started) / 60:.1f} min, exploration batches so far {explorer.used}")
            loss_sum, loss_n = 0.0, 0
            if holdout is not None and step + 1 >= recipe.select_from_step and held > best_score:
                best_score, best_state, kept_step = held, copy.deepcopy(model.state_dict()), step + 1

    train_minutes = round((time.time() - started) / 60, 1)
    if best_state is not None:
        model.load_state_dict(best_state)
    val_scores, val_score = validation_ndcg(model, val)
    summary = dict(kept_step=kept_step, total_steps=total, train_minutes=train_minutes, val_ndcg=val_score,
                   holdout_ndcg=None if holdout is None else best_score)
    return model, val_scores, summary


def _train(recipe: TrainingRecipe, seed: int, device: torch.device, log: RunLog) -> tuple[Ranker, np.ndarray, dict]:
    """Build and train the model.

    Returns the model with the kept weights loaded, its validation scores and
    a summary of the run. The train split is freed when this returns.
    """
    run = _prepare_run(recipe, seed, device, log)
    return _run_steps(run, recipe, log)


def train_member(recipe: TrainingRecipe, seed: int, name: str, out_dir: Path | None = None,
                 device: str | torch.device = "cuda:0") -> RunFiles:
    """Train one member, write its checkpoint, log and train-only val and test predictions; return their paths."""
    files = RunFiles.under(Path(out_dir) if out_dir else path("outputs.runs"), name)
    files.refuse_existing()
    device = torch.device(device)
    log = RunLog(files.log)
    log(f"==== {time.ctime()} args: {dict(variant=name, seed=seed, **recipe.settings())}")

    with tf32_matmul(True):  # as in the members' runs: TF32 for float32 products
        model, val_scores, summary = _train(recipe, seed, device, log)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        files.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        save_checkpoint(files.checkpoint, model, name=name, seed=seed, recipe=recipe.settings(), **summary)
        save_scores(files.predictions["val"], val_scores)
        log(f"kept step {summary['kept_step']} val_ndcg {summary['val_ndcg']:.5f}; "
            f"train time {summary['train_minutes']} min")
        test = load_split("test", recipe.model.view, device)
        save_scores(files.predictions["test"], predict(model, test))
        log(f"saved {files.checkpoint.name} and the val and test predictions")
    return files
