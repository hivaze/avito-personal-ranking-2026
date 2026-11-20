"""Fine-tune of a trained member on the validation days, out of fold.

Why. A member learns on the train days, whose feeds the production system put
in its own order. On the validation days, as on the test days that follow
them, the candidates were shown in random order, so a short fine-tune on the
validation feeds moves a member towards the traffic it will be scored on. It also lets the member read
42 extra inputs that exist only for val and test
(:mod:`avito_ranker.features.finetune_extras`). The validation labels are the
ones we score with, so the fine-tune runs out of fold.

How. The validation feeds are split into five folds that hold whole users
(``GroupKFold``). Every feed of a not-logged-in session carries user id 0,
which pools unrelated people, so each such feed is a group of its own. For each
fold and each of three seeds, a copy of the member

* gets the fine-tune branch, an MLP of the 42 extras added to the candidate
  token; its last layer starts at zero, so the copy starts out scoring exactly
  like the member;
* gets the validation-days row of the domain embedding, a copy of its
  random-order row;
* trains for 4 epochs on the feeds of the other four folds: listwise
  cross-entropy on the clicks, AdamW with learning rate 5e-5 (5e-4 for the
  branch) and weight decay 0.02 on every parameter, a linear warm-up over 30
  steps times a cosine that falls from the first step to 0 at the end of the
  last epoch, gradient norm clipped at 1, batches of at most 128 feeds and
  8,192 padded candidates;
* keeps an exponential moving average of its weights (decay 0.99) from the
  middle of the first epoch on. The average after the last step is the fold
  model.

The result is one score per candidate row of each split. On val it is the
out-of-fold score: the mean of the three fold models of the row's own fold. On
test it is the mean of all fifteen fold models. The numbers above live in
``configs/recipes/finetune.toml``.

Leak discipline.

* A fold model learns from the feeds of four folds and scores only the feeds
  of the fifth fold and the test feeds. A logged-in user's feeds all sit in one
  fold, so no feed of that user is scored by a model that learned from the
  user's labels. Feeds of not-logged-in sessions cannot be tied to a person and
  are spread one by one. :func:`check_folds` checks the folds before any
  training.
* Nothing is chosen on a label it is then scored with. Epochs, schedule and
  averaging are fixed in the recipe, and the fold model is always the final
  average: no best epoch, no best seed. The held-out NDCG in the log is
  computed after the scores are final and nothing reads it.
* The scaling of the extras (log for count-like columns, mean and standard
  deviation) is fitted on all val rows. It reads input values, no labels, and
  test reuses it.
* Test rows are only scored.

Numerics, as in the runs that wrote the stored predictions: TF32 for float32
matrix products during training as well as scoring; the dense block computed
in float32 before the bfloat16 autocast region; ``torch.manual_seed(1000 k +
s)`` and ``numpy.random.default_rng(1000 k + s)`` for fold k and seed s, which
fix the weights of the new branch, the dropout masks and the batch order. The
backward pass on the GPU is not bit-reproducible from run to run, so a re-run
matches the stored predictions closely, not bit for bit (docs/10-reproduce.md
has the comparison).
"""
from __future__ import annotations

import math
import time
import tomllib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import GroupKFold
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from avito_ranker.data.flatten import feed_offsets, rows_of_feeds
from avito_ranker.data.loader import SplitTensors, load_split
from avito_ranker.features.names import FINETUNE_EXTRAS
from avito_ranker.metrics import ndcg
from avito_ranker.model import Checkpoint, Ranker, build_ranker
from avito_ranker.model.inputs import InputView
from avito_ranker.paths import recipe_dir
from avito_ranker.training.loss import mean_listwise_loss
from avito_ranker.training.predict import predict, tf32_matmul

SEED_STRIDE = 1000  # fold k, seed s runs under seed 1000 k + s, as the stored runs did


@dataclass(frozen=True)
class FinetuneRecipe:
    """Hyper-parameters of the fine-tune. ``configs/recipes/finetune.toml`` explains each one."""

    file_tag: str
    n_folds: int
    seeds: int
    learning_rate: float
    extras_learning_rate_multiplier: float
    weight_decay: float
    gradient_clip: float
    epochs: int
    warmup_steps: int
    ema_decay: float
    ema_start_epoch: float
    max_feeds: int
    max_candidates: int


def default_recipe_file() -> Path:
    """The production fine-tune recipe, ``finetune.toml`` in the registered recipe folder."""
    return recipe_dir() / "finetune.toml"


def load_finetune_recipe(file: str | Path | None = None) -> FinetuneRecipe:
    """Read a fine-tune recipe (by default the production one)."""
    with Path(file or default_recipe_file()).open("rb") as handle:
        return FinetuneRecipe(**tomllib.load(handle))


@dataclass
class FoldReport:
    """What the log says about one fold. Reported only; no step reads it."""

    fold: int
    n_training_feeds: int
    n_held_out_feeds: int
    held_out_ndcg_before: float          # the member before the fine-tune
    held_out_ndcg_per_seed: list[float]  # each seed's fold model alone
    held_out_ndcg_after: float           # the mean of the seeds' scores
    minutes: float


@dataclass
class FinetuneResult:
    """Scores of the fine-tuned member, one float32 per candidate row of each split."""

    val_scores: np.ndarray   # out of fold: each row scored by the models of its own fold only
    test_scores: np.ndarray  # mean of all fold models
    val_ndcg_before: float
    val_ndcg_after: float
    folds: list[FoldReport]


def load_finetune_splits(view: InputView, device: str | torch.device = "cuda",
                         extras: Sequence[str] = FINETUNE_EXTRAS) -> tuple[SplitTensors, SplitTensors]:
    """Load val and test with the member's inputs and the fine-tune extras.

    The scaling of the extras is fitted on val and reused for test, so that a
    value means the same in both splits.
    """
    val = load_split("val", view, device)
    scaling = val.attach_finetune_extras(extras)
    test = load_split("test", view, device)
    test.attach_finetune_extras(extras, scaling)
    return val, test


def user_folds(user_ids: np.ndarray, n_folds: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split the feeds into folds of whole users; return (training feeds, held-out feeds) per fold.

    ``GroupKFold`` places the groups from the largest down, each into the fold
    with the fewest feeds so far. User id 0 marks not-logged-in sessions; each
    of those feeds becomes its own group (ids -1, -2, ...), so they spread over
    the folds instead of forming one large pseudo-user.
    """
    groups = np.asarray(user_ids).astype(np.int64)
    groups = np.where(groups == 0, -1 - np.arange(len(groups)), groups)
    feeds = np.arange(len(groups))
    return list(GroupKFold(n_splits=n_folds).split(feeds, groups=groups))


def check_folds(user_ids: np.ndarray, folds: Sequence[tuple[np.ndarray, np.ndarray]]) -> None:
    """Raise unless every feed is held out exactly once and no logged-in user is on both sides of a fold."""
    user_ids = np.asarray(user_ids)
    n_feeds = len(user_ids)
    held_out = np.sort(np.concatenate([fold_held_out for _, fold_held_out in folds]))
    if not np.array_equal(held_out, np.arange(n_feeds)):
        raise ValueError("the folds do not hold out every feed exactly once")
    for k, (training, fold_held_out) in enumerate(folds):
        if len(training) + len(fold_held_out) != n_feeds or np.intersect1d(training, fold_held_out).size:
            raise ValueError(f"fold {k}: training and held-out feeds overlap or miss feeds")
        shared = np.intersect1d(user_ids[training], user_ids[fold_held_out])
        shared = shared[shared != 0]
        if shared.size:
            raise ValueError(f"fold {k}: {shared.size} logged-in users have feeds on both sides")


def learning_rate(step: int, total_steps: int, recipe: FinetuneRecipe) -> float:
    """Learning rate of the member's own weights at optimiser step ``step`` (counted from 0).

    A linear warm-up over the first ``warmup_steps`` steps, times a cosine that
    falls from 1 at the first step to 0 at ``total_steps``.
    """
    warmup = min(1.0, (step + 1) / recipe.warmup_steps)
    cosine = 0.5 * (1 + math.cos(math.pi * min(step, total_steps) / total_steps))
    return recipe.learning_rate * warmup * cosine


def _optimizer(model: Ranker, recipe: FinetuneRecipe) -> torch.optim.AdamW:
    """AdamW over all weights, weight decay on every one; the fine-tune branch gets a larger learning rate."""
    named = list(model.named_parameters())
    groups = [
        {"params": [p for name, p in named if not name.startswith("finetune_mlp.")], "lr_multiplier": 1.0},
        {"params": [p for name, p in named if name.startswith("finetune_mlp.")],
         "lr_multiplier": recipe.extras_learning_rate_multiplier},
    ]
    return torch.optim.AdamW(groups, lr=recipe.learning_rate, weight_decay=recipe.weight_decay)


def _epoch_batches(val: SplitTensors, feeds: np.ndarray, recipe: FinetuneRecipe,
                   rng: np.random.Generator) -> list[np.ndarray]:
    """The batches of one epoch, in random order.

    ``plan_batches`` shuffles already. The stored runs then permuted the list
    of batches once more with the same generator; that draw is part of their
    seeded sequence, so it is kept.
    """
    batches = val.plan_batches(feeds, recipe.max_candidates, recipe.max_feeds, shuffle=True, rng=rng)
    return [batches[i] for i in rng.permutation(len(batches))]


def train_fold_model(checkpoint: Checkpoint, val: SplitTensors, training_feeds: np.ndarray, seed: int,
                     recipe: FinetuneRecipe) -> Ranker:
    """Fine-tune one copy of the member on ``training_feeds`` of val and return the average of its weights.

    ``val`` must carry the fine-tune extras. The model learns only from the
    labels of ``training_feeds``.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = build_ranker(checkpoint, n_finetune_extras=val.finetune_extras.shape[1]).to(val.device)
    model.use_validation_domain()
    model.train()
    optimizer = _optimizer(model, recipe)

    steps_per_epoch = len(val.plan_batches(training_feeds, recipe.max_candidates, recipe.max_feeds))
    total_steps = steps_per_epoch * recipe.epochs
    average_from = int(recipe.ema_start_epoch * steps_per_epoch)
    average = None
    step = 0
    for _ in range(recipe.epochs):
        for feeds in _epoch_batches(val, training_feeds, recipe, rng):
            rate = learning_rate(step, total_steps, recipe)
            for group in optimizer.param_groups:
                group["lr"] = rate * group["lr_multiplier"]
            batch = val.gather(feeds)
            dense = model.dense_features(batch)  # float32, outside mixed precision
            with torch.autocast(val.device.type, dtype=torch.bfloat16):
                scores = model(batch, dense)
            loss = mean_listwise_loss(scores, batch.target, batch.candidate_mask)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), recipe.gradient_clip)
            optimizer.step()
            if step >= average_from:
                if average is None:  # the first update copies the current weights
                    average = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(recipe.ema_decay))
                average.update_parameters(model)
            step += 1
    if average is None:
        raise ValueError("the weight average never started: ema_start_epoch is past the last step")
    return average.module


def feed_ndcg(scores: np.ndarray, split: SplitTensors, feed_ids: np.ndarray) -> float:
    """NDCG over the given feeds of a split with labels."""
    feed_ids = np.sort(feed_ids)
    sizes = split.feed_sizes[feed_ids]
    rows = rows_of_feeds(split.candidate_offsets[feed_ids], sizes)
    return ndcg(scores[rows], split.targets[rows], feed_offsets(sizes), sizes)


def finetune(checkpoint: Checkpoint, val: SplitTensors, test: SplitTensors, recipe: FinetuneRecipe,
             log: Callable[[str], None] = print) -> FinetuneResult:
    """Fine-tune a member on val in folds of whole users; score val out of fold and test with every fold model.

    ``val`` and ``test`` come from :func:`load_finetune_splits`. The module
    docstring describes the method and its leak discipline.
    """
    if checkpoint.config.n_finetune_extras:
        raise ValueError("the checkpoint has a fine-tune branch already; start from a member as it left training")
    if val.finetune_extras is None or test.finetune_extras is None:
        raise ValueError("val and test need the fine-tune extras (load_finetune_splits)")
    folds = user_folds(val.user_ids, recipe.n_folds)
    check_folds(val.user_ids, folds)
    log(f"feeds of not-logged-in sessions, one group each: {int((val.user_ids == 0).sum())}")

    with tf32_matmul(True):
        scores_before = predict(build_ranker(checkpoint).to(val.device), val)
        val_scores = np.zeros(val.n_candidates, np.float32)
        test_scores = np.zeros(test.n_candidates, np.float32)
        reports = []
        started = time.time()
        for fold, (training, held_out) in enumerate(folds):
            per_seed = []
            for s in range(recipe.seeds):
                model = train_fold_model(checkpoint, val, training, SEED_STRIDE * fold + s, recipe)
                # The fold model scores only feeds it never learned from: its own held-out fold, and test.
                # predict leaves the rows of every other val feed at 0, so each val row sums the scores of
                # its own fold's models and of no other.
                held_out_scores = predict(model, val, held_out)
                val_scores += held_out_scores / recipe.seeds
                test_scores += predict(model, test) / (recipe.n_folds * recipe.seeds)
                per_seed.append(feed_ndcg(held_out_scores, val, held_out))
                del model
            report = FoldReport(fold, len(training), len(held_out), feed_ndcg(scores_before, val, held_out),
                                per_seed, feed_ndcg(val_scores, val, held_out), (time.time() - started) / 60)
            reports.append(report)
            log(f"fold {fold}: {report.n_training_feeds:,} feeds to learn from, {report.n_held_out_feeds:,} held out; "
                f"held-out NDCG {report.held_out_ndcg_before:.4f} before, {report.held_out_ndcg_after:.4f} after "
                f"(per seed {', '.join(f'{x:.4f}' for x in per_seed)}); {report.minutes:.1f} min")

    result = FinetuneResult(
        val_scores, test_scores,
        val_ndcg_before=ndcg(scores_before, val.targets, val.candidate_offsets, val.feed_sizes),
        val_ndcg_after=ndcg(val_scores, val.targets, val.candidate_offsets, val.feed_sizes),
        folds=reports,
    )
    log(f"val NDCG out of fold: {result.val_ndcg_after:.5f} after the fine-tune, {result.val_ndcg_before:.5f} before")
    return result
