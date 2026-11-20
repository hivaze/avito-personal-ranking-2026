"""The distillation teacher: the averaged train scores of hard-label members.

A student learns from a mix of the clicks and the softmax of a teacher's
scores over each feed (``avito_ranker.training.loss.distillation_target``).
The teacher is one float32 score per train candidate, computed once before
the students train and stored:

1. Dump. Each hard-label member scores every train candidate
   (:func:`dump_train_scores`). The scoring runs as for the train split in
   training: every feed carries its exploration-probability mix of the domain
   rows, and float32 products run at full precision (no TF32). The scores of
   a group of members are summed into one file ``<group>.npy``; the number of
   members goes into ``<group>.n`` next to it.
2. Average (:func:`average_dumps`). The teacher is the sum of the group files
   divided by the total member count, minus its mean inside each feed. Only
   differences inside a feed reach the softmax, so the centring changes no
   target; it keeps the stored numbers small (raw member scores sit near
   +125).

The two teachers of the final students are recipes:
``configs/recipes/teacher_mean11.toml`` (11 members) and
``teacher_mean25.toml`` (25 members). Thirteen of the 25 can be averaged from
their stored group files but not dumped again: seven have no recipe and no
stored checkpoint, and six use a recurrent (GRU) block over the history that
this package does not contain.

The sum is accumulated in float64, which makes it independent of the order of
the files. The stored ``teacher_mean11.npy`` was written with a float32
running sum and differs from this function's result by at most 3.3e-5;
``teacher_mean25.npy`` by at most 4.8e-7.
"""
from __future__ import annotations

import time
import tomllib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from avito_ranker.data.flatten import read_feeds
from avito_ranker.data.loader import SplitTensors, load_split
from avito_ranker.io import refuse_existing
from avito_ranker.model import load_ranker
from avito_ranker.paths import path
from avito_ranker.training.predict import predict, save_scores
from avito_ranker.training.train import read_exploration_probability


def count_file(dump: Path) -> Path:
    """The text file that holds the number of members summed in ``dump``."""
    return dump.with_suffix(".n")


def dump_train_scores(checkpoints: Sequence[Path], target: Path, device: str | torch.device = "cuda:0",
                      log: Callable[[str], None] = print) -> None:
    """Score every train candidate with each checkpoint and store the sum of the scores and the member count.

    The train split is loaded once per input view, so list members of the
    same view together. One member takes about 105 s on one GPU.
    """
    refuse_existing(target, count_file(target))
    device = torch.device(device)
    feed_sizes = read_feeds("train")["M"].to_numpy().astype(np.int64)
    exploration = read_exploration_probability(feed_sizes)
    split: SplitTensors | None = None
    total = None
    for checkpoint in checkpoints:
        started = time.time()
        model = load_ranker(checkpoint, device)
        if split is None or split.view != model.config.view:
            split = None
            if device.type == "cuda":
                torch.cuda.empty_cache()
            split = load_split("train", model.config.view, device)
            split.set_domain_mix(exploration)
        scores = predict(model, split, tf32=False)
        total = scores if total is None else total + scores
        log(f"{checkpoint}: mean |score| {float(np.abs(scores).mean()):.3f}, sd {float(scores.std()):.3f}, "
            f"{time.time() - started:.0f} s")
        del model
    save_scores(target, total)
    count_file(target).write_text(str(len(checkpoints)))


def read_dump(dump: Path) -> tuple[np.ndarray, int]:
    """The summed scores of a dump (memory-mapped) and its member count."""
    return np.load(dump, mmap_mode="r"), int(count_file(dump).read_text())


def centre_per_feed(scores: np.ndarray, offsets: np.ndarray, sizes: np.ndarray) -> np.ndarray:
    """Subtract from each score the mean of its feed. Feeds are consecutive rows starting at ``offsets``."""
    means = np.add.reduceat(scores, offsets) / sizes
    return scores - np.repeat(means, sizes)


def average_dumps(dumps: Sequence[Path], offsets: np.ndarray, sizes: np.ndarray) -> np.ndarray:
    """The teacher: the per-feed-centred sum of the dumps divided by their total member count, float32."""
    total = np.zeros(int(sizes.sum()), dtype=np.float64)
    n_members = 0
    for dump in dumps:
        scores, count = read_dump(dump)
        if scores.shape != total.shape:
            raise ValueError(f"{dump}: {scores.shape[0]:,} scores for {total.shape[0]:,} train candidates")
        total += scores
        n_members += count
    return (centre_per_feed(total, offsets, sizes) / n_members).astype(np.float32)


@dataclass(frozen=True)
class TeacherRecipe:
    """Which dumps make a teacher, read from ``configs/recipes/teacher_mean*.toml``."""

    name: str                          # the [teachers] entry of configs/paths.toml it is written to
    dumps: tuple[str, ...]             # file names under [teachers] dumps, without .npy
    members: tuple[tuple[str, ...], ...]

    @classmethod
    def read(cls, file: str | Path) -> TeacherRecipe:
        """Read a teacher recipe: its ``name`` and one ``[[dumps]]`` entry per group file."""
        with Path(file).open("rb") as handle:
            table = tomllib.load(handle)
        return cls(table["name"], tuple(d["file"] for d in table["dumps"]),
                   tuple(tuple(d["members"]) for d in table["dumps"]))

    @property
    def n_members(self) -> int:
        """Number of hard-label members averaged into the teacher."""
        return sum(len(group) for group in self.members)

    def dump_files(self) -> list[Path]:
        """The group files of the recipe, in the folder registered as ``teachers.dumps``."""
        return [path("teachers.dumps") / f"{name}.npy" for name in self.dumps]


def build_teacher(recipe: TeacherRecipe, target: Path | None = None) -> Path:
    """Average the recipe's dumps on the train split and write the teacher file; return its path."""
    target = Path(target) if target else path(f"teachers.{recipe.name}")
    refuse_existing(target)
    files = recipe.dump_files()
    counts = [read_dump(file)[1] for file in files]
    for file, count, group in zip(files, counts, recipe.members):
        if count != len(group):
            raise ValueError(f"{file}: {count} members on disk, the recipe lists {len(group)}")
    feeds = read_feeds("train")
    sizes = feeds["M"].to_numpy().astype(np.int64)
    offsets = feeds["cand_off"].to_numpy().astype(np.int64)
    save_scores(target, average_dumps(files, offsets, sizes))
    return target

