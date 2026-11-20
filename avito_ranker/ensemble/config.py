"""An ensemble recipe: who the members are, how they are weighted, where the files go.

A recipe is one TOML file in ``configs/ensembles/``. The member names in it are
the names of the stored prediction files and of the training logs, so they are
the names the runs had when they were made. Everything else uses plain words:

``full_view``
    Students that read all 34 pre-computed inputs.
``popularity_light``
    Students trained without the eight popularity inputs. Most of them get one
    capped exposure count back (27 inputs, popularity-light); a recipe may
    also list students that get nothing back (26 inputs, popularity-free).
    Why they exist and why their pool gets a small weight: see
    ``configs/recipes/model_popularity_light.toml``.

The weight of the popularity-light pool is one constant (``popularity_light``
in ``[weights]``), or follows the feed-size rule when the recipe also sets
``popularity_light_short`` and ``short_feed_below``: a row of a feed with
fewer than ``short_feed_below`` candidates gets ``popularity_light_short``,
every other row ``popularity_light``. FD16P20@m0.4_0.2 uses 0.4 below 50
candidates and 0.2 from 50 up.

Every member has four prediction files, two branches times two splits:

``finetuned``
    The member after its fine-tune on the validation days. On val this is the
    out-of-fold prediction of the five user-grouped folds, on test the mean of
    the fold models.
``train_only``
    The member as it left training, before it saw any validation label.
"""
from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

from avito_ranker.paths import path

BRANCHES = ("finetuned", "train_only")
SCORED_SPLITS = ("val", "test")   # the splits a member predicts; train is never scored for the ensemble

_POOL_BRANCH_CODE = {"finetuned": "ft", "train_only": "to"}
_WEIGHT_KEYS = {"popularity_light", "popularity_light_short", "short_feed_below", "finetuned_branch"}


@dataclass(frozen=True)
class Record:
    """What the working records say about the stored submission of a recipe."""

    sha256: str         # of the parquet file
    val_ndcg: float     # validation NDCG of the final score
    public_ndcg: float  # public leaderboard score
    gate_output: str    # file name of the stored gate output (folder: records.gate_outputs)


@dataclass(frozen=True)
class EnsembleConfig:
    """A parsed recipe. See the module docstring for the vocabulary."""

    name: str
    full_view: tuple[str, ...]
    popularity_light: tuple[str, ...]
    popularity_light_weight: float  # each branch: (1 - w) z(full-view pool) + w z(popularity-light pool);
                                    # under the feed-size rule, w on feeds of short_feed_below candidates or more
    finetuned_weight: float         # final score: w z(fine-tuned branch) + (1 - w) z(train-only branch)
    finetune_tag: str               # tag of the fine-tune recipe in member file and log names
    pool_tag: str                   # tag of this ensemble's stored pool files
    submission: str                 # file name of the submission parquet
    reference_pool_tag: str         # the earlier submission the gate measures this one against
    record: Record
    popularity_light_weight_short: float | None = None  # feed-size rule: w on feeds of fewer candidates
    short_feed_below: int | None = None                 # feed-size rule: a feed with fewer candidates is short

    @property
    def members(self) -> tuple[str, ...]:
        """All members, full view first: the order the gate audits them in."""
        return self.full_view + self.popularity_light

    @property
    def has_feed_size_rule(self) -> bool:
        """Whether the popularity-light weight depends on the number of candidates of the feed."""
        return self.short_feed_below is not None

    def popularity_light_weight_text(self) -> str:
        """The popularity-light weight as the read-outs print it: ``0.2``, or the feed-size rule."""
        if not self.has_feed_size_rule:
            return f"{self.popularity_light_weight:g}"
        return (f"{self.popularity_light_weight_short:g} on feeds of fewer than {self.short_feed_below} candidates, "
                f"{self.popularity_light_weight:g} on the others")


def _feed_size_rule(weights: dict) -> tuple[float | None, int | None]:
    """``(popularity_light_short, short_feed_below)`` of a ``[weights]`` table, both None without the rule."""
    unknown = set(weights) - _WEIGHT_KEYS
    if unknown:
        raise ValueError(f"unknown keys in [weights]: {sorted(unknown)}; known: {sorted(_WEIGHT_KEYS)}")
    given = [key in weights for key in ("popularity_light_short", "short_feed_below")]
    if not any(given):
        return None, None
    if not all(given):
        raise ValueError("the feed-size rule needs both popularity_light_short and short_feed_below in [weights]")
    short, below = weights["popularity_light_short"], weights["short_feed_below"]
    if not isinstance(below, int) or isinstance(below, bool) or below < 2:
        raise ValueError(f"short_feed_below must be an integer of at least 2, not {below!r}")
    return float(short), below


def load_ensemble_config(config_path: str | Path) -> EnsembleConfig:
    """Read a recipe from ``configs/ensembles/<name>.toml``."""
    with Path(config_path).open("rb") as fh:
        raw = tomllib.load(fh)
    short_weight, short_feed_below = _feed_size_rule(raw["weights"])
    config = EnsembleConfig(
        name=raw["name"],
        full_view=tuple(raw["members"]["full_view"]),
        popularity_light=tuple(raw["members"]["popularity_light"]),
        popularity_light_weight=float(raw["weights"]["popularity_light"]),
        finetuned_weight=float(raw["weights"]["finetuned_branch"]),
        finetune_tag=raw["files"]["finetune_tag"],
        pool_tag=raw["files"]["pool_tag"],
        submission=raw["files"]["submission"],
        reference_pool_tag=raw["gate"]["reference_pool_tag"],
        record=Record(**raw["record"]),
        popularity_light_weight_short=short_weight,
        short_feed_below=short_feed_below,
    )
    for weight in (config.popularity_light_weight, config.popularity_light_weight_short, config.finetuned_weight):
        if weight is not None and not 0.0 <= weight <= 1.0:
            raise ValueError(f"{config_path}: weight {weight} outside [0, 1]")
    return config


def member_file(config: EnsembleConfig, member: str, branch: str, split: str, folder: Path | None = None) -> Path:
    """Location of one member's predictions for a branch and a split.

    ``folder`` defaults to the registered ``outputs.predictions``, where the
    stored member predictions are. Steps 4, 6 and 7 write new predictions to
    ``<outputs.runs>/predictions``.
    """
    if branch == "finetuned":
        suffix = f"{config.finetune_tag}_val_oof" if split == "val" else f"{config.finetune_tag}_test"
    elif branch == "train_only":
        suffix = f"trainonly_{split}"
    else:
        raise ValueError(f"unknown branch {branch!r}; choose from {BRANCHES}")
    folder = path("outputs.predictions") if folder is None else Path(folder)
    return folder / f"nn_{member}_{suffix}.npy"


def pool_file(pool_tag: str, branch: str, split: str, folder: Path | None = None) -> Path:
    """Location of a stored pool: the member-pooled scores of one branch on one split."""
    folder = path("outputs.predictions") if folder is None else Path(folder)
    return folder / f"ens_{pool_tag}_{_POOL_BRANCH_CODE[branch]}_{split}.npy"


def submission_file(config: EnsembleConfig, folder: Path | None = None) -> Path:
    """Location of the recipe's submission parquet."""
    folder = path("outputs.submissions") if folder is None else Path(folder)
    return folder / config.submission
