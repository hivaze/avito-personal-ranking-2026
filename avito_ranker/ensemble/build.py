"""Turn member predictions into the final score of an ensemble.

With ``z`` the within-feed z-score (``avito_ranker.ensemble.zscore``):

1. Pool. For each branch, ``pool = mean over members of z(member)``. Members
   count equally; no weight is fitted on validation.
2. Mix the views. For each branch,
   ``branch = (1 - w) z(full-view pool) + w z(popularity-light pool)`` with
   ``w = 0.2`` in FD16P20, FD16P20vf and W8P20. Under the feed-size rule of a
   recipe ``w`` is a float32 value per row: in FD16P20@m0.4_0.2, 0.4 on the
   rows of feeds with fewer than 50 candidates and 0.2 on the others.
3. Mix the branches. ``final = 0.5 z(fine-tuned branch) + 0.5 z(train-only branch)``.

The fine-tuned branch has seen validation labels (out of fold), the train-only
branch has not. The weights 0.2, 0.4 and 0.5 and the feed size 50 are
constants of the recipe files in ``configs/ensembles/``; nothing in this
module fits a weight.

All arithmetic is float32 and the members are summed in the order of the
recipe. Both are needed to reproduce the stored submissions byte for byte.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from avito_ranker.ensemble.config import (
    BRANCHES,
    SCORED_SPLITS,
    EnsembleConfig,
    member_file,
    pool_file,
    submission_file,
)
from avito_ranker.ensemble.feeds import FeedIndex, load_feed_index
from avito_ranker.ensemble.submission import checked_submission_frame
from avito_ranker.ensemble.zscore import feed_zscore
from avito_ranker.io import refuse_existing


@dataclass(frozen=True)
class Branches:
    """The two score columns of an ensemble on one split, in flat row order."""

    finetuned: np.ndarray
    train_only: np.ndarray


def missing_member_files(config: EnsembleConfig, members_dir: Path | None = None) -> list[Path]:
    """Prediction files the recipe needs and ``members_dir`` (default: ``outputs.predictions``) lacks."""
    wanted = (member_file(config, m, b, s, members_dir)
              for m in config.members for b in BRANCHES for s in SCORED_SPLITS)
    return [p for p in wanted if not p.exists()]


def pool_members(config: EnsembleConfig, members: tuple[str, ...], split: str, feeds: FeedIndex,
                 members_dir: Path | None = None) -> Branches:
    """Average the within-feed z-scores of ``members``, one average per branch."""
    def pooled(branch: str) -> np.ndarray:
        """Mean of the members' within-feed z-scores on one branch."""
        return np.mean([feed_zscore(feeds, np.load(member_file(config, m, branch, split, members_dir)))
                        for m in members], 0)

    return Branches(pooled("finetuned"), pooled("train_only"))


def popularity_light_weight(config: EnsembleConfig, feeds: FeedIndex) -> float | np.ndarray:
    """The weight of the popularity-light pool: the recipe's constant, or one value per row.

    Under the feed-size rule the result is float32, one value per row of the
    split: ``popularity_light_short`` on the rows of feeds with fewer than
    ``short_feed_below`` candidates, ``popularity_light`` on the others. The
    lab builder made the same float32 array, and ``mix_views`` then computes
    ``1 - w`` in float32, as it did.
    """
    if not config.has_feed_size_rule:
        return config.popularity_light_weight
    feed_size_of_row = np.repeat(feeds.sizes, feeds.sizes)
    return np.where(feed_size_of_row >= config.short_feed_below,
                    config.popularity_light_weight, config.popularity_light_weight_short).astype(np.float32)


def mix_views(full_view: Branches, popularity_light: Branches, weight: float | np.ndarray,
              feeds: FeedIndex) -> Branches:
    """Blend the two pools branch by branch: ``(1 - weight) z(full) + weight z(light)``.

    ``weight`` is a number, or a float32 array with one weight per row
    (``popularity_light_weight``).
    """
    def mixed(full: np.ndarray, light: np.ndarray) -> np.ndarray:
        return (1 - weight) * feed_zscore(feeds, full) + weight * feed_zscore(feeds, light)

    return Branches(
        mixed(full_view.finetuned, popularity_light.finetuned),
        mixed(full_view.train_only, popularity_light.train_only),
    )


def final_score(branches: Branches, feeds: FeedIndex, finetuned_weight: float) -> np.ndarray:
    """Blend the two branches into the score that is ranked for the submission."""
    return (finetuned_weight * feed_zscore(feeds, branches.finetuned)
            + (1 - finetuned_weight) * feed_zscore(feeds, branches.train_only))


def build_branches(config: EnsembleConfig, split: str, feeds: FeedIndex, members_dir: Path | None = None) -> Branches:
    """Steps 1 and 2 of the module docstring for one split, from the member files."""
    full_view = pool_members(config, config.full_view, split, feeds, members_dir)
    popularity_light = pool_members(config, config.popularity_light, split, feeds, members_dir)
    return mix_views(full_view, popularity_light, popularity_light_weight(config, feeds), feeds)


def build_ensemble(config: EnsembleConfig, members_dir: Path | None = None) -> dict[str, Branches]:
    """Build the branches of both splits from the member files in ``members_dir``.

    ``members_dir`` defaults to the registered ``outputs.predictions``. Raises
    if a member file is missing.
    """
    missing = missing_member_files(config, members_dir)
    if missing:
        shown = ", ".join(p.name for p in missing[:6])
        folder = missing[0].parent
        raise FileNotFoundError(f"{len(missing)} member prediction files are missing in {folder}, "
                                f"for example: {shown}")
    return {split: build_branches(config, split, load_feed_index(split), members_dir) for split in SCORED_SPLITS}


def load_pool(pool_tag: str, split: str, folder: Path | None = None) -> Branches:
    """Read the stored branches of an ensemble."""
    return Branches(*(np.load(pool_file(pool_tag, branch, split, folder)) for branch in BRANCHES))


def write_ensemble(config: EnsembleConfig, branches: dict[str, Branches], folder: Path | None = None) -> list[Path]:
    """Store the pools of both splits as float32 and write the submission file.

    Files go to ``folder`` when given, otherwise to the registered predictions
    and submissions folders. Everything is checked before the first file is
    written: none of the five files may exist (a stored pool or submission is
    never replaced), and the submission's keys must equal the sample
    submission's. The five files are then written under temporary names and
    renamed at the end, so a failure leaves none of them behind.
    """
    pools = {(branch, split): pool_file(config.pool_tag, branch, split, folder)
             for split in SCORED_SPLITS for branch in BRANCHES}
    submission = submission_file(config, folder)
    refuse_existing(*pools.values(), submission)
    test_score = final_score(branches["test"], load_feed_index("test"), config.finetuned_weight)
    frame = checked_submission_frame(test_score.astype(np.float32))

    partial = {target: target.with_name(f".{target.name}.partial") for target in (*pools.values(), submission)}
    try:
        for (branch, split), target in pools.items():
            with open(partial[target], "wb") as handle:
                np.save(handle, getattr(branches[split], branch).astype(np.float32))
        frame.write_parquet(partial[submission])
        refuse_existing(*partial)              # nothing appeared under the final names in the meantime
        for target, temporary in partial.items():
            os.rename(temporary, target)
    finally:
        for temporary in partial.values():
            temporary.unlink(missing_ok=True)
    return [*pools.values(), submission]
