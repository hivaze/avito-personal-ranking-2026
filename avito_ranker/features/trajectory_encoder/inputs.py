"""From raw encoder scores to the encoder inputs of the ranker.

Three things happen between a model's raw scores and an input column.

1. Scale alignment. A val row is scored by a ``no_test`` model, a test row by
   an ``all`` model, a train row by a ``window`` model. These are different
   networks with different score scales. The ``all`` model of the same
   variant is the reference: each raw score ``x`` of another model is mapped
   to ``a x + b`` so that its mean and standard deviation match the
   reference's on the same feeds (all val feeds; a 60,000-feed sample of a
   train block). No labels are involved. The -2 sentinel ("no history", "no
   ID") is left alone. The ``all`` models on val and on train samples are
   used only for this, never as inputs: on those rows they have seen later
   snapshots. The ranker's six columns enter as within-feed rank, z-score or
   presence, which an affine map leaves unchanged; the alignment matters for
   the fine-tune extras, which enter raw, and keeps ``fe_s`` and ``new_s``
   inside the range that ``clip_safe`` checks.
2. Within-feed z-score of the two head scores, per model.
3. The mean over the two variants.

The ranker reads six columns (file names in brackets):

``fe_z``, ``new_z``   within-feed z of the two head scores  [pre2_fe_z, pre2_new_z]
``fe_s``, ``new_s``   the head scores as ``(x + 20) / 6``, a fixed map that keeps them inside the
                      loader's clip range of +-10; the model uses only their within-feed rank and z
                      [pre2_fe_s, pre2_new_s]
``hmax32``            the largest similarity to the 32 newest history items  [pre2_hmax32]
``fe``                the head score itself; the loader only derives the "has encoder inputs" flag
                      from it  [pre2_fe]

Train feeds before the first block have no encoder inputs: every column is
NaN there (27.4% of train feeds), and the presence flag tells the model so.

Why it is safe: see ``corpus.py`` for the snapshot rule. Alignment and
z-scores read no labels.

Run time: CPU, seconds for val and test, under a minute for train.
"""
from __future__ import annotations

import warnings

import numpy as np

from avito_ranker.data.flatten import feed_offsets, read_feeds
from avito_ranker.ensemble.zscore import feed_rank_z
from avito_ranker.features.store import n_candidates
from avito_ranker.features.trajectory_encoder.files import (
    SCORE_KEYS,
    VARIANTS,
    WINDOWS,
    score_file,
)
from avito_ranker.features.trajectory_encoder.score import candidate_rows, target_feeds

SENTINEL = -2.0
RESCALE_SHIFT, RESCALE_DIVISOR = 20.0, 6.0
RESCALE_LOW, RESCALE_HIGH = 1e-3, 9.99
COLUMNS = ("fe", "fe_z", "new", "new_z", "hmax32", "hmax8", "fe_noid", "idmax32")


def _raw(content: str, target: str, model: str) -> dict[str, np.ndarray]:
    """The six stored raw scores of a model on a target, float32."""
    return {key: np.load(score_file(content, target, model, key)).astype(np.float32) for key in SCORE_KEYS}


def _moments(x: np.ndarray) -> tuple[float, float]:
    """Mean and standard deviation over the real values (finite and above the sentinel)."""
    real = np.isfinite(x) & (x > SENTINEL)
    return x[real].astype(np.float64).mean(), x[real].astype(np.float64).std()


def alignment(x: np.ndarray, reference: np.ndarray) -> tuple[float, float]:
    """``(a, b)`` such that ``a x + b`` has the mean and standard deviation of ``reference``."""
    mean_x, std_x = _moments(x)
    mean_r, std_r = _moments(reference)
    slope = std_r / std_x
    return slope, mean_r - slope * mean_x


def _apply(x: np.ndarray, line: tuple[float, float]) -> np.ndarray:
    """``a x + b`` on the real values; the sentinel stays as it is."""
    return np.where(x > SENTINEL, line[0] * x + line[1], x).astype(np.float32)


def _model_columns(raw: dict[str, np.ndarray], lines: dict[str, tuple[float, float]] | None,
                   cand_off: np.ndarray, n_cand: np.ndarray) -> dict[str, np.ndarray]:
    """The columns of one model: aligned scores, their within-feed z, sentinel of ``idmax32`` turned into NaN."""
    aligned = {key: _apply(raw[key], lines[key]) if lines else raw[key] for key in SCORE_KEYS}
    idmax = aligned["idmax32"]
    return {
        "fe": aligned["fe"],
        "fe_z": feed_rank_z(cand_off, n_cand, aligned["fe"])[:, 1],
        "new": aligned["new"],
        "new_z": feed_rank_z(cand_off, n_cand, aligned["new"])[:, 1],
        "hmax32": aligned["hmax32"],
        "hmax8": aligned["hmax8"],
        "fe_noid": aligned["fe_noid"],
        "idmax32": np.where(idmax > SENTINEL, idmax, np.nan),
    }


def _mean_over_models(per_model: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Column-wise mean over the two variants; ``idmax32`` ignores a variant that has no value."""
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN rows of idmax32
        for name in COLUMNS:
            columns = [m[name] for m in per_model]
            mean = np.nanmean(np.stack(columns), axis=0) if name == "idmax32" else np.mean(columns, axis=0)
            out[name] = mean.astype(np.float32)
    return out


def eval_columns(split: str, content: str) -> dict[str, np.ndarray]:
    """The eight encoder columns for val or test rows, scored with the given content table."""
    feeds = read_feeds(split)
    cand_off = feeds["cand_off"].to_numpy().astype(np.int64)
    n_cand = feeds["M"].to_numpy().astype(np.int64)
    per_model = []
    for variant in VARIANTS:
        if split == "val":
            raw = _raw(content, "val", f"no_test_{variant}")
            reference = _raw(content, "val", f"all_{variant}")
            lines = {key: alignment(raw[key], reference[key]) for key in SCORE_KEYS}
        else:
            raw, lines = _raw(content, "test", f"all_{variant}"), None
        per_model.append(_model_columns(raw, lines, cand_off, n_cand))
    return _mean_over_models(per_model)


def _window_columns(window: int, content: str, sample_feeds: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Candidate rows of a train block and their eight columns."""
    _, feeds = target_feeds(f"window{window}", sample_feeds)
    _, sample = target_feeds(f"window{window}_sample", sample_feeds)
    rows = candidate_rows("train", feeds)
    sample_rows = candidate_rows("train", sample)
    at = np.searchsorted(rows, sample_rows)
    if not (rows[at] == sample_rows).all():
        raise ValueError(f"window {window}: the alignment sample is not a subset of the block")
    n_cand = read_feeds("train")["M"].to_numpy().astype(np.int64)[feeds]
    cand_off = feed_offsets(n_cand)
    per_model = []
    for variant in VARIANTS:
        raw = _raw(content, f"window{window}", f"window{window}_{variant}")
        lines = {}
        for key in SCORE_KEYS:
            reference = np.load(score_file(content, f"window{window}_sample", f"all_{variant}", key))
            lines[key] = alignment(raw[key][at], reference)
        per_model.append(_model_columns(raw, lines, cand_off, n_cand))
    return rows, _mean_over_models(per_model)


def train_columns(sample_feeds: int, content: str = "full") -> dict[str, np.ndarray]:
    """The eight encoder columns for all train rows; NaN outside the three blocks."""
    full = {name: np.full(n_candidates("train"), np.nan, np.float32) for name in COLUMNS}
    for window in WINDOWS:
        rows, columns = _window_columns(window, content, sample_feeds)
        for name, values in columns.items():
            full[name][rows] = values
    return full


def clip_safe(x: np.ndarray) -> np.ndarray:
    """``(x + 20) / 6`` in float32, NaN kept.

    The map must never reach its clip bounds, otherwise the within-feed rank
    of the result would differ from the rank of ``x``; that is checked.
    """
    x = np.asarray(x, np.float32)
    present = ~np.isnan(x)
    scaled = (x + RESCALE_SHIFT) / RESCALE_DIVISOR
    if present.any():
        low, high = float(x[present].min()), float(x[present].max())
        lowest, highest = (low + RESCALE_SHIFT) / RESCALE_DIVISOR, (high + RESCALE_SHIFT) / RESCALE_DIVISOR
        if not (lowest > RESCALE_LOW and highest < RESCALE_HIGH):
            raise ValueError(f"encoder score range [{low}, {high}] reaches the clip bounds of the rescaling")
    return np.where(present, np.clip(scaled, RESCALE_LOW, RESCALE_HIGH), np.nan).astype(np.float32)


def ranker_inputs(split: str, sample_feeds: int) -> dict[str, np.ndarray]:
    """The six encoder arrays the ranker reads for a split, keyed by file name."""
    columns = train_columns(sample_feeds) if split == "train" else eval_columns(split, "full")
    return {
        "pre2_fe_z": columns["fe_z"],
        "pre2_new_z": columns["new_z"],
        "pre2_fe_s": clip_safe(columns["fe"]),
        "pre2_new_s": clip_safe(columns["new"]),
        "pre2_hmax32": columns["hmax32"],
        "pre2_fe": columns["fe"],
    }


def id_contribution(split: str, content: str = "partial") -> np.ndarray:
    """How much the candidate's ID embedding adds to the ``fe`` score, for val or test rows.

    ``slope * (fe - fe_noid)`` on the raw scores, averaged over the variants;
    the slope is the alignment slope of ``fe`` on val and 1 on test. Using one
    slope for both terms makes a candidate without an ID exactly 0 on every
    split.
    """
    parts = []
    for variant in VARIANTS:
        model = f"no_test_{variant}" if split == "val" else f"all_{variant}"
        raw = _raw(content, split, model)
        slope = 1.0
        if split == "val":
            reference = np.load(score_file(content, "val", f"all_{variant}", "fe")).astype(np.float32)
            slope = _moments(reference)[1] / _moments(raw["fe"])[1]
        parts.append(slope * (raw["fe"].astype(np.float64) - raw["fe_noid"].astype(np.float64)))
    return np.mean(parts, 0).astype(np.float32)
