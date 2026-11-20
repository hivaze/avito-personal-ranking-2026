"""The pre-computed inputs of a member, and how each one enters the model.

A pre-computed input is one stored float32 value per candidate (see
``avito_ranker.features``). Which inputs a member reads is its *view*:

* the full view has 34 inputs;
* the popularity-light view drops the 8 inputs that count how often an item was
  seen or co-visited and adds one clipped exposure count, 27 inputs in all.

A view is data. It is listed in ``configs/recipes/model_*.toml`` and stored in
every checkpoint, so the same code serves both kinds of member.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

SCALINGS = ("log_count", "clip_5", "clip_10", "presence")


@dataclass(frozen=True)
class InputColumn:
    """One pre-computed input.

    ``file`` is the array name in the feature cache (``<split>/<file>.npy``).

    ``scaling`` is the fixed transform applied when the array is loaded. It
    does not depend on the split, so train, val and test rows are treated
    alike:

    * ``log_count``: ``log1p(max(x, 0))``, for counts;
    * ``clip_5``: clip to [-5, 5], for values that are already z-scores;
    * ``clip_10``: clip to [-10, 10], for everything else;
    * ``presence``: 1 where the stored value exists and 0 where it is NaN.

    ``rank_and_z`` adds two more columns computed inside the model: the
    percentile rank and the z-score of the input among the candidates of its
    feed. ``keep_raw = False`` drops the raw column and keeps only those two,
    which we use for inputs whose scale differs between splits.
    """

    file: str
    scaling: str
    rank_and_z: bool = False
    keep_raw: bool = True

    def __post_init__(self) -> None:
        if self.scaling not in SCALINGS:
            raise ValueError(f"input {self.file!r}: unknown scaling {self.scaling!r}, expected one of {SCALINGS}")
        if not self.keep_raw and not self.rank_and_z:
            raise ValueError(f"input {self.file!r} would not enter the model at all")

    def scale(self, values: np.ndarray) -> np.ndarray:
        """Apply the column's fixed transform to a stored array; missing values become 0."""
        if self.scaling == "presence":
            return (~np.isnan(values)).astype(np.float32)
        values = values.astype(np.float32)
        if self.scaling == "clip_5":
            values = np.clip(values, -5, 5)
        elif self.scaling == "log_count":
            values = np.log1p(np.maximum(values, 0))
        else:
            values = np.clip(values, -10, 10)
        return np.nan_to_num(values, nan=0.0)


@dataclass(frozen=True)
class InputView:
    """An ordered list of pre-computed inputs. The order fixes the column order of the dense block."""

    columns: tuple[InputColumn, ...]

    def __post_init__(self) -> None:
        keys = [(c.file, c.scaling == "presence") for c in self.columns]
        if len(set(keys)) != len(keys):
            raise ValueError("an input is listed twice in the view")

    @property
    def n_inputs(self) -> int:
        """Number of stored inputs read per candidate."""
        return len(self.columns)

    @property
    def raw(self) -> list[int]:
        """Positions of the inputs that enter the dense block as they are stored."""
        return [i for i, c in enumerate(self.columns) if c.keep_raw]

    @property
    def ranked(self) -> list[int]:
        """Positions of the inputs that also enter as within-feed rank and z-score."""
        return [i for i, c in enumerate(self.columns) if c.rank_and_z]

    @property
    def n_dense_columns(self) -> int:
        """Columns the view adds to the dense block: the raw ones plus two per ranked input."""
        return len(self.raw) + 2 * len(self.ranked)

    @classmethod
    def from_rows(cls, rows: list[dict]) -> InputView:
        """Build a view from the ``inputs`` list of a recipe or a checkpoint."""
        return cls(tuple(InputColumn(**row) for row in rows))

    def to_rows(self) -> list[dict]:
        """The view as plain dictionaries, the form stored in recipes and checkpoints."""
        return [asdict(column) for column in self.columns]
