"""Writing artefacts without losing the stored ones.

Two kinds of output exist, with one rule each.

* Rebuildable arrays and tables (flat arrays, inputs, corpora, encoder
  scores): ``save_array`` and ``save_frame`` leave a stored file alone when
  the rebuild gives the same values, and refuse to replace it with other
  values unless ``overwrite`` is set. Running a builder against the stored
  files therefore checks them.
* Outputs of a run (checkpoints, predictions, pools, submissions, teacher
  files): ``refuse_existing`` stops before any work when one of them exists.

A replaced file is unlinked first and written as a new file. Stored inputs
are hard-linked between folders (``data.feature_cache`` shares 266 files with
``data.features``), and writing into a shared inode would change the copy in
the other folder as well.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

COMPARE_STEP_VALUES = 50_000_000  # values compared at a time when a build meets a stored array


def refuse_existing(*files: Path) -> None:
    """Raise ``FileExistsError`` naming every one of ``files`` that exists."""
    existing = [str(f) for f in files if Path(f).exists()]
    if existing:
        raise FileExistsError(f"refusing to replace {', '.join(existing)}")


def clear_for_rewrite(target: Path, overwrite: bool) -> None:
    """Prepare ``target`` to be written as a new file.

    Without ``overwrite`` an existing file is refused; with it the file is
    unlinked, so a hard-linked copy elsewhere keeps its stored contents.
    """
    target = Path(target)
    if target.exists():
        if not overwrite:
            raise FileExistsError(f"{target} exists; pass overwrite to replace it")
        target.unlink()
    target.parent.mkdir(parents=True, exist_ok=True)


def save_array(target: Path, array: np.ndarray, overwrite: bool = False) -> Path:
    """Write an array to ``target`` unless the file already holds exactly these values.

    A stored array is never replaced by different values unless ``overwrite``
    is set: the stored artefacts took hours to build and trained checkpoints
    depend on their exact values. With ``overwrite`` the old file is unlinked
    first, so a hard-linked copy elsewhere keeps its stored values.
    """
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        stored = np.load(target, mmap_mode="r")
        if stored.dtype == array.dtype and stored.shape == array.shape and _same_values(stored, array):
            return target
        del stored
        if not overwrite:
            raise FileExistsError(f"{target} exists and holds other values; pass overwrite to replace it")
        target.unlink()
    np.save(target, array)
    return target


def save_frame(target: Path, frame: pl.DataFrame, overwrite: bool = False) -> Path:
    """Write a table as parquet under the rules of ``save_array``: equal tables are left alone."""
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if pl.read_parquet(target).equals(frame):
            return target
        if not overwrite:
            raise FileExistsError(f"{target} exists and holds another table; pass overwrite to replace it")
        target.unlink()
    frame.write_parquet(target)
    return target


def _same_values(a: np.ndarray, b: np.ndarray) -> bool:
    """Element-wise equality with NaN equal to NaN, compared in pieces to keep memory flat."""
    if a.ndim == 0 or len(a) == 0:
        return bool(np.array_equal(a, b, equal_nan=True))
    step = max(1, COMPARE_STEP_VALUES // max(1, a[0].size))
    return all(np.array_equal(a[i:i + step], b[i:i + step], equal_nan=True) for i in range(0, len(a), step))
