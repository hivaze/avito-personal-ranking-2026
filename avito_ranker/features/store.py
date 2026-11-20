"""Reading and writing the pre-computed candidate inputs.

Every input is one float32 array per split, aligned with the flat candidate
arrays of that split (``cand_item_id.npy`` and its siblings). The ranker reads
them from the folder registered as ``data.feature_cache``.

All builders write through ``save_array`` (``avito_ranker.io``), which never
replaces stored values by different ones unless asked to, and unlinks a file
before replacing it, so that a hard-linked copy keeps its stored values.
"""
from __future__ import annotations

from pathlib import Path

import numba as nb
import numpy as np

from avito_ranker.data.flatten import read_flat
from avito_ranker.io import save_array
from avito_ranker.paths import path

__all__ = ["load_input", "n_candidates", "save_array", "save_input", "save_inputs", "set_threads"]


def set_threads(threads: int) -> None:
    """Cap the numba thread pool; polars and the OpenBLAS pool are capped by the scripts (``avito_ranker.threads``)."""
    nb.set_num_threads(max(1, min(threads, nb.config.NUMBA_NUM_THREADS)))


def n_candidates(split: str) -> int:
    """Number of candidate rows of a split."""
    return len(read_flat(split, "cand_fidx"))


def save_input(out_dir: Path, split: str, name: str, values: np.ndarray, overwrite: bool = False) -> Path:
    """Write one candidate input as float32 to ``<out_dir>/<split>/<name>.npy`` (rules of ``save_array``).

    The fine-tune extras reuse 17 arrays of the train stage; when both families
    are built into one folder the second finds them in place.
    """
    values = np.ascontiguousarray(values, dtype=np.float32)
    expected = n_candidates(split)
    if values.shape != (expected,):
        raise ValueError(f"{split}/{name}: shape {values.shape}, expected ({expected},)")
    return save_array(out_dir / split / f"{name}.npy", values, overwrite)


def save_inputs(out_dir: Path, split: str, columns: dict[str, np.ndarray], overwrite: bool = False) -> list[str]:
    """Write several inputs of one split and return their names."""
    for name, values in columns.items():
        save_input(out_dir, split, name, values, overwrite)
    return list(columns)


def load_input(split: str, name: str, folder: Path | None = None, mmap: bool = True) -> np.ndarray:
    """Read one stored input (default folder: ``data.feature_cache``)."""
    source = (folder or path("data.feature_cache")) / split / f"{name}.npy"
    return np.load(source, mmap_mode="r" if mmap else None)
