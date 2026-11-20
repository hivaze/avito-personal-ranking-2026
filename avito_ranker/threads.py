"""CPU thread caps of a script run.

The machine the pipeline ran on is shared, so every script that works on the
CPU takes ``--threads`` (default 12). Polars and OpenBLAS size their thread
pools once, when they are imported, from ``POLARS_MAX_THREADS`` and
``OPENBLAS_NUM_THREADS``; numba sizes its pool from ``NUMBA_NUM_THREADS``.
A script therefore calls :func:`limit_threads` after parsing its arguments
and before it imports any other module of the package (nearly all of them
load polars). This module imports nothing heavy for that reason.

Without a cap, polars starts one thread per core (240 on the machine we
used). The LightGBM fit of the position proxy is the one exception: it keeps
the 40 threads of the stored fit (``avito_ranker.features.exploration``).
"""
from __future__ import annotations

import os
import sys

DEFAULT_THREADS = 12


def limit_threads(threads: int, openblas_threads: int | None = None) -> None:
    """Cap polars and numba at ``threads`` threads, and OpenBLAS at ``openblas_threads`` when given.

    Raises if polars or numba is already loaded, because then the cap would
    not take effect.
    """
    loaded = [name for name in ("polars", "numba") if name in sys.modules]
    if loaded:
        raise RuntimeError(f"limit_threads must run before {' and '.join(loaded)} is imported")
    if openblas_threads is not None:
        if "numpy" in sys.modules:
            raise RuntimeError("the OpenBLAS cap must be set before numpy is imported")
        os.environ["OPENBLAS_NUM_THREADS"] = str(openblas_threads)
    threads = max(1, int(threads))
    os.environ["POLARS_MAX_THREADS"] = str(threads)
    os.environ["NUMBA_NUM_THREADS"] = str(threads)
