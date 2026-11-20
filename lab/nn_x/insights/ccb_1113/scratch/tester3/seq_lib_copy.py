"""Shared helpers for the seq_ family (history sequence structure and intent)."""
import json, os, sys, time
from pathlib import Path

import numba as nb
import numpy as np
import polars as pl

sys.path.insert(0, "<repo>/src")
from metrics import ndcg_per_feed  # noqa: E402

ROOT = Path("<repo>/data")
FLAT = ROOT / "flat"
FEATS = ROOT / "feats"
CLU = ROOT / "clu_work"
WORK = ROOT / "seq_work"
WORK.mkdir(exist_ok=True)
N_CAT = 54
N_ET = 10
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)


def feeds(split):
    return pl.read_parquet(FLAT / split / "feeds.parquet")


def arr(split, name, mmap=True):
    return np.load(FLAT / split / f"{name}.npy", mmap_mode="r" if mmap else None)


def item_table(split):
    d = ROOT / "competition" / split / "item_embeddings"
    meta = json.loads((d / "data.meta").read_text())
    return np.memmap(d / "data.memmap", dtype=np.int8, mode="r", shape=tuple(meta["shape"]))


KEEP = set((Path(__file__).parent / "KEEP.txt").read_text().strip().split(",")) \
    if (Path(__file__).parent / "KEEP.txt").exists() else None


def save(split, name, x):
    """SEQ_KEEP_ONLY=1 -> only names in KEEP.txt are written (reproduce the final family without dropped files)."""
    assert name.startswith("seq_")
    if os.environ.get("SEQ_KEEP_ONLY") == "1" and KEEP is not None and name not in KEEP:
        return
    x = np.asarray(x, dtype=np.float32)
    n = len(arr(split, "cand_fidx"))
    assert len(x) == n, (name, len(x), n)
    (FEATS / split).mkdir(parents=True, exist_ok=True)
    np.save(FEATS / split / f"{name}.npy", x)


def single_ndcg(x, split="val", seed=0):
    """Single-feature NDCG on val, both directions, random tie-break, NaN -> bottom."""
    f = feeds(split)
    M = f["M"].to_numpy().astype(np.int64)
    off = f["cand_off"].to_numpy()
    y = np.asarray(arr(split, "cand_target")).astype(np.float64)
    x = np.asarray(x, dtype=np.float64)
    rng = np.random.default_rng(seed)
    noise = rng.random(len(x)) * 1e-7
    nan = np.isnan(x)
    out = []
    for sgn in (1.0, -1.0):
        s = sgn * x
        s = np.where(nan, -1e30, s)
        # rank-transform (equal values share a rank) so the noise only breaks exact ties
        sv = np.sort(s)
        r = np.searchsorted(sv, s, side="left").astype(np.float64)
        out.append(float(ndcg_per_feed(r + noise, y, off, M).mean()))
    return out
