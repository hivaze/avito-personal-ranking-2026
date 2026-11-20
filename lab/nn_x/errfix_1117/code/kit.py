"""Analysts' toolkit for the error-analysis round of 2026-11-17.

    import sys; sys.path.insert(0, "<repo>/lab/nn_x/errfix_1117/code")
    import kit
    df = kit.load()                       # discovery half of validation, one row per candidate, with labels
    kit.evaluate(df, new_scores)          # paired NDCG change against the model's score, per-feed SE
    f = kit.folds(df)                     # 5 folds by user, for cross-fitted parameter choices

A rule is a Python file with a function `adjust(df: pl.DataFrame) -> np.ndarray` that returns a new score
for every row of `df`, in row order. It must use feature columns only (never `label`), read no files,
and keep its fitted parameters as constants in the file. The sealed samples call it without `label`.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import polars as pl

HERE = Path(__file__).resolve().parent.parent
DISC = HERE / "data/disc.parquet"
_DISC = 1.0 / np.log2(np.arange(2, 302))


def load(columns: list[str] | None = None) -> pl.DataFrame:
    """The discovery sample: feeds contiguous, ordered by `fidx`; rows of a feed in stored order."""
    return pl.read_parquet(DISC, columns=columns)


def feed_layout(df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(offsets, sizes) of the feeds of `df`, which must be contiguous by `fidx`."""
    f = df["fidx"].to_numpy()
    starts = np.r_[0, np.nonzero(f[1:] != f[:-1])[0] + 1]
    sizes = np.diff(np.r_[starts, len(f)])
    return starts.astype(np.int64), sizes.astype(np.int64)


def ndcg_per_feed(scores: np.ndarray, labels: np.ndarray, offsets: np.ndarray, sizes: np.ndarray) -> np.ndarray:
    """Full-list NDCG per feed, ties ordered pessimistically (negatives first), as on the leaderboard."""
    out = np.empty(len(sizes))
    for k, (o, m) in enumerate(zip(offsets, sizes)):
        s, y = scores[o:o + m], labels[o:o + m]
        order = np.lexsort((y, -s))          # by score desc, then label asc: negatives first inside a tie
        ys = y[order].astype(np.float64)
        npos = int(ys.sum())
        if npos == 0:
            out[k] = 1.0
            continue
        out[k] = (ys * _DISC[:m]).sum() / _DISC[:npos].sum()
    return out


def evaluate(df: pl.DataFrame, new_scores: np.ndarray, base: str = "score", by: str | None = None) -> dict:
    """Paired change of mean NDCG from `base` to `new_scores` over the feeds of `df`.

    Returns delta, its standard error over feeds, z, the share of feeds whose order changed and,
    with `by` (a feed-level column), the same per value of that column.
    """
    off, sz = feed_layout(df)
    y = df["label"].to_numpy()
    b = df[base].to_numpy().astype(np.float64)
    n = np.asarray(new_scores, dtype=np.float64)
    assert n.shape == b.shape and np.isfinite(n).all(), "new scores must be finite, one per row"
    nb, nn = ndcg_per_feed(b, y, off, sz), ndcg_per_feed(n, y, off, sz)
    d = nn - nb
    changed = np.array([not np.array_equal(np.argsort(-b[o:o + m], kind="stable"), np.argsort(-n[o:o + m], kind="stable"))
                        for o, m in zip(off, sz)])
    res = dict(feeds=len(sz), base=float(nb.mean()), new=float(nn.mean()), delta=float(d.mean()),
               se=float(d.std(ddof=1) / np.sqrt(len(d))), changed=float(changed.mean()))
    res["z"] = res["delta"] / res["se"] if res["se"] > 0 else 0.0
    if by is not None:
        key = df[by].to_numpy()[off]
        res["by"] = {str(v): dict(feeds=int((key == v).sum()), delta=float(d[key == v].mean()))
                     for v in np.unique(key)}
    return res


def folds(df: pl.DataFrame, k: int = 5, seed: int = 0) -> np.ndarray:
    """Fold id per row, by user (each user-0 feed is its own group)."""
    u = df["user_id"].to_numpy().astype(np.int64)
    g = np.where(u == 0, -(df["fidx"].to_numpy().astype(np.int64) + 1), u)
    groups = np.unique(g)
    rng = np.random.default_rng(seed)
    fid = dict(zip(groups.tolist(), rng.integers(0, k, len(groups)).tolist()))
    return np.array([fid[x] for x in g.tolist()], dtype=np.int8)


def load_rule(path: str | Path):
    """Import a rule file and return its `adjust` function."""
    spec = importlib.util.spec_from_file_location(Path(path).stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.adjust


def check_rule(path: str | Path, df: pl.DataFrame | None = None) -> dict:
    """Run a rule file on the discovery sample the way the sealed samples will run it (no label column)."""
    df = load() if df is None else df
    adjust = load_rule(path)
    new = adjust(df.drop("label"))
    return evaluate(df, new)
