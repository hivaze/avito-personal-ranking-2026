"""Paired per-feed comparison with the standard slices of the acceptance rule.

  python nn/pcmp.py val  A:B [A2:B2 ...]   # A = baseline, B = treatment; names of preds/nn_<name>.npy (e.g. v12_all_e10_trainonly_val)
  python nn/pcmp.py hold A:B [A2:B2 ...]   # names of preds/holdout/nn_<name>_holdout.npy
A side may be an equal-weight per-feed z ensemble "x+y+z". Several pairs (seeds) are averaged per feed before the SE.
val slices: all, M<50, M>=50, Tue, Thu. holdout slices: all, late (latest 20% of holdout feeds by timestamp).
"""
import sys

import numpy as np
import polars as pl

sys.path.insert(0, "<repo>/src")
from metrics import ndcg_per_feed  # noqa: E402

ROOT = "<repo>"


def fz(s, fid, M):
    s = s.astype(np.float64)
    mu = np.bincount(fid, s, len(M)) / M
    sd = np.sqrt(np.maximum(np.bincount(fid, s * s, len(M)) / M - mu ** 2, 0)) + 1e-6
    return (s - mu[fid]) / sd[fid]


def main(mode, pairs):
    if mode == "val":
        f = pl.read_parquet(f"{ROOT}/data/flat/val/feeds.parquet")
        M = f["M"].to_numpy().astype(np.int64)
        off = f["cand_off"].to_numpy().astype(np.int64)
        y = np.load(f"{ROOT}/data/flat/val/cand_target.npy").astype(np.float64)
        day = (f["timestamp"].to_numpy() + 3 * 3600) // 86400
        day = np.where(day <= day.min() + 1, 0, 1)
        path = lambda n: f"{ROOT}/nn_x/r6/preds/nn_{n}.npy"  # noqa: E731
        slices = {"all": np.ones(len(M), bool), "M<50": M < 50, "M>=50": M >= 50, "Tue": day == 0, "Thu": day == 1}
    else:
        D = f"{ROOT}/nn_x/r6/preds/holdout"
        M, off = np.load(f"{D}/holdout_meta.npy")
        y = np.load(f"{D}/holdout_y.npy").astype(np.float64)
        tr = pl.read_parquet(f"{ROOT}/data/flat/train/feeds.parquet", columns=["timestamp", "M"])
        pr = np.nan_to_num(np.load(f"{ROOT}/eda/labels_position/out/train_prandom.npz")["p_random"].astype(np.float32), nan=0.0)
        trM = tr["M"].to_numpy()
        pr[trM < 50] = 0.0
        cand = np.where((pr > 0.9) & (trM >= 50))[0]
        hold = np.sort(np.random.default_rng(12345).choice(cand, 30000, replace=False))
        assert (trM[hold] == M).all()
        ts = tr["timestamp"].to_numpy()[hold]
        path = lambda n: f"{D}/nn_{n}_holdout.npy"  # noqa: E731
        slices = {"all": np.ones(len(M), bool), "late20%": ts >= np.quantile(ts, 0.8)}
    fid = np.repeat(np.arange(len(M)), M)
    load = lambda s: np.mean([fz(np.load(path(n)), fid, M) for n in s.split("+")], 0)  # noqa: E731
    ds, rows = [], []
    for p in pairs:
        a, b = p.split(":")
        pa, pb = ndcg_per_feed(load(a), y, off, M), ndcg_per_feed(load(b), y, off, M)
        ds.append(pb - pa)
        rows.append(f"  {b[:60]} {pb.mean():.5f} vs {a[:60]} {pa.mean():.5f}: {pb.mean() - pa.mean():+.5f}")
    print("\n".join(rows))
    d = np.mean(ds, 0)
    print("  pooled: " + "  ".join(f"{k} {d[m].mean():+.5f}+-{d[m].std() / np.sqrt(m.sum()):.5f}" for k, m in slices.items()))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
