"""Paired per-feed deltas (leak-free feeds) of qt_<tag>.npz results vs a reference tag. Per-seed deltas use matching folds."""
import sys, numpy as np
from pathlib import Path
import polars as pl

D = Path("<repo>/eda/errors/synthesis/out")
ref = sys.argv[1]
tags = sys.argv[2:]
r = np.load(D / f"qt_{ref}.npz")
kf = ~r["leak"]
ts = pl.read_parquet("<repo>/data/flat/val/feeds.parquet")["timestamp"].to_numpy()
thu = ts >= np.median(ts)
print(f"ref {ref}: stacker {r['per'][:, kf].mean():.5f} blend {r['perb'][:, kf].mean():.5f}  (n feeds {kf.sum()})")
for t in tags:
    a = np.load(D / f"qt_{t}.npz")
    s = min(len(a["per"]), len(r["per"]))
    for key, lab in (("per", "stacker"), ("perb", "blend")):
        d = (a[key][:s] - r[key][:s]).mean(0)
        dk = d[kf]; dt = d[kf & thu]
        seeds = [(a[key][i] - r[key][i])[kf].mean() for i in range(s)]
        print(f"{t:14s} {lab:7s} {a[key][:s, kf].mean():.5f}  d {dk.mean():+.5f} ± {dk.std() / np.sqrt(len(dk)):.5f}  "
              f"(t={dk.mean() / (dk.std() / np.sqrt(len(dk))):+.1f})  later-half d {dt.mean():+.5f} ± {dt.std() / np.sqrt(len(dt)):.5f}  "
              f"per-seed {[round(x, 5) for x in seeds]}")
