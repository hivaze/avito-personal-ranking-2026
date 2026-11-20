"""Observed / expected positives by feature bin, where expected = per-feed softmax(c * score) * n_pos.

A ratio far from 1 (in units of its Poisson SE) marks a region where the model's within-feed probabilities are off.
"""
from __future__ import annotations

import numpy as np
import polars as pl

import prep
from prep import kit

df = prep.derive(kit.load())
y = df["label"].to_numpy().astype(np.float64)
s = df["score"].to_numpy().astype(np.float64)
off, sz = kit.feed_layout(df)
fid = np.repeat(np.arange(len(sz)), sz)
c = prep.softmax_scale(s, y, off, sz)
e = np.exp(c * s - np.maximum.reduceat(c * s, off)[fid])
p = e / np.add.reduceat(e, off)[fid]
npos = np.add.reduceat(y, off)[fid]
ex = p * npos
df = df.with_columns(pl.Series("ex", ex), pl.Series("yf", y))
print("c =", c)
names = [n for n in prep.feature_names(df) if not n.startswith("fm_")]
rows = []
for n in names:
    x = df[n].cast(pl.Float64).fill_nan(None).to_numpy()
    nan = ~np.isfinite(x)
    u = np.unique(x[~nan])
    if len(u) <= 12:
        bins = [(f"={v:g}", x == v) for v in u]
    else:
        qs = np.unique(np.quantile(x[~nan], np.linspace(0, 1, 11)))
        b = np.clip(np.searchsorted(qs, x, side="right") - 1, 0, len(qs) - 2)
        bins = [(f"[{qs[i]:.3g},{qs[i+1]:.3g}]", (b == i) & ~nan) for i in range(len(qs) - 1)]
    if nan.any():
        bins.append(("NaN", nan))
    for lab, m in bins:
        o, x_ = y[m].sum(), ex[m].sum()
        if x_ < 20:
            continue
        z = (o - x_) / np.sqrt(x_)
        rows.append((n, lab, int(m.sum()), o, x_, o / x_, z))
t = pl.DataFrame(rows, schema=["feat", "bin", "n", "obs", "exp", "ratio", "z"], orient="row")
t.write_csv("calib_scan.csv")
pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_width_chars(200)
print(t.sort(pl.col("z").abs(), descending=True).head(70))
