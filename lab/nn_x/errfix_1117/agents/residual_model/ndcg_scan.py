"""NDCG-direct cross-fitted scan of one-term residuals: new = score + w * g(x), w chosen on a grid by mean NDCG
on 4 folds and applied to the 5th. g(x) = x standardised over all rows (NaN -> median), optionally gated.

usage: python ndcg_scan.py <gate: none|head30|big100|small50> [out.csv]
Multiplicity: each feature x gate is one variant (the grid choice is inside the cross-fit).
"""
import sys

import numpy as np
import polars as pl

import prep
from prep import kit

gate = sys.argv[1] if len(sys.argv) > 1 else "none"
out = sys.argv[2] if len(sys.argv) > 2 else f"ndcg_scan_{gate}.csv"
df = prep.derive(kit.load())
y = df["label"].to_numpy()
s = df["score"].to_numpy().astype(np.float64)
off, sz = kit.feed_layout(df)
F = prep.FastNDCG(y, off, sz)
fold = kit.folds(df)
ffold = fold[off]
base = F(s)
rk = df["rank"].to_numpy()
M = df["M"].to_numpy()
G = dict(none=np.ones(len(s)), head30=(rk <= 30).astype(float), big100=(M >= 100).astype(float),
         small50=(M < 50).astype(float))[gate]
const = {"M", "H", "guest", "uvec_missing", "hour", "weekday", "newest_type", "newest_src", "newest_cat",
         "n_hist_cats32", "n_feed_cats", "score", "s_to", "fidx", "rank", "gap_top", "gap_next", "rank_pct"}
names = [n for n in prep.feature_names(df) if n not in const and not n.startswith("fm_")]
W = np.array([-0.3, -0.2, -0.1, -0.05, -0.02, 0, 0.02, 0.05, 0.1, 0.2, 0.3])
rows = []
for n in names:
    x = df[n].cast(pl.Float64).fill_nan(None).to_numpy()
    nan = ~np.isfinite(x)
    x = np.where(nan, np.nanmedian(x), x)
    x = (x - x.mean()) / (x.std() + 1e-12) * G
    nd = np.stack([F(s + w * x + 1e-7 * s) if w != 0 else base for w in W]) - base   # (W, feeds)
    oof = np.zeros(len(sz))
    picks = []
    for k in range(5):
        trm = ffold != k
        j = int(np.argmax(nd[:, trm].mean(1)))
        picks.append(W[j])
        oof[~trm] = nd[j, ~trm]
    d, se = oof.mean(), oof.std(ddof=1) / np.sqrt(len(oof))
    jin = int(np.argmax(nd.mean(1)))
    rows.append((n, gate, d, se, d / se if se > 0 else 0.0, W[jin], nd[jin].mean(), str(picks)))
    print(f"{n:22s} {gate} oof={d:+.5f} se={se:.5f} z={d / se if se > 0 else 0:+.2f} in-sample w={W[jin]:+.2f} "
          f"d={nd[jin].mean():+.5f} picks={picks}", flush=True)
t = pl.DataFrame(rows, schema=["feat", "gate", "delta", "se", "z", "w_in", "d_in", "picks"], orient="row")
t.sort("z", descending=True).write_csv(out)
