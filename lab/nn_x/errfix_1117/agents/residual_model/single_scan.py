"""One-feature residual scan: new = c*score + w*x, w fitted by per-feed softmax MLE on 4 folds, NDCG measured on
the 5th. One variant per feature (no grid), so the multiplicity is the number of features scanned.

usage: python single_scan.py [head]   (head: x is multiplied by 1[rank <= 30])
"""
import sys

import numpy as np
import polars as pl
from scipy.optimize import minimize_scalar

import prep
from prep import kit

head = len(sys.argv) > 1 and sys.argv[1] == "head"
df = prep.derive(kit.load())
y = df["label"].to_numpy().astype(np.float64)
s = df["score"].to_numpy().astype(np.float64)
off, sz = kit.feed_layout(df)
fid = np.repeat(np.arange(len(sz)), sz)
fold = kit.folds(df)
rk = df["rank"].to_numpy()
const = {"M", "H", "guest", "uvec_missing", "hour", "weekday", "newest_type", "newest_src", "newest_cat",
         "n_hist_cats32", "n_feed_cats", "score", "s_to", "fidx"}
names = [n for n in prep.feature_names(df) if n not in const and not n.startswith("fm_")]
cfold = []
for k in range(5):
    tr = fold != k
    trf = np.unique(fid[tr])
    cfold.append(prep.softmax_scale(s[tr], y[tr], np.r_[0, np.cumsum(sz[trf])[:-1]], sz[trf]))


def nll(w, z0, x, rows_y, o, f, npos):
    z = z0 + w * x
    mx = np.maximum.reduceat(z, o)
    lz = np.log(np.add.reduceat(np.exp(z - mx[f]), o)) + mx
    return -((rows_y * z).sum() - (npos * lz).sum())


folds_idx = []
for k in range(5):
    tr = fold != k
    trf = np.unique(fid[tr])
    o = np.r_[0, np.cumsum(sz[trf])[:-1]]
    f = np.repeat(np.arange(len(trf)), sz[trf])
    folds_idx.append((tr, o, f, np.add.reduceat(y[tr], o)))

rows = []
for n in names:
    x = df[n].cast(pl.Float64).fill_nan(None).to_numpy()
    nan = ~np.isfinite(x)
    x = np.where(nan, np.nanmedian(x), x)
    x = (x - x.mean()) / (x.std() + 1e-12)
    if head:
        x = x * (rk <= 30)
    oof = np.zeros(len(y))
    ws = []
    for k, (tr, o, f, npos) in enumerate(folds_idx):
        c = cfold[k]
        r = minimize_scalar(nll, bounds=(-1, 1), method="bounded",
                            args=(c * s[tr], x[tr], y[tr], o, f, npos), options=dict(xatol=1e-4))
        ws.append(r.x)
        te = fold == k
        oof[te] = c * s[te] + r.x * x[te]
    ev = kit.evaluate(df, oof)
    rows.append((n, float(np.mean(ws)), float(np.std(ws)), ev["delta"], ev["se"], ev["z"]))
    print(f"{n:22s} w={np.mean(ws):+.4f}±{np.std(ws):.4f} oof={ev['delta']:+.5f} se={ev['se']:.5f} z={ev['z']:+.2f}",
          flush=True)
t = pl.DataFrame(rows, schema=["feat", "w", "w_sd", "delta", "se", "z"], orient="row").sort("z", descending=True)
t.write_csv("single_scan_head.csv" if head else "single_scan.csv")
print(t.head(15))
