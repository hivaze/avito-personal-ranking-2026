"""Cross-fitted additive residual: per-feed softmax (conditional logit) with offset c*score plus L2-penalised
linear terms on (a) standardised features or (b) one-hot decile bins of each feature (a binned GAM).

usage: python clogit_cv.py <mode: lin|bins> <l2> [featureset: all|nofm]
"""
from __future__ import annotations

import sys

import numpy as np
import polars as pl
from scipy.optimize import minimize

import prep
from prep import kit

mode, l2 = sys.argv[1], float(sys.argv[2])
fset = sys.argv[3] if len(sys.argv) > 3 else "nofm"
base = sys.argv[4] if len(sys.argv) > 4 else "score"
df = prep.derive(kit.load())
names = [n for n in prep.feature_names(df) if n not in ("s_ft", "br_diff", "score", "s_to", "fidx")]
if fset == "nofm":
    names = [n for n in names if not n.startswith("fm_")]
# feed-constant columns are useless in a within-feed softmax
const = {"M", "H", "guest", "uvec_missing", "hour", "weekday", "newest_type", "newest_src", "newest_cat",
         "n_hist_cats32", "n_feed_cats"}
names = [n for n in names if n not in const and not n.startswith("fm_")]
y = df["label"].to_numpy().astype(np.float64)
s = df[base].to_numpy().astype(np.float64)
off, sz = kit.feed_layout(df)
fid = np.repeat(np.arange(len(sz)), sz)
fold = kit.folds(df)

cols, cnames = [], []
for n in names:
    x = df[n].cast(pl.Float64).fill_nan(None).to_numpy()
    nan = ~np.isfinite(x)
    if mode == "lin":
        v = np.where(nan, np.nanmean(x), x)
        v = (v - v.mean()) / (v.std() + 1e-9)
        cols.append(v); cnames.append(n)
        if nan.mean() > 0.01:
            cols.append(nan.astype(float) - nan.mean()); cnames.append(n + "_nan")
    else:
        u = np.unique(x[~nan])
        if len(u) <= 10:
            edges = u
            b = np.searchsorted(u, x)
        else:
            edges = np.unique(np.quantile(x[~nan], np.linspace(0, 1, 11))[1:-1])
            b = np.searchsorted(edges, x, side="right")
        nb = b[~nan].max() + 1
        for i in range(nb):
            m = (b == i) & ~nan
            if m.sum() < 2000:
                continue
            cols.append(m.astype(float)); cnames.append(f"{n}#{i}")
        if nan.sum() > 2000:
            cols.append(nan.astype(float)); cnames.append(n + "#nan")
X = np.column_stack(cols).astype(np.float64)
print("design", X.shape, flush=True)


def fit(tr_rows, c):
    trf = np.unique(fid[tr_rows])
    Xt, yt, st = X[tr_rows], y[tr_rows], s[tr_rows]
    szt = sz[trf]
    o = np.r_[0, np.cumsum(szt)[:-1]]
    f = np.repeat(np.arange(len(szt)), szt)
    npos = np.add.reduceat(yt, o)
    nf = len(szt)

    def obj(w):
        z = c * st + Xt @ w
        mx = np.maximum.reduceat(z, o)
        e = np.exp(z - mx[f])
        se = np.add.reduceat(e, o)
        lz = np.log(se) + mx
        ll = (yt * z).sum() - (npos * lz).sum()
        p = e / se[f]
        g = Xt.T @ (yt - npos[f] * p)
        return -ll / nf + l2 * (w @ w), -g / nf + 2 * l2 * w
    r = minimize(obj, np.zeros(X.shape[1]), jac=True, method="L-BFGS-B", options=dict(maxiter=500))
    return r.x


oof = np.zeros(len(y))
for k in range(5):
    tr = fold != k
    trf = np.unique(fid[tr])
    c = prep.softmax_scale(s[tr], y[tr], np.r_[0, np.cumsum(sz[trf])[:-1]], sz[trf])
    w = fit(tr, c)
    te = fold == k
    oof[te] = c * s[te] + X[te] @ w
r = kit.evaluate(df, oof, base=base)
line = f"clogit {mode} l2={l2} fset={fset} base={base} p={X.shape[1]}: delta={r['delta']:+.5f} se={r['se']:.5f} z={r['z']:.2f}"
print(line)
open("cv_log.txt", "a").write(line + "\n")
c = prep.softmax_scale(s, y, off, sz)
w = fit(np.ones(len(y), bool), c)
ins = kit.evaluate(df, c * s + X @ w, base=base)
print("in-sample", ins["delta"])
o = np.argsort(-np.abs(w))[:40]
for i in o:
    print(f"{cnames[i]:30s} {w[i]:+.4f}")
np.save(f"oof_clogit_{mode}_{l2}_{base}.npy", oof)
