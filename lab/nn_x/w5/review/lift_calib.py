"""Reviewer info: M<50 production lift computed on ALL M<50 feeds (as in trunc_lift.json) vs (1-p_soft)-weighted
(production-only estimate) vs p_soft-weighted (hidden exploration)."""
import numpy as np, polars as pl
ROOT = "<repo>"
f = pl.read_parquet(f"{ROOT}/data/flat/train/feeds.parquet")
M = f["M"].to_numpy().astype(np.int64); n = len(M)
rk = np.load(f"{ROOT}/nn_x/insights/data_usage/out/nn_artifacts/train_rkend_true.npy")
y = np.load(f"{ROOT}/data/flat/train/cand_target.npy").astype(np.float64)
pr = np.load(f"{ROOT}/eda/labels_position/out/train_prandom.npz")
ps = np.nan_to_num(pr["p_random"].astype(np.float64), nan=0.0)
fid = np.repeat(np.arange(n), M)
sel = M[fid] < 50
rr, yy, ww = np.minimum(rk[sel], 3), y[sel], ps[fid[sel]]
for nm, w in (("all M<50 (json)", np.ones_like(ww)), ("(1-p_soft) weighted", 1 - ww), ("p_soft weighted", ww)):
    rate = np.array([(yy * w)[rr == k].sum() / w[rr == k].sum() for k in range(4)])
    print(nm, "rates", np.round(rate, 4), "log lift", np.round(np.log(rate[:3] / rate[3]), 3))
