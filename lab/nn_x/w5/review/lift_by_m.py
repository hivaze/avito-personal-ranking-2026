"""Reviewer info: heterogeneity of the TRUNC lifts by M bucket inside the two production groups ((1-p_soft) weighted)."""
import numpy as np, polars as pl
ROOT = "<repo>"
f = pl.read_parquet(f"{ROOT}/data/flat/train/feeds.parquet")
M = f["M"].to_numpy().astype(np.int64); n = len(M)
rk = np.minimum(np.load(f"{ROOT}/nn_x/insights/data_usage/out/nn_artifacts/train_rkend_true.npy"), 3)
y = np.load(f"{ROOT}/data/flat/train/cand_target.npy").astype(np.float64)
pr = np.load(f"{ROOT}/eda/labels_position/out/train_prandom.npz")
ps = np.nan_to_num(pr["p_random"].astype(np.float64), nan=0.0)
pn = ps.copy(); pn[M < 50] = 0
fid = np.repeat(np.arange(n), M)
for lo, hi in ((1, 16), (17, 32), (33, 49), (50, 100), (101, 200), (201, 10**6)):
    fsel = (M >= lo) & (M <= hi) & ~((M >= 50) & (pn > 0.5))
    sel = fsel[fid]
    rr, yy, w = rk[sel], y[sel], 1 - ps[fid[sel]]
    rate = np.array([(yy * w)[rr == k].sum() / max(w[rr == k].sum(), 1e-9) for k in range(4)])
    print(f"M {lo}-{hi}: feeds {int(fsel.sum())}, weighted rates {np.round(rate, 4)}, log lift {np.round(np.log(rate[:3] / rate[3]), 3)}")
