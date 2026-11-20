"""Critic check 4 (CPU): spot-check two stream numbers with own code: newcat group mass vs actual share; other-user val click coverage/rate."""
import os
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS"):
    os.environ[k] = "12"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys; sys.path.insert(0, "<repo>/src")
import numpy as np, polars as pl
from common import feed_meta, FLAT
P = "<repo>/preds"
fv, M, off = feed_meta("val"); n = len(M); fid = np.repeat(np.arange(n), M)
y = np.load(FLAT / "val/cand_target.npy").astype(np.float64)
def z(x):
    x = np.asarray(x, np.float64); mu = np.bincount(fid, x, n) / M
    sd = np.sqrt(np.maximum(np.bincount(fid, x * x, n) / M - mu ** 2, 0)) + 1e-9
    return (x - mu[fid]) / sd[fid]
s = 0.5 * z(np.load(f"{P}/ens_FD16P20_m04_02kd_ft_val.npy")) + 0.5 * z(np.load(f"{P}/ens_FD16P20_m04_02kd_to_val.npy"))
mx = np.maximum.reduceat(s, off); e = np.exp(s - mx[fid]); q = e / np.bincount(fid, e, n)[fid]
K = np.bincount(fid, y, n)
F = pl.read_parquet("<repo>/lab/nn_x/w12/fixmiss/posthoc/out/valfeat.parquet", columns=["cat_hist_share"])
g = (np.nan_to_num(F["cat_hist_share"].to_numpy(), nan=0) <= 0).astype(float)
mf = np.bincount(fid, q * g, n); af = np.bincount(fid, y * g, n) / np.maximum(K, 1); share = np.bincount(fid, g, n) / M
ok = (share > 0) & (share < 1) & (K > 0)
print(f"newcat: feeds with both {ok.mean():.3f}; model mass {mf[ok].mean():.4f} vs actual {af[ok].mean():.4f}; positives in newcat {(y*g).sum()/y.sum():.3f}")
# calibration of mass by mass decile
qq = np.quantile(mf[ok], np.linspace(0, 1, 6)[1:-1]); b = np.digitize(mf[ok], qq)
print("newcat mass quintiles: model", np.round(np.bincount(b, mf[ok]) / np.bincount(b), 4), "actual", np.round(np.bincount(b, af[ok]) / np.bincount(b), 4))
# other-user val click coverage (any day, both halves: own user excluded)
iv = np.load(FLAT / "val/cand_item_id.npy").astype(np.int64); ur = np.repeat(fv["user_id"].to_numpy(), M)
d = pl.DataFrame({"i": iv, "u": ur, "y": y})
tot = d.group_by("i").agg(pl.col("y").sum().alias("cy"))
own = d.group_by(["i", "u"]).agg(pl.col("y").sum().alias("oy"))
dd = d.with_row_index("r").join(tot, on="i", how="left").join(own, on=["i", "u"], how="left").sort("r")
oc = (dd["cy"] - dd["oy"]).to_numpy()
cov = oc > 0
print(f"rows with another user's val click on the item (all users as source): {cov.mean():.4%}; positive rate there {y[cov].mean():.4f} vs {y.mean():.4f}")
