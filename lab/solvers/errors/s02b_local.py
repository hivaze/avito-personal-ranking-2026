"""Symmetric local residual AUC: for positives with K <= rank <= M-1-K, compare with the negatives in ranks [r-K, r+K].
With a symmetric window the v11 score itself has local AUC ~0.5, so a feature far from 0.5 carries residual (unused) ordering info.
Also reports the slice of low-ranked positives (pct >= 0.5) only."""
from common_e import meta, ROOT, SCR
import numpy as np
import numba as nb
import polars as pl

f, M, off, fid = meta("val")
n = len(M)
y = np.load(f"{ROOT}/data/flat/val/cand_target.npy").astype(np.float64)
r = np.load(f"{SCR}/out/v11_rank.npy")
s = np.load(f"{SCR}/out/v11_val.npy")
pct = r / np.maximum(M[fid] - 1, 1)
Z = np.load(f"{SCR}/out/attrs.npz")
A = {k: Z[k].astype(np.float64) for k in Z.files if k not in ("et_am", "es_am")}
A["v11"] = s
for nm in ["pre_final_val", "pre2_final_val", "nn_final5_trainonly_val", "nn_final5_val_oof", "nn_final4_val_oof", "xgb_x300_luo_val"]:
    A["P:" + nm] = np.load(f"{ROOT}/preds/{nm}.npy").astype(np.float64)
order = np.lexsort((r, fid))


@nb.njit
def local_pairs(order, off, M, y, K):
    pa = []
    pb = []
    for f_ in range(len(M)):
        o = off[f_]
        m = M[f_]
        for i in range(K, m - K):
            a = order[o + i]
            if y[a] <= 0:
                continue
            for j in range(i - K, i + K + 1):
                b = order[o + j]
                if y[b] > 0:
                    continue
                pa.append(a)
                pb.append(b)
    return np.array(pa), np.array(pb)


def lauc(x, pa, pb):
    xa, xb = x[pa], x[pb]
    okp = ~(np.isnan(xa) | np.isnan(xb))
    d = np.where(xa > xb, 1.0, np.where(xa == xb, 0.5, 0.0))
    d = np.where(okp, d, 0.5)
    a = d[okp].mean()
    fs = np.bincount(fid[pa], np.where(okp, d - a, 0), n)
    se = np.sqrt((fs ** 2).sum()) / okp.sum()
    return a, (a - 0.5) / max(se, 1e-12), (xa[okp] == xb[okp]).mean()


rows = []
for K in (5, 10):
    pa, pb = local_pairs(order, off, M, y, K)
    low = pct[pa] >= 0.5
    print(f"K={K}: pairs {len(pa)}, positives {len(np.unique(pa))}, low-ranked-positive pairs {low.sum()}")
    for k, x in A.items():
        a, z, eq = lauc(x, pa, pb)
        a2, z2, _ = lauc(x, pa[low], pb[low])
        rows.append(dict(K=K, attr=k, local_auc=a, z=z, eq=eq, low_auc=a2, low_z=z2))
T = pl.DataFrame(rows).pivot(on="K", index="attr", values=["local_auc", "z", "low_auc", "low_z"])
T = T.sort("z_5", descending=True)
pl.Config.set_tbl_rows(100)
pl.Config.set_tbl_width_chars(250)
pl.Config.set_float_precision(4)
print(T)
T.write_csv("<repo>/solvers/out/errors/s02b_local.csv")
