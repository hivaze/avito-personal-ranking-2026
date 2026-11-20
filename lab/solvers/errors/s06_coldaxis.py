"""The cold/warm axis: pair-type AUCs on val, cold-item z by split (val / public / private), FT vs TO disagreement on test,
share of top-k slots taken by cold items, observable-segment routing of the FT/TO weight (nested user folds)."""
from common_e import meta, ndcg_per_feed, ROOT, SCR, fz, PRIV_BOUND
import numpy as np
import numba as nb
import polars as pl
from sklearn.model_selection import GroupKFold


@nb.njit
def pair_auc(score, y, cold, off, M):
    # returns counts [type][gt, eq, n] for types: 0 cold+/warm-, 1 warm+/cold-, 2 warm+/warm-, 3 cold+/cold-
    out = np.zeros((4, 3))
    for f_ in range(len(M)):
        o = off[f_]
        for i in range(o, o + M[f_]):
            if y[i] <= 0:
                continue
            for j in range(o, o + M[f_]):
                if y[j] > 0:
                    continue
                t = 0
                if cold[i] and not cold[j]:
                    t = 0
                elif (not cold[i]) and cold[j]:
                    t = 1
                elif (not cold[i]) and (not cold[j]):
                    t = 2
                else:
                    t = 3
                if score[i] > score[j]:
                    out[t, 0] += 1
                elif score[i] == score[j]:
                    out[t, 1] += 1
                out[t, 2] += 1
    return out


f, M, off, fid = meta("val")
y = np.load(f"{ROOT}/data/flat/val/cand_target.npy").astype(np.float64)
cold = np.load(f"{ROOT}/eda/errors/synthesis/derived/val/cold_item.npy").astype(np.bool_)
P = {"v11": np.load(f"{SCR}/out/v11_val.npy"), "FT5": np.load(f"{ROOT}/preds/nn_final5_val_oof.npy").astype(np.float64),
     "TO5": np.load(f"{ROOT}/preds/nn_final5_trainonly_val.npy").astype(np.float64),
     "FT5cd": np.load(f"{ROOT}/preds/nn_final5cd_val_oof.npy").astype(np.float64),
     "TO5cd12": np.load(f"{ROOT}/preds/nn_final5cd12_trainonly_val.npy").astype(np.float64),
     "mix4": 0.5 * fz(np.load(f"{ROOT}/preds/nn_final4_val_oof.npy"), fid, M) + 0.5 * fz(np.load(f"{ROOT}/preds/nn_final4_trainonly_val.npy"), fid, M),
     "xgb_luo": np.load(f"{ROOT}/preds/xgb_x300_luo_val.npy").astype(np.float64),
     "pre_final": np.load(f"{ROOT}/preds/pre_final_val.npy").astype(np.float64)}
print("pair-type AUC on val (cold = never a train candidate nor in a train history):")
for k, v in P.items():
    c = pair_auc(v, y, cold, off, M)
    a = (c[:, 0] + 0.5 * c[:, 1]) / c[:, 2]
    print(f"  {k:10s} cold+/warm- {a[0]:.4f}  warm+/cold- {a[1]:.4f}  warm+/warm- {a[2]:.4f}  cold+/cold- {a[3]:.4f}  (pair shares {np.round(c[:, 2] / c[:, 2].sum(), 3)})")
# observed positive rate of cold vs warm candidates by day
ts = f["timestamp"].to_numpy()
day = (ts + 3 * 3600) // 86400
day = np.where(day <= day.min() + 1, 0, 1)
for d_, lab in ((0, "Tue"), (1, "Thu")):
    m = day[fid] == d_
    print(f"  {lab}: cold share {cold[m].mean():.3f}; P(pos|cold)/P(pos) {y[m & cold].mean() / y[m].mean():.3f}; P(pos|warm)/P(pos) {y[m & ~cold].mean() / y[m].mean():.3f}")


def cold_stats(split, cold_s, zFT, zTO, zmix, fid_s, M_s, mask_feed, lab):
    m = mask_feed[fid_s]
    cc = cold_s & m
    ww = (~cold_s) & m
    # top-k share of cold items
    zr = zmix
    order = np.lexsort((-zr, fid_s))
    _, M2, off2, _ = meta(split)
    rank = np.empty(len(zr), np.int64)
    rank[order] = np.arange(len(zr)) - np.repeat(off2, M2)
    t1 = (rank == 0) & m
    t3 = (rank < 3) & m
    corr = np.corrcoef(zFT[m], zTO[m])[0, 1]
    print(f"  {lab:8s} cold share {cc.sum() / m.sum():.3f} | mean z of cold items: FT {zFT[cc].mean():+.3f} TO {zTO[cc].mean():+.3f} v11 {zmix[cc].mean():+.3f} "
          f"| FT-TO z gap on cold {zFT[cc].mean() - zTO[cc].mean():+.3f} | top-1 cold share {cold_s[t1].mean():.3f} top-3 {cold_s[t3].mean():.3f} "
          f"(ratio to base {cold_s[t1].mean() / (cc.sum() / m.sum()):.2f}) | corr(zFT,zTO) {corr:.4f}")


print("\ncold-item calibration and FT/TO disagreement by split:")
zFT, zTO = fz(P["FT5"], fid, M), fz(P["TO5"], fid, M)
zmix = 0.5 * zFT + 0.5 * zTO
cold_stats("val", cold, zFT, zTO, zmix, fid, M, np.ones(len(M), bool), "val")
cold_stats("val", cold, zFT, zTO, zmix, fid, M, day == 0, "val Tue")
cold_stats("val", cold, zFT, zTO, zmix, fid, M, day == 1, "val Thu")
ft, Mt, offt, fidt = meta("test")
coldt = np.load(f"{ROOT}/eda/errors/synthesis/derived/test/cold_item.npy").astype(np.bool_)
zFTt = fz(np.load(f"{ROOT}/preds/nn_final5_test.npy"), fidt, Mt)
zTOt = fz(np.load(f"{ROOT}/preds/nn_final5_trainonly_test.npy"), fidt, Mt)
zmixt = 0.5 * zFTt + 0.5 * zTOt
tst = ft["timestamp"].to_numpy()
pub = tst < PRIV_BOUND
cold_stats("test", coldt, zFTt, zTOt, zmixt, fidt, Mt, pub, "public")
cold_stats("test", coldt, zFTt, zTOt, zmixt, fidt, Mt, ~pub, "private")
dt = (tst + 3 * 3600) // 86400
for i, dd in enumerate(np.unique(dt)):
    cold_stats("test", coldt, zFTt, zTOt, zmixt, fidt, Mt, dt == dd, f"test d{i}")
# rank correlation of v11 test vs v10 and FT5 vs TO5 per feed by split, and per-feed disagreement
