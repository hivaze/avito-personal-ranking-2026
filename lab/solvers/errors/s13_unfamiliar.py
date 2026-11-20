"""Ranking inside the 'unfamiliar' subsets: same-feed pairs (pos, neg) where BOTH candidates have their category absent from
the history (or both maxsim32 < 0.7). AUC of v11 and of single signals inside those pairs."""
from common_e import meta, ROOT, SCR
import numpy as np
import numba as nb

f, M, off, fid = meta("val")
n = len(M)
y = np.load(f"{ROOT}/data/flat/val/cand_target.npy").astype(np.float64)
s = np.load(f"{SCR}/out/v11_val.npy")
Z = np.load(f"{SCR}/out/attrs.npz")
S = {"v11": s, "FT5": np.load(f"{ROOT}/preds/nn_final5_val_oof.npy").astype(np.float64),
     "TO5": np.load(f"{ROOT}/preds/nn_final5_trainonly_val.npy").astype(np.float64),
     "pre_final": np.load(f"{ROOT}/preds/pre_final_val.npy").astype(np.float64),
     "xgb_luo": np.load(f"{ROOT}/preds/xgb_x300_luo_val.npy").astype(np.float64)}
for k in ["f1", "f2", "f3", "f4", "f5", "maxsim32", "seq_max_all", "cos_u", "it_hist_all_luo", "seq_cv_pop", "it_tr_ctr", "ctx_nd_max",
          "ctx_nd_c90", "feed_cat_share", "centrality", "cold_item", "ag_age", "n_samecat_cand"]:
    S[k] = Z[k].astype(np.float64)


@nb.njit
def sub_auc(x, y, sel, off, M):
    gt = 0.0
    tot = 0.0
    for f_ in range(len(M)):
        o = off[f_]
        for i in range(o, o + M[f_]):
            if y[i] <= 0 or not sel[i]:
                continue
            for j in range(o, o + M[f_]):
                if y[j] > 0 or not sel[j]:
                    continue
                if np.isnan(x[i]) or np.isnan(x[j]):
                    continue
                if x[i] > x[j]:
                    gt += 1
                elif x[i] == x[j]:
                    gt += 0.5
                tot += 1
    return gt / max(tot, 1), tot


subsets = {"cat_not_in_hist": Z["cat_in_hist"] == 0, "maxsim<0.7": Z["maxsim32"] < 0.7,
           "cat_in_hist & maxsim>=0.7": (Z["cat_in_hist"] > 0) & (Z["maxsim32"] >= 0.7)}
for sn, sel in subsets.items():
    sel = sel.astype(np.bool_)
    print(f"{sn}: candidate share {sel.mean():.3f}, positive share {(y[sel] > 0).sum() / (y > 0).sum():.3f}, P(pos|sub)/P(pos) {y[sel].mean() / y.mean():.3f}")
    out = []
    for k, x in S.items():
        a, t = sub_auc(x, y, sel, off, M)
        out.append((a, k))
    print("   " + "  ".join(f"{k} {a:.3f}" for a, k in sorted(out, reverse=True)) + f"  (pairs {int(t)})")
