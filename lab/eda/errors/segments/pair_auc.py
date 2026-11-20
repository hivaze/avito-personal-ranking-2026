"""Pairwise AUC by (positive, negative) coldness type, whole feed, val leak-masked: is the cold problem cold-vs-cold
discrimination or cold-vs-warm ordering?"""
import os
os.environ.setdefault("NUMBA_NUM_THREADS", "16")
import numpy as np, polars as pl, numba as nb
ROOT = "<repo>"; FLAT = f"{ROOT}/data/flat"; OUT = f"{ROOT}/eda/errors/segments"
d = pl.read_parquet(f"{OUT}/feed_table.parquet"); keep = ~d["leak"].to_numpy()
f = pl.read_parquet(f"{FLAT}/val/feeds.parquet")
M = f["M"].to_numpy().astype(np.int64); off = f["cand_off"].to_numpy()
y = np.load(f"{FLAT}/val/cand_target.npy").astype(np.float64)
cat = np.load(f"{FLAT}/val/cand_category_id.npy").astype(np.int64)
item = np.load(f"{FLAT}/val/cand_item_id.npy")
seen = np.zeros(111_400_000, bool)
seen[np.load(f"{FLAT}/train/cand_item_id.npy", mmap_mode="r")[:]] = True
seen[np.load(f"{FLAT}/train/hist_item_id.npy", mmap_mode="r")[:]] = True
cold = (~seen[item]).astype(np.int64); del seen

@nb.njit(parallel=True)
def pair_stats(s, yy, cc, cat, off, M, keep, same_cat):
    # out[f, t, 0] = wins, out[f, t, 1] = pairs; t = 2*cold_pos + cold_neg
    out = np.zeros((len(M), 4, 2))
    for f in nb.prange(len(M)):
        if not keep[f]:
            continue
        o, m = off[f], M[f]
        for i in range(o, o + m):
            if yy[i] > 0:
                for j in range(o, o + m):
                    if yy[j] == 0 and (same_cat < 0 or (same_cat == 1) == (cat[i] == cat[j])):
                        t = 2 * cc[i] + cc[j]
                        out[f, t, 1] += 1
                        if s[i] > s[j]:
                            out[f, t, 0] += 1
                        elif s[i] == s[j]:
                            out[f, t, 0] += 0.5
    return out

sc = {"nn_oof": np.load(f"{ROOT}/preds/nn_final2_val_oof.npy"), "nn_tr": np.load(f"{ROOT}/preds/nn_final2_trainonly_val.npy"),
      "stk_clean": np.load(f"{ROOT}/preds/stk2_res4_clean_val.npy"), "cb": np.load(f"{ROOT}/preds/cb_tr300k_none_val.npy"),
      "maxsim32": np.load(f"{ROOT}/data/feats/val/maxsim32.npy"), "f1": np.load(f"{FLAT}/val/cand_f1.npy"),
      "f4": np.load(f"{FLAT}/val/cand_f4.npy"), "f5": np.load(f"{FLAT}/val/cand_f5.npy")}
names = ["warm+ vs warm-", "warm+ vs cold-", "cold+ vs warm-", "cold+ vs cold-"]
for same_cat, title in [(-1, "all pairs"), (1, "same-category pairs"), (0, "cross-category pairs")]:
    print(f"\n### Pairwise AUC, {title} (pair-weighted over clean feeds)")
    print("| score | " + " | ".join(names) + " |"); print("|---" * 5 + "|")
    for k, s in sc.items():
        st = pair_stats(np.nan_to_num(s.astype(np.float64), nan=-1e9), y, cold, cat, off, M, keep, same_cat)
        w, p = st[:, :, 0].sum(0), st[:, :, 1].sum(0)
        print(f"| {k} | " + " | ".join(f"{w[t] / p[t]:.4f}" for t in range(4)) + " |")
    print(f"| pair share | " + " | ".join(f"{p[t] / p.sum():.3f}" for t in range(4)) + " |")
