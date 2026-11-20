"""Adversarial validation on the NN train-stage inputs (cvnd preset + f1..f5 + similarity/category inputs), candidate level.
Pairs: (a) val Tue vs val Thu (control), (b) val vs private test, (c) random-like train Sep 17-19 (M>=50) vs private test (M>=50),
(d) random-like train Sep 17-19 vs val (M>=50). 200K rows per side, 5-fold CV grouped by feed, LightGBM 16 threads."""
import os
from common_e import meta, ROOT, SCR, PRIV_BOUND
import numpy as np
import polars as pl
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

FEATS = ["seq_cv_invd20", "seq_cv_pop", "seq_cv_norm", "it_hist_all_luo", "it_hist_et1_luo", "it_hist_et2_luo", "it_hist_et3_luo",
         "it_hist_et4_luo", "ctx_nd_hcent", "ctx_nd_knn_f1", "ctx_nd_knn_h", "ctx_nd_knn_h_dev", "ctx_nd_knn_u", "ctx_nd_knn_u_dev",
         "ctx_nd_max", "ctx_nd_top5h", "ctx_nd_top5u", "ctx_nd_ucent", "ctx_hn_top1w", "ctx_hn_rrw", "ctx_hn_bestrk", "ctx_hn_margin",
         "ctx_hn_soft05", "ctx_hn_e1_rrw", "ctx_hs_mutual", "ctx_hs_k128", "ctx_hs_e1", "ctx_hs_t10",
         "f1", "f2", "f3", "f4", "f5", "maxsim32", "cos_u", "cos_h10", "cat_cnt_all", "cat_min_hpos", "H", "M"]
rng = np.random.default_rng(0)


def path(split, n):
    p = f"{ROOT}/data/feats/{split}/{n}.npy"
    return p if os.path.exists(p) else f"{ROOT}/nn/feat_cache/{split}/{n}.npy"


def sample(split, feed_mask, k):
    f, M, off, fid = meta(split)
    feeds = np.where(feed_mask)[0]
    rows = np.concatenate([np.arange(off[i], off[i] + M[i]) for i in feeds])
    if len(rows) > k:
        rows = np.sort(rng.choice(rows, k, replace=False))
    X = np.stack([np.asarray(np.load(path(split, n), mmap_mode="r")[rows], dtype=np.float32) for n in FEATS], 1)
    return X, fid[rows]


fv, Mv, _, _ = meta("val")
tsv = fv["timestamp"].to_numpy()
dayv = (tsv + 3 * 3600) // 86400
tue = dayv == dayv.min()
ft, Mt, _, _ = meta("test")
priv = ft["timestamp"].to_numpy() >= PRIV_BOUND
ftr = pl.read_parquet(f"{ROOT}/data/flat/train/feeds.parquet", columns=["timestamp", "M"])
pr = np.nan_to_num(np.load(f"{ROOT}/eda/labels_position/out/train_prandom.npz")["p_random"].astype(np.float32), nan=0.0)
tstr = ftr["timestamp"].to_numpy()
late = tstr >= np.sort(tstr)[0] + 0  # placeholder
d17 = 1789603200 + 0  # Sep 17 00:00 UTC-ish; refine below using day index
dtr = (tstr + 3 * 3600) // 86400
trmask = (pr > 0.9) & (ftr["M"].to_numpy() >= 50) & (dtr >= dtr.max() - 2)
print("train random-like late feeds:", trmask.sum())
K = 200000
S = {"val_Tue": sample("val", tue, K), "val_Thu": sample("val", ~tue, K), "val": sample("val", np.ones(len(Mv), bool), K),
     "val_M50": sample("val", Mv >= 50, K), "private": sample("test", priv, K), "private_M50": sample("test", priv & (Mt >= 50), K),
     "train_late_rand": sample("train", trmask, K)}


def adv(a, b, drop=()):
    keep = [i for i, n in enumerate(FEATS) if n not in drop]
    Xa, ga = S[a]
    Xb, gb = S[b]
    X = np.concatenate([Xa, Xb])[:, keep]
    yy = np.r_[np.zeros(len(Xa)), np.ones(len(Xb))]
    g = np.r_[ga, gb + 10 ** 8]
    oof = np.zeros(len(yy))
    imp = np.zeros(len(keep))
    for tr, te in GroupKFold(5).split(X, yy, g):
        b_ = lgb.train(dict(objective="binary", learning_rate=0.1, num_leaves=31, min_data_in_leaf=200, num_threads=16, verbose=-1, seed=0),
                       lgb.Dataset(X[tr], yy[tr]), 150)
        oof[te] = b_.predict(X[te])
        imp += b_.feature_importance("gain")
    auc = roc_auc_score(yy, oof)
    top = sorted(zip(imp / imp.sum(), [FEATS[i] for i in keep]), reverse=True)[:8]
    print(f"  {a} vs {b} (drop {len(drop)}): AUC {auc:.4f}; top gain share: " + ", ".join(f"{n} {s:.2f}" for s, n in top), flush=True)


POP = ["seq_cv_invd20", "seq_cv_pop", "seq_cv_norm", "it_hist_all_luo", "it_hist_et1_luo", "it_hist_et2_luo", "it_hist_et3_luo", "it_hist_et4_luo"]
adv("val_Tue", "val_Thu", ("M", "H"))
adv("val", "private", ("M", "H"))
adv("val", "private", ["M", "H"] + POP)
adv("val_M50", "private_M50", ("M", "H"))
adv("train_late_rand", "val_M50", ("M", "H"))
adv("train_late_rand", "val_M50", ["M", "H"] + POP)
adv("train_late_rand", "private_M50", ("M", "H"))
adv("train_late_rand", "private_M50", ["M", "H"] + POP)
