"""Intent-gated mixture: does a feed-level 'exploration' gate + an exploration expert help v11?
explore feed := every positive has maxsim32 < 0.75 and its category not among the last 10 history events (SHIFT-9 definition).
score = z(v11) + g * beta * z(a), a in {-maxsim32, log1p(it_hist_all_luo), -cat_cnt_all}; g = oracle label or nested gate prob.
Nested over the nn/finetune.py user folds (gate fit and beta choice on 4 folds)."""
import os
from common_e import meta, ndcg_per_feed, ROOT, SCR, fz
import numpy as np
import polars as pl
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

f, M, off, fid = meta("val")
n = len(M)
y = np.load(f"{ROOT}/data/flat/val/cand_target.npy").astype(np.float64)
z = fz(np.load(f"{SCR}/out/v11_val.npy"), fid, M)
pf = pl.read_parquet(f"{SCR}/out/perfeed.parquet")
Z = np.load(f"{SCR}/out/attrs.npz")
ms = Z["maxsim32"].astype(np.float64)
cmh = Z["cat_min_hpos"].astype(np.float64)
rel = (ms >= 0.75) | (cmh < 10)
pos = y > 0
nrelpos = np.bincount(fid, (pos & rel).astype(float), n)
explore = (nrelpos == 0).astype(float)
print(f"explore feeds: {explore.mean():.4f}")
base = ndcg_per_feed(z, y, off, M)
print(f"v11 on explore feeds {base[explore > 0].mean():.4f}, others {base[explore == 0].mean():.4f}")


def fmean(x):
    return np.bincount(fid, x, n) / M


def fmax(x):
    return np.maximum.reduceat(x, off)


# feed-level observable features
order = np.lexsort((-z, fid))
zs_sorted = z[order]
top1 = zs_sorted[off]
top2 = zs_sorted[np.minimum(off + 1, off + M - 1)]
p = np.exp(z - np.repeat(np.maximum.reduceat(z, off), M))
p = p / np.repeat(np.add.reduceat(p, off), M)
ent = -np.bincount(fid, p * np.log(p + 1e-12), n) / np.log(M)
F = {"M": M, "H": pf["H"].to_numpy(), "n_uniq_cat": fmean(Z["hist_n_uniq_cat"]), "coldshare": pf["coldshare"].to_numpy(),
     "u0": pf["u0"].to_numpy().astype(float), "newU": pf["cold_user"].to_numpy().astype(float),
     "ms_mean": fmean(ms), "ms_max": fmax(ms), "cos_u_mean": fmean(np.nan_to_num(Z["cos_u"])), "catin_share": fmean(Z["cat_in_hist"]),
     "catrecent_share": fmean(Z["cat_recent3"]), "luo0_share": fmean(Z["luo_zero"]), "top_margin": top1 - top2, "v11_entropy": ent,
     "last_et": fmean(Z["last_et"])}
for k in range(1, 7):
    F[f"et{k}"] = fmean(Z[f"hist_et{k}_share"])
for k in range(0, 5):
    F[f"es{k}"] = fmean(Z[f"hist_es{k}_share"])
X = np.stack(list(F.values()), 1).astype(np.float32)
uid = f["user_id"].to_numpy().astype(np.int64)
folds = list(GroupKFold(n_splits=5).split(np.arange(n), groups=uid))
fold_of = np.zeros(n, int)
for k, (_, ho) in enumerate(folds):
    fold_of[ho] = k
g = np.zeros(n)
for k in range(5):
    tr = fold_of != k
    b = lgb.train(dict(objective="binary", learning_rate=0.05, num_leaves=15, min_data_in_leaf=100, num_threads=16, verbose=-1, seed=0),
                  lgb.Dataset(X[tr], explore[tr]), 200)
    g[fold_of == k] = b.predict(X[fold_of == k])
print(f"gate AUC (nested) {roc_auc_score(explore, g):.4f}; mean g {g.mean():.3f}")
A = {"-maxsim32": -ms, "log_luo": np.log1p(Z["it_hist_all_luo"].astype(np.float64)), "-cat_cnt_all": -np.log1p(Z["cat_cnt_all"].astype(np.float64)),
     "-maxsim+luo": -fz(ms, fid, M) + fz(np.log1p(Z["it_hist_all_luo"].astype(np.float64)), fid, M)}
betas = [0.25, 0.5, 1.0]
for an, a in A.items():
    za = fz(a, fid, M)
    for gn, gv in (("oracle", explore), ("gate", g), ("gate^2", g ** 2), ("const", np.full(n, 1.0))):
        D = np.stack([ndcg_per_feed(z + gv[fid] * bt * za, y, off, M) - base for bt in betas])
        nest = np.zeros(n)
        for k in range(5):
            tr = fold_of != k
            j = int(np.argmax(np.r_[0.0, D[:, tr].mean(1)]))
            nest[fold_of == k] = 0.0 if j == 0 else D[j - 1, fold_of == k]
        best = D.mean(1)
        print(f"  {an:12s} {gn:7s}: per-beta {np.round(best, 5)}  nested {nest.mean():+.5f}+-{nest.std() / np.sqrt(n):.5f}")
