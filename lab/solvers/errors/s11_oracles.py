"""'If the model knew X' upper bounds on v11 (val): (a) category of the positive(s) (block oracle: candidates in a positive's
category first, v11 order inside); (b) whether the feed's positives are all cold (oracle feed flag + cold boost delta) and
the same with a nested realistic gate; (c) per-feed oracle choice between v11 and an A/A-like twin (noise floor)."""
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
base = ndcg_per_feed(z, y, off, M)
pf = pl.read_parquet(f"{SCR}/out/perfeed.parquet")
ccat = np.load(f"{ROOT}/data/flat/val/cand_category_id.npy").astype(np.int64)
key = fid * 64 + ccat
poscat = np.zeros(n * 64, bool)
poscat[key[y > 0]] = True
incat = poscat[key].astype(float)
nd = ndcg_per_feed(z + 100 * incat, y, off, M)
print(f"(a) category oracle (all positive categories first): {nd.mean():.5f} (+{nd.mean() - base.mean():.4f}); share of candidates in a positive category {incat.mean():.3f}")
# single top category only: the category of the best-ranked positive
cold = np.load(f"{ROOT}/eda/errors/synthesis/derived/val/cold_item.npy").astype(float)
allcold = (pf["poscold"].to_numpy() == "all_cold").astype(float)
for d in (0.25, 0.5, 1.0, 2.0):
    nd = ndcg_per_feed(z + d * allcold[fid] * cold, y, off, M)
    print(f"(b) oracle all-cold-positive flag, cold boost {d}: {nd.mean() - base.mean():+.5f}")
# realistic gate for 'all positives cold'
Z = np.load(f"{SCR}/out/attrs.npz")


def fmean(x):
    return np.bincount(fid, x, n) / M


X = np.stack([M, pf["H"].to_numpy(), pf["coldshare"].to_numpy(), pf["u0"].to_numpy(), pf["cold_user"].to_numpy(),
              fmean(Z["maxsim32"].astype(np.float64)), fmean(Z["luo_zero"]), fmean(Z["cat_in_hist"]),
              fmean(z * cold) / np.maximum(fmean(cold), 1e-6), fmean(Z["f4"].astype(np.float64))] +
             [fmean(Z[f"hist_et{k}_share"]) for k in range(1, 7)], 1).astype(np.float32)
uid = f["user_id"].to_numpy().astype(np.int64)
folds = list(GroupKFold(n_splits=5).split(np.arange(n), groups=uid))
fold_of = np.zeros(n, int)
for k, (_, ho) in enumerate(folds):
    fold_of[ho] = k
g = np.zeros(n)
for k in range(5):
    tr = fold_of != k
    b = lgb.train(dict(objective="binary", learning_rate=0.05, num_leaves=15, min_data_in_leaf=100, num_threads=16, verbose=-1, seed=0),
                  lgb.Dataset(X[tr], allcold[tr]), 200)
    g[fold_of == k] = b.predict(X[fold_of == k])
print(f"    nested gate AUC for all-cold-positive feeds: {roc_auc_score(allcold, g):.4f} (base rate {allcold.mean():.3f})")
betas = [0.25, 0.5, 1.0]
D = np.stack([ndcg_per_feed(z + bt * g[fid] * cold, y, off, M) - base for bt in betas])
nest = np.zeros(n)
for k in range(5):
    tr = fold_of != k
    j = int(np.argmax(np.r_[0.0, D[:, tr].mean(1)]))
    nest[fold_of == k] = 0.0 if j == 0 else D[j - 1, fold_of == k]
print(f"    realistic gate x cold boost: per-beta {np.round(D.mean(1), 5)}, nested {nest.mean():+.5f}+-{nest.std() / np.sqrt(n):.5f}")
# (c) A/A-like oracle: v11 vs v11 with FT half swapped for the W3 re-FT (nn_final5v2) -> per-feed max
zb = 0.5 * fz(np.load(f"{ROOT}/preds/nn_final5v2_val_oof.npy"), fid, M) + 0.5 * fz(np.load(f"{ROOT}/preds/nn_final5_trainonly_val.npy"), fid, M)
nb_ = ndcg_per_feed(zb, y, off, M)
print(f"(c) v11-twin (FT half = nn_final5v2): {nb_.mean():.5f}; per-feed oracle max gain {np.maximum(nb_, base).mean() - base.mean():+.5f}; feeds differing {(np.abs(nb_ - base) > 1e-9).mean():.3f}")
