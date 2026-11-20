"""O5: behavioural fingerprints in the item vectors (val table items seen as val candidates).
- popularity direction: within-category Spearman of log popularity with cos(item, mean user_emb) and with
  cos(item, mean item vector); popularity = log1p(item_stats hist_all[item_id]) (history occurrences over all splits,
  label-free) and log1p(cand_cnt_train) (times shown in train feeds);
- held-out within-category linear R2 of log popularity from the 128 dims (2-fold);
- user tower vs items: cos(mean user, mean item); share of user vectors' variance inside the item PCA subspace
  (top-32/64 item PCs) vs a random subspace; train user_emb (500K sample) and val user_emb.
Outputs out/o05.json."""
import json, time
import numpy as np, polars as pl

T0 = time.time()
R = "<repo>"; OUT = f"{R}/nn_x/emb/origin/out"; NV = 2516711
res = {}
rng = np.random.default_rng(0)


def log(*a):
    print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)


def l(split, n):
    return np.load(f"{R}/data/flat/{split}/{n}.npy", mmap_mode="r")


x = np.asarray(np.memmap(f"{R}/data/competition/val/item_embeddings/data.memmap", dtype=np.int8, mode="r", shape=(NV, 128))).astype(np.float64)
x /= np.linalg.norm(x, axis=1, keepdims=True)
item = np.full(NV, -1, np.int64); cat = np.zeros(NV, np.int64)
for kind in ("cand", "hist"):
    e = np.asarray(l("val", f"{kind}_emb_idx")); it = np.asarray(l("val", f"{kind}_item_id")); c = np.asarray(l("val", f"{kind}_category_id"))
    m = e >= 0; item[e[m]] = it[m]; cat[e[m]] = c[m]
ce = np.asarray(l("val", "cand_emb_idx"))
iscand = np.zeros(NV, bool); iscand[ce[ce >= 0]] = True
z = np.load(f"{R}/data/feats/item_stats.npz")
pops = {"log_hist_all": np.log1p(z["hist_all"][item].astype(np.float64)),
        "log_cand_cnt_train": np.log1p(z["cand_cnt_train"][item].astype(np.float64))}
ue_v = l("val", "user_emb").astype(np.float64); ue_v = ue_v[np.linalg.norm(ue_v, axis=1) > 0]
UT = l("train", "user_emb"); ut = np.asarray(UT[np.sort(rng.choice(len(UT), 500000, replace=False))]).astype(np.float64)
ut = ut[np.linalg.norm(ut, axis=1) > 0]
mu_user = ut.mean(0) / np.linalg.norm(ut.mean(0))
mu_userv = ue_v.mean(0) / np.linalg.norm(ue_v.mean(0))
mu_item = x.mean(0) / np.linalg.norm(x.mean(0))
res["mean_dirs"] = {"cos_mu_user_train_mu_item": float(mu_user @ mu_item), "cos_mu_user_val_mu_item": float(mu_userv @ mu_item),
                    "cos_mu_user_train_val": float(mu_user @ mu_userv),
                    "user_mean_norm_ratio_train": float(np.linalg.norm((ut / np.linalg.norm(ut, axis=1, keepdims=True)).mean(0))),
                    "item_mean_norm_ratio": float(np.linalg.norm(x.mean(0)))}
log(res["mean_dirs"])
rows = np.nonzero(iscand)[0]
cc = cat[rows]; K = cat.max() + 1


def rank_within(v, g):
    df = pl.DataFrame({"v": v, "g": g})
    return df.select((pl.col("v").rank("average").over("g") / pl.len().over("g")).alias("r"))["r"].to_numpy()


def dm(v, g):
    s = np.bincount(g, weights=v, minlength=K); n = np.bincount(g, minlength=K)
    return v - (s / np.maximum(n, 1))[g]


su = x[rows] @ mu_user; si = x[rows] @ mu_item
# a direction orthogonal to both, as a placebo
pl_dir = rng.standard_normal(128); pl_dir -= (pl_dir @ mu_item) * mu_item; pl_dir /= np.linalg.norm(pl_dir)
sp = x[rows] @ pl_dir
# user-minus-item component: the part of the mean user direction orthogonal to the mean item direction
du = mu_user - (mu_user @ mu_item) * mu_item; du /= np.linalg.norm(du)
sd_ = x[rows] @ du
res["popularity"] = {}
for k, p in pops.items():
    pr = p[rows]
    rp = rank_within(pr, cc)
    out = {}
    for nm, s in (("cos_mean_user", su), ("cos_mean_item", si), ("cos_user_orth_item", sd_), ("cos_random_placebo", sp)):
        rs = rank_within(s, cc)
        out[nm] = float(np.corrcoef(rs, rp)[0, 1])
    Z = np.column_stack([dm(x[rows, d], cc) for d in range(128)]); y = dm(pr, cc)
    h = rng.random(len(rows)) < 0.5
    w = np.linalg.lstsq(Z[h], y[h], rcond=None)[0]
    out["within_cat_linear_R2_heldout"] = float(1 - ((y[~h] - Z[~h] @ w) ** 2).mean() / y[~h].var())
    yc = pr - dm(pr, cc)
    out["category_only_R2"] = float(1 - ((pr - yc) ** 2).mean() / pr.var())
    res["popularity"][k] = out
    log(k, out)
# user vectors inside the item subspace
xc = x[rng.choice(NV, 400000, replace=False)]
mu = xc.mean(0)
ev, V = np.linalg.eigh(np.cov((xc - mu).T)); V = V[:, ::-1]
uc = ut / np.linalg.norm(ut, axis=1, keepdims=True); uc = uc - uc.mean(0)
tot = (uc ** 2).sum()
sub = {}
for kk in (16, 32, 64):
    sub[kk] = {"user_var_share_in_item_topPCs": float(((uc @ V[:, :kk]) ** 2).sum() / tot),
               "user_var_share_in_item_bottomPCs": float(((uc @ V[:, -kk:]) ** 2).sum() / tot),
               "random_subspace_expected": kk / 128}
res["user_in_item_subspace"] = sub
log(sub)
json.dump(res, open(f"{OUT}/o05.json", "w"), indent=1)
log("done")
