"""O4: is there an appended attribute block (hypothesis d) or any index structure in the 128 dims?
Val table, all 2.52M rows (category per row from cand/hist arrays).
- aligned contiguous blocks (8/16/32/64): CV of the block norm across items vs median CV over 50 random dim subsets
  of the same size (a separately normalised appended block would have CV ~ 0);
- per-dim eta^2(category), per-dim |corr| with item age (f4), f5, log history popularity (item-level means over val
  candidates), and their dependence on dim index / aligned block;
- dim-dim correlation: mean |corr| adjacent (|i-j| = 1) vs all pairs; within vs between aligned 16/32 blocks;
- per-dim skew / excess kurtosis / distinct-value counts (a one-hot or categorical block would be discrete).
Outputs out/o04.json."""
import json, time
import numpy as np
from scipy.stats import skew, kurtosis, spearmanr

T0 = time.time()
R = "<repo>"; OUT = f"{R}/nn_x/emb/origin/out"; NV = 2516711
res = {}


def log(*a):
    print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)


def l(n):
    return np.load(f"{R}/data/flat/val/{n}.npy", mmap_mode="r")


x = np.asarray(np.memmap(f"{R}/data/competition/val/item_embeddings/data.memmap", dtype=np.int8, mode="r", shape=(NV, 128))).astype(np.float32)
cat = np.zeros(NV, np.int64)
for kind in ("cand", "hist"):
    e = np.asarray(l(f"{kind}_emb_idx")); c = np.asarray(l(f"{kind}_category_id")); m = e >= 0
    cat[e[m]] = c[m]
rng = np.random.default_rng(0)
# 1) block norm CV
blk = {}
for b in (8, 16, 32, 64):
    cv_al = []
    for s in range(0, 128, b):
        nb = np.linalg.norm(x[:, s:s + b], axis=1); cv_al.append(float(nb.std() / nb.mean()))
    cv_rd = []
    for _ in range(50):
        d = rng.choice(128, b, replace=False); nb = np.linalg.norm(x[:, d], axis=1); cv_rd.append(float(nb.std() / nb.mean()))
    blk[b] = {"aligned_cv": cv_al, "aligned_min": min(cv_al), "random_median": float(np.median(cv_rd)),
              "random_min": float(np.min(cv_rd)), "min_ratio": min(cv_al) / float(np.median(cv_rd))}
    log("block", b, blk[b]["aligned_min"], blk[b]["random_median"])
res["block_norm_cv"] = blk
# 2) per-dim eta^2(category)
K = cat.max() + 1
cnt = np.bincount(cat, minlength=K).astype(np.float64)
mu = x.mean(0); tot = ((x - mu) ** 2).sum(0)
eta = np.zeros(128)
for d in range(128):
    s = np.bincount(cat, weights=x[:, d], minlength=K)
    m = cnt > 0
    eta[d] = ((s[m] ** 2 / cnt[m]).sum() - NV * mu[d] ** 2) / tot[d]
res["eta2_category"] = {"min": float(eta.min()), "median": float(np.median(eta)), "max": float(eta.max()),
                        "share_gt_0.5": float((eta > 0.5).mean()), "spearman_vs_index": float(spearmanr(eta, np.arange(128))[0]),
                        "block32_means": [float(eta[s:s + 32].mean()) for s in range(0, 128, 32)],
                        "multivariate_between_share": float(((np.bincount(cat, minlength=K)[:, None] * 0) .sum()))}
# multivariate R2 of category one-hot (trace ratio)
S = np.zeros((K, 128))
for d in range(128):
    S[:, d] = np.bincount(cat, weights=x[:, d], minlength=K)
m = cnt > 0
between = ((S[m] ** 2 / cnt[m][:, None]).sum(0) - NV * mu ** 2).sum()
res["eta2_category"]["multivariate_between_share"] = float(between / tot.sum())
log("eta2", res["eta2_category"])
# 3) per-dim corr with item-level attributes (val candidates)
ce = np.asarray(l("cand_emb_idx")); ok = ce >= 0
import sys
sys.path.insert(0, f"{R}/feats_v2/life_")
from post_time import age_hours
import polars as pl
fts = pl.read_parquet(f"{R}/data/flat/val/feeds.parquet")["timestamp"].to_numpy().astype(np.float64)
ct = fts[np.asarray(l("cand_fidx"))]
age = age_hours(np.asarray(l("cand_f4")), np.load(f"{R}/data/feats/life_f4_age_knots.npy"))
tpost = ct - age * 3600
attrs = {"f4": np.asarray(l("cand_f4")), "f5": np.asarray(l("cand_f5")), "f2": np.asarray(l("cand_f2")),
         "f1": np.asarray(l("cand_f1")), "f3": np.asarray(l("cand_f3")),
         "log_hist_pop": np.log1p(np.load(f"{R}/data/feats/val/it_hist_all_luo.npy")), "tpost_day": tpost / 86400}
cn = np.bincount(ce[ok], minlength=NV).astype(np.float64)
has = cn > 0
rows = np.nonzero(has)[0]
pc = {}
for k_, v in attrs.items():
    v = np.asarray(v, np.float64); okv = ok & np.isfinite(v)
    sm = np.bincount(ce[okv], weights=v[okv], minlength=NV); nn = np.bincount(ce[okv], minlength=NV)
    rr = np.nonzero(nn > 0)[0]
    a = sm[rr] / nn[rr]
    xs = x[rr]
    # within-category: demean both by category
    cc = cat[rr]
    def dm(z):
        s_ = np.bincount(cc, weights=z, minlength=K); n_ = np.bincount(cc, minlength=K)
        return z - (s_ / np.maximum(n_, 1))[cc]
    ad = dm(a)
    cors = np.array([np.corrcoef(dm(xs[:, d].astype(np.float64)), ad)[0, 1] for d in range(128)])
    # linear R2 (within category), 2-fold
    Z = np.column_stack([dm(xs[:, d].astype(np.float64)) for d in range(128)])
    h = rng.random(len(rr)) < 0.5
    w = np.linalg.lstsq(Z[h], ad[h], rcond=None)[0]
    r2 = 1 - ((ad[~h] - Z[~h] @ w) ** 2).mean() / ad[~h].var()
    pc[k_] = {"n_items": int(len(rr)), "max_abs_corr": float(np.abs(cors).max()), "argmax_dim": int(np.abs(cors).argmax()),
              "median_abs_corr": float(np.median(np.abs(cors))), "top5_abs": [float(v_) for v_ in np.sort(np.abs(cors))[::-1][:5]],
              "spearman_abscorr_vs_index": float(spearmanr(np.abs(cors), np.arange(128))[0]),
              "within_cat_linear_R2_heldout": float(r2)}
    log("attr", k_, pc[k_])
res["per_dim_attr_corr_within_cat"] = pc
# 4) dim-dim correlation structure
Cm = np.corrcoef(x[rng.choice(NV, 500000, replace=False)].T)
A = np.abs(Cm); iu = np.triu_indices(128, 1)
adj = np.array([A[i, i + 1] for i in range(127)])
res["dimcorr"] = {"mean_abs_all": float(A[iu].mean()), "mean_abs_adjacent": float(adj.mean()), "max_abs": float(A[iu].max())}
for b in (16, 32):
    blkid = np.arange(128) // b
    same = blkid[iu[0]] == blkid[iu[1]]
    res["dimcorr"][f"within_block{b}"] = float(A[iu][same].mean()); res["dimcorr"][f"between_block{b}"] = float(A[iu][~same].mean())
log("dimcorr", res["dimcorr"])
# 5) marginal shape per dim
xs = x[rng.choice(NV, 500000, replace=False)]
sk = skew(xs, axis=0); ku = kurtosis(xs, axis=0)
res["marginals"] = {"skew_range": [float(sk.min()), float(sk.max())], "skew_median": float(np.median(sk)),
                    "exkurt_range": [float(ku.min()), float(ku.max())], "exkurt_median": float(np.median(ku)),
                    "sd_range": [float(xs.std(0).min()), float(xs.std(0).max())],
                    "mean_range": [float(xs.mean(0).min()), float(xs.mean(0).max())],
                    "spearman_sd_vs_index": float(spearmanr(xs.std(0), np.arange(128))[0]),
                    "spearman_absmean_vs_index": float(spearmanr(np.abs(xs.mean(0)), np.arange(128))[0])}
log("marginals", res["marginals"])
json.dump(res, open(f"{OUT}/o04.json", "w"), indent=1)
log("done")
