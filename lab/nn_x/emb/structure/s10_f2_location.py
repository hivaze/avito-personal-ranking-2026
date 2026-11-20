"""(3f) Is the category-independent 'feed-level' subspace the same thing as f2 (suspected user x location match)?
VAL, descriptive: within-feed Spearman of f2 with the subspace similarity, within-feed AUC of f2, of the subspace cos, and of an
unfitted z-sum; and f2 levels vs mode match."""
import json, numpy as np, polars as pl
from scipy import stats

OUT = "<repo>/nn_x/emb/structure/"
V = "<repo>/data/flat/val/"
NV = 2_516_711
E = np.fromfile("<repo>/data/competition/val/item_embeddings/data.memmap", dtype=np.int8).reshape(NV, 128)
feeds = pl.read_parquet(V + "feeds.parquet"); F = len(feeds); M = feeds["M"].to_numpy()
ce = np.load(V + "cand_emb_idx.npy"); cf = np.load(V + "cand_fidx.npy").astype(np.int64); tg = np.load(V + "cand_target.npy").astype(np.float64)
cc = np.load(V + "cand_category_id.npy").astype(np.int64)
he = np.load(V + "hist_emb_idx.npy"); hf = np.load(V + "hist_fidx.npy").astype(np.int64); hc = np.load(V + "hist_category_id.npy").astype(np.int64)
hp = np.load(V + "hist_hpos.npy")
f2 = np.load(V + "cand_f2.npy").astype(np.float64)
CM = np.load(OUT + "cache/val_cat_means.npy"); Pf = np.load(OUT + "cache/feedlevel_subspace16.npy"); K = CM.shape[0]
ok = ce >= 0; hok = he >= 0
R = (E[np.maximum(ce, 0)].astype(np.float32) - CM[cc]) @ Pf
HR = (E[he[hok]].astype(np.float32) - CM[np.minimum(hc[hok], K - 1)]) @ Pf
nh = np.bincount(hf[hok], minlength=F).astype(np.float64)
HM = np.stack([np.bincount(hf[hok], weights=HR[:, d], minlength=F) for d in range(Pf.shape[1])], 1) / np.maximum(nh, 1)[:, None]
sub = (R * HM[cf]).sum(1) / (np.linalg.norm(R, axis=1) * np.linalg.norm(HM[cf], axis=1) + 1e-9)
sub = np.where(ok & (nh[cf] > 0), sub, np.nan)


def within_auc(s, rm=None):
    rm = ok & np.isfinite(s) if rm is None else rm & ok & np.isfinite(s)
    o = np.lexsort((s, cf)); y = tg[o]; ff = cf[o]; r_ = rm[o]; y = y[r_]; ff = ff[r_]
    rank = np.arange(len(y)) - np.searchsorted(ff, ff)
    npos = np.bincount(ff, weights=y, minlength=F); nn = np.bincount(ff, minlength=F) - npos
    sr = np.bincount(ff, weights=(rank + 1) * y, minlength=F); okf = (npos > 0) & (nn > 0)
    return float(((sr[okf] - npos[okf] * (npos[okf] + 1) / 2) / (npos[okf] * nn[okf])).mean())


def fz(s):
    s = np.nan_to_num(s, nan=np.nanmean(s))
    mu = np.bincount(cf, weights=s, minlength=F) / M; sd = np.sqrt(np.maximum(np.bincount(cf, weights=s * s, minlength=F) / M - mu ** 2, 1e-12))
    return (s - mu[cf]) / sd[cf]


# within-feed spearman (mean over feeds with M >= 30)
rho = []
for f in np.nonzero(M >= 30)[0][:6000]:
    a = slice(feeds["cand_off"][int(f)], feeds["cand_off"][int(f)] + M[f])
    x, y = sub[a], f2[a]; m = np.isfinite(x)
    if m.sum() > 10 and np.std(y[m]) > 0 and np.std(x[m]) > 0:
        rho.append(stats.spearmanr(x[m], y[m])[0])
# random-tie AUC for f2 (it is discrete within a feed): add tiny noise
rng = np.random.default_rng(0)
f2n = f2 + rng.random(len(f2)) * 1e-7
res = dict(within_feed_spearman_f2_vs_subcos_mean=float(np.nanmean(rho)), n_feeds=len(rho),
           auc_f2=within_auc(f2n), auc_subcos=within_auc(sub), auc_z_f2_plus_subcos=within_auc(fz(f2) + fz(sub) + rng.random(len(f2)) * 1e-9),
           auc_subcos_within_f2_level=None)
# subspace signal after conditioning on f2: AUC of subcos inside (feed, f2 level) groups -> use f2-rank-bucketed residual
g = np.round(f2, 3)
key = cf * 10000 + np.searchsorted(np.unique(g), g)
uk, inv = np.unique(key, return_inverse=True)
mu = np.bincount(inv, weights=np.nan_to_num(sub), minlength=len(uk)) / np.bincount(inv, minlength=len(uk))
res["auc_subcos_minus_feed_f2level_mean"] = within_auc(sub - mu[inv] + f2 * 100)  # rank within f2 level first, then subspace residual
res["f2_distinct_levels_per_feed_median"] = float(np.median(np.array([len(np.unique(f2[feeds["cand_off"][int(f)]:feeds["cand_off"][int(f)] + M[f]])) for f in range(0, F, 7)])))
print(res, flush=True)
json.dump(res, open(OUT + "s10_f2_location.json", "w"), indent=1)
