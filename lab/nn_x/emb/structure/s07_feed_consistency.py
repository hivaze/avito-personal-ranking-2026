"""(3c) Which dims are 'user/feed-level' rather than 'category-level'? Hypothesis: the right-skewed, category-agnostic dims
(0, 6, 43, 77, 83 ...) carry a property shared by everything one user sees/clicks regardless of category (e.g. location / seller type).
VAL: per dim, between-feed variance share (eta^2 of feed id) of candidate vectors and of history vectors, raw and after removing
the per-category mean; and the across-feed correlation between the feed's history mean and its candidate mean (category-residualised)."""
import json, numpy as np, polars as pl
from scipy import stats

OUT = "<repo>/nn_x/emb/structure/"
V = "<repo>/data/flat/val/"
NV = 2_516_711
E = np.fromfile("<repo>/data/competition/val/item_embeddings/data.memmap", dtype=np.int8).reshape(NV, 128)
ce = np.load(V + "cand_emb_idx.npy"); cc = np.load(V + "cand_category_id.npy").astype(np.int64); cf = np.load(V + "cand_fidx.npy").astype(np.int64)
he = np.load(V + "hist_emb_idx.npy"); hc = np.load(V + "hist_category_id.npy").astype(np.int64); hf = np.load(V + "hist_fidx.npy").astype(np.int64)
ok = ce >= 0; Xc = E[ce[ok]].astype(np.float32); cc, cf = cc[ok], cf[ok]
okh = he >= 0; Xh = E[he[okh]].astype(np.float32); hc, hf = hc[okh], hf[okh]
F = int(max(cf.max(), hf.max()) + 1)


def resid_cat(X, c):
    K = c.max() + 1; n = np.bincount(c, minlength=K).astype(np.float64)
    M = np.stack([np.bincount(c, weights=X[:, d], minlength=K) for d in range(128)], 1) / np.maximum(n, 1)[:, None]
    return X - M[c]


def eta2_group(X, g, G):
    n = np.bincount(g, minlength=G).astype(np.float64)
    S = np.stack([np.bincount(g, weights=X[:, d], minlength=G) for d in range(128)], 1)
    gm = S / np.maximum(n, 1)[:, None]; mu = X.mean(0)
    between = (n[:, None] * (gm - mu) ** 2).sum(0)
    total = ((X - mu) ** 2).sum(0)
    return between / total, gm, n


res = {}
for nm, X, c, g in [("cand", Xc, cc, cf), ("hist", Xh, hc, hf)]:
    e_raw, _, n_ = eta2_group(X, g, F)
    Xr = resid_cat(X, c)
    e_res, gm, n_ = eta2_group(Xr, g, F)
    # null: random groups of the same sizes
    gperm = np.random.default_rng(0).permutation(g)
    e_null, _, _ = eta2_group(Xr, gperm, F)
    res[nm] = dict(eta2_feed_raw=e_raw.round(4).tolist(), eta2_feed_catresid=e_res.round(4).tolist(), null_mean=float(e_null.mean()),
                   catresid_min_med_max=[float(e_res.min()), float(np.median(e_res)), float(e_res.max())],
                   top10_dims_catresid=[(int(i), round(float(e_res[i]), 4)) for i in np.argsort(-e_res)[:10]])
    if nm == "cand":
        gm_c, n_c = gm, n_
    else:
        gm_h, n_h = gm, n_
    print(nm, res[nm]["catresid_min_med_max"], res[nm]["top10_dims_catresid"], "null", res[nm]["null_mean"], flush=True)
both = (n_c >= 20) & (n_h >= 20)
corr = np.array([np.corrcoef(gm_c[both, d], gm_h[both, d])[0, 1] for d in range(128)])
res["hist_vs_cand_feedmean_corr_catresid"] = corr.round(4).tolist()
res["hist_vs_cand_corr_min_med_max"] = [float(corr.min()), float(np.median(corr)), float(corr.max())]
res["hist_vs_cand_corr_top10"] = [(int(i), round(float(corr[i]), 3)) for i in np.argsort(-corr)[:10]]
# relation with per-dim skew and eta2(category)
s2 = json.load(open(OUT + "s02_probes.json")) if __import__("os").path.exists(OUT + "s02_probes.json") else None
z = np.load(OUT + "cache/train_sample.npz"); Q = z["E"].astype(np.float64)
sk = stats.skew(Q, 0)
cat = z["cat"].astype(np.int64); K = cat.max() + 1; n = np.bincount(cat, minlength=K)
cm = np.stack([np.bincount(cat, weights=Q[:, d], minlength=K) for d in range(128)], 1) / np.maximum(n, 1)[:, None]
eta_cat = (n[:, None] * (cm - Q.mean(0)) ** 2).sum(0) / (Q.var(0) * len(Q))
e_res = np.array(res["cand"]["eta2_feed_catresid"])
res["spearman_feedeta_vs_skew"] = float(stats.spearmanr(e_res, sk)[0])
res["spearman_feedeta_vs_eta2cat"] = float(stats.spearmanr(e_res, eta_cat)[0])
res["spearman_histcandcorr_vs_skew"] = float(stats.spearmanr(corr, sk)[0])
grp = [0, 6, 43, 77, 83]
res["skewed_lowcat_dims"] = {int(d): dict(skew=round(float(sk[d]), 3), eta2_cat=round(float(eta_cat[d]), 4), feed_eta_catresid=round(float(e_res[d]), 4),
                                          hist_cand_corr=round(float(corr[d]), 3)) for d in grp}
res["other_dims_median"] = dict(feed_eta_catresid=float(np.median(np.delete(e_res, grp))), hist_cand_corr=float(np.median(np.delete(corr, grp))))
# correlation among the skewed dims
R = np.corrcoef(Q[:, grp].T)
res["skewed_dims_corr_matrix"] = R.round(3).tolist()
print(json.dumps({k: v for k, v in res.items() if k not in ("cand", "hist", "hist_vs_cand_feedmean_corr_catresid")}, indent=0), flush=True)
json.dump(res, open(OUT + "s07_feed_consistency.json", "w"), indent=1)
