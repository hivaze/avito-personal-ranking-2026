"""O8e: which dims carry the location (K2) directions? top-|w| dims of the top-4 between-region directions, the
per-dim share of between-region variance (location eta^2 per dim, pooled over 4 categories) and its block profile."""
import json, numpy as np
exec(open("o08c_geo_struct_more.py").read().split("res = {}")[0])
cats = [2, 50, 10, 48]; good = None
for c in cats:
    u, n = np.unique(comp[cat == c], return_counts=True); s = set(u[n >= 25]); good = s if good is None else good & s
good = np.array(sorted(good)); M = []; btw = np.zeros(128); tot = np.zeros(128)
for c in cats:
    m = (cat == c) & np.isin(comp, good); mu = X[m].mean(0); tot += ((X[m] - mu) ** 2).sum(0)
    for g in good:
        mg = m & (comp == g); d = X[mg].mean(0) - mu; M.append(d * np.sqrt(mg.sum())); btw += mg.sum() * d ** 2
U = np.linalg.svd(np.array(M), full_matrices=False)[2][:4]
eta = btw / tot
out = {"top_dims_per_location_dir": [np.argsort(-np.abs(w))[:10].tolist() for w in U],
       "loc_eta2_per_dim_quantiles": [float(v) for v in np.quantile(eta, [0, .25, .5, .75, .9, 1])],
       "loc_eta2_top10_dims": np.argsort(-eta)[:10].tolist(), "loc_eta2_top10_vals": [round(float(v), 3) for v in np.sort(eta)[::-1][:10]],
       "loc_eta2_block16_means": [round(float(eta[s:s + 16].mean()), 3) for s in range(0, 128, 16)],
       "spearman_loc_eta2_vs_index": float(__import__("scipy.stats").stats.spearmanr(eta, np.arange(128))[0]),
       "share_dims_with_eta2_gt_0.3": float((eta > 0.3).mean())}
print(json.dumps(out)); json.dump(out, open(f"{OUT}/o08e.json", "w"), indent=1)
