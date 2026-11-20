"""O8d: are the location and category signals axis-aligned (appended block) or spread over all 128 dims?
Participation ratio over dims (PR = (sum w^2)^2 / sum w^4 for unit direction w; 128 = evenly spread, ~1 = one dim)
of (i) the top-8 between-region directions (K2 components, 4 categories, as in o08c) and (ii) the top-8 between-category
directions (val table); share of |w|^2 in the best aligned 16-dim block. Outputs out/o08d.json."""
import json, numpy as np
exec(open("o08c_geo_struct_more.py").read().split("res = {}")[0])  # loads X, comp, cat (same selection as o08c)
cats = [2, 50, 10, 48]; good = None
for c in cats:
    u, n = np.unique(comp[cat == c], return_counts=True); s = set(u[n >= 25]); good = s if good is None else good & s
good = np.array(sorted(good)); M = []
for c in cats:
    m = (cat == c) & np.isin(comp, good); mu = X[m].mean(0)
    for g in good:
        mg = m & (comp == g); M.append((X[mg].mean(0) - mu) * np.sqrt(mg.sum()))
U = np.linalg.svd(np.array(M), full_matrices=False)[2][:8]
def pr(w): w2 = w ** 2; return float(w2.sum() ** 2 / (w2 ** 2).sum())
def blk(w): w2 = w ** 2; return float(max(w2[s:s + 16].sum() for s in range(0, 128, 16)) / w2.sum())
res = {"location_dirs_PR": [pr(w) for w in U], "location_dirs_best16block_share": [blk(w) for w in U]}
NV = 2516711
xv = np.asarray(np.memmap(f"{R}/data/competition/val/item_embeddings/data.memmap", dtype=np.int8, mode="r", shape=(NV, 128))).astype(np.float64)
xv /= np.linalg.norm(xv, axis=1, keepdims=True)
cv = np.zeros(NV, np.int64)
for kind in ("cand", "hist"):
    e = np.load(f"{R}/data/flat/val/{kind}_emb_idx.npy"); c_ = np.load(f"{R}/data/flat/val/{kind}_category_id.npy"); m = e >= 0; cv[e[m]] = c_[m]
mu = xv.mean(0); Mc = []
for c in np.unique(cv):
    m = cv == c; Mc.append((xv[m].mean(0) - mu) * np.sqrt(m.sum()))
Uc = np.linalg.svd(np.array(Mc), full_matrices=False)[2][:8]
res["category_dirs_PR"] = [pr(w) for w in Uc]; res["category_dirs_best16block_share"] = [blk(w) for w in Uc]
rng = np.random.default_rng(0); rw = rng.standard_normal((2000, 128))
res["random_dir_PR_median"] = float(np.median([pr(w) for w in rw])); res["random_dir_best16block_median"] = float(np.median([blk(w) for w in rw]))
res["cos_location_top4_vs_category_top8_subspace_share"] = float(((U[:4] @ Uc.T) ** 2).sum() / 4)
print(json.dumps(res)); json.dump(res, open(f"{OUT}/o08d.json", "w"), indent=1)
