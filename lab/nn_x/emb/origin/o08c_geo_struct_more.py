"""O8c (= O8b with cats [2,50,10,48] and >= 25 items per component, to get more shared components): structure of the location signal (K2 components = regions, see o08).
(1) cross-category transfer: a logistic classifier of the component trained on category A, tested on category B
    (same component set) -> high transfer = an additive location embedding shared across categories (structured input);
(2) within-category between-component variance share of the 128-d vectors (multivariate eta^2);
(3) rank of the location subspace: share of the between-component scatter captured by its top-k directions (pooled
    within-category-centred class means).
Outputs out/o08c.json."""
import json, time
import numpy as np
from sklearn.linear_model import LogisticRegression

T0 = time.time()
R = "<repo>"; OUT = f"{R}/nn_x/emb/origin/out"; NT = 81557944


def log(*a):
    print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)


k2 = np.load(f"{R}/nn_x/codes/features/out/k2_item_comp.npy")
items, comp = k2[:, 0], k2[:, 1]
MAXI = 111_400_000
want = np.zeros(MAXI, bool); want[items] = True
row = np.full(MAXI, -1, np.int64); catv = np.zeros(MAXI, np.int16)
for kind in ("cand", "hist"):
    E = np.load(f"{R}/data/flat/train/{kind}_emb_idx.npy", mmap_mode="r"); I = np.load(f"{R}/data/flat/train/{kind}_item_id.npy", mmap_mode="r")
    Cc = np.load(f"{R}/data/flat/train/{kind}_category_id.npy", mmap_mode="r")
    for s in range(0, len(E), 100_000_000):
        e = np.asarray(E[s:s + 100_000_000]); i = np.asarray(I[s:s + 100_000_000]); m = want[i] & (e >= 0)
        row[i[m]] = e[m]; catv[i[m]] = np.asarray(Cc[s:s + 100_000_000])[m]
r_ = row[items]; ok = r_ >= 0
items, comp, r_ = items[ok], comp[ok], r_[ok]; cat = catv[items].astype(np.int64)
mm = np.memmap(f"{R}/data/competition/train/item_embeddings/data.memmap", dtype=np.int8, mode="r", shape=(NT, 128))
o = np.argsort(r_); X = np.empty((len(r_), 128), np.float64); X[o] = np.asarray(mm[r_[o]]).astype(np.float64)
X /= np.linalg.norm(X, axis=1, keepdims=True)
res = {}
# components present with >= 60 items in each of the 4 largest categories
cats = [2, 50, 10, 48]
good = None
for c in cats:
    u, n = np.unique(comp[cat == c], return_counts=True); s = set(u[n >= 25])
    good = s if good is None else good & s
good = np.array(sorted(good))
log("shared components", good.tolist())
res["shared_components"] = good.tolist()
tr = {}
for a in cats:
    ma = (cat == a) & np.isin(comp, good)
    clf = LogisticRegression(max_iter=500, C=1.0).fit(X[ma], comp[ma])
    for b in cats:
        if b == a:
            continue
        mb = (cat == b) & np.isin(comp, good)
        acc = float((clf.predict(X[mb]) == comp[mb]).mean())
        maj = float(np.bincount(np.searchsorted(good, comp[mb])).max() / mb.sum())
        tr[f"{a}->{b}"] = [round(acc, 4), round(maj, 4), int(mb.sum())]
res["cross_category_transfer_acc_majority_n"] = tr
accs = np.array([v[0] for v in tr.values()]); majs = np.array([v[1] for v in tr.values()])
res["transfer_mean_acc"] = float(accs.mean()); res["transfer_mean_majority"] = float(majs.mean())
log("transfer mean acc", accs.mean(), "majority", majs.mean())
# within-category between-component variance share, and subspace rank
eta = {}
Mpool = []
for c in cats:
    m = (cat == c) & np.isin(comp, good)
    Xc = X[m] - X[m].mean(0); yc = comp[m]
    tot = (Xc ** 2).sum()
    btw = 0.0
    for g in good:
        mg = yc == g
        if mg.sum():
            mu = Xc[mg].mean(0); btw += mg.sum() * (mu ** 2).sum(); Mpool.append(mu * np.sqrt(mg.sum()))
    eta[c] = float(btw / tot)
res["within_cat_between_component_variance_share"] = eta
Mp = np.array(Mpool)
sv = np.linalg.svd(Mp, compute_uv=False) ** 2
res["location_scatter_topk_share"] = {k: float(sv[:k].sum() / sv.sum()) for k in (1, 2, 3, 4, 6, 8)}
# direction consistency: per-component mean offsets (within category) correlated across categories
offs = {}
for c in cats:
    m = (cat == c) & np.isin(comp, good)
    mu_c = X[m].mean(0)
    offs[c] = np.stack([X[m & (comp == g)].mean(0) - mu_c for g in good])
cc = []
for i, a in enumerate(cats):
    for b in cats[i + 1:]:
        A = offs[a]; B = offs[b]
        cc.append(float(np.mean([A[k] @ B[k] / np.linalg.norm(A[k]) / np.linalg.norm(B[k]) for k in range(len(good))])))
res["mean_cos_same_component_offset_across_categories"] = float(np.mean(cc))
res["offset_norm_mean"] = float(np.mean([np.linalg.norm(offs[c], axis=1).mean() for c in cats]))
log({k: v for k, v in res.items() if k != "cross_category_transfer_acc_majority_n"})
json.dump(res, open(f"{OUT}/o08b.json", "w"), indent=1)
log("done")
