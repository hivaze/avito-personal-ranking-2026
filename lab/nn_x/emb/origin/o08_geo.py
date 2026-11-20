"""O8: is location (non-text input of Avito's item tower) linearly present in the item vectors?
Location proxy: K2 components from the parallel codes workflow (nn_x/codes/features/out/k2_item_comp.npy: item_id ->
component of the fixed cross-category partition formed by f2 ties; 04_k2_geo.log shows components with time-zone-shifted
activity, i.e. regions). Items: those 60K items (train table vectors). Within each large category, 5-fold CV multinomial
logistic regression (128 dims, standardised) predicting the component among the top-8 components; baseline = majority;
null = same pipeline with labels permuted within category. Also mean cos of same-component vs different-component
pairs within category. Outputs out/o08.json."""
import json, time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

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
items, comp, r_ = items[ok], comp[ok], r_[ok]; cat = catv[items]
mm = np.memmap(f"{R}/data/competition/train/item_embeddings/data.memmap", dtype=np.int8, mode="r", shape=(NT, 128))
o = np.argsort(r_); X = np.empty((len(r_), 128), np.float32); X[o] = np.asarray(mm[r_[o]]).astype(np.float32)
X /= np.linalg.norm(X, axis=1, keepdims=True)
log("items with vectors", len(items), "components", len(np.unique(comp)))
rng = np.random.default_rng(0)
res = {"n_items": int(len(items)), "per_cat": {}}
cats, cc = np.unique(cat, return_counts=True)
for c in cats[np.argsort(-cc)][:8]:
    m = cat == c
    comps, cn = np.unique(comp[m], return_counts=True)
    top = comps[np.argsort(-cn)][:8]
    top = top[np.isin(top, comps[cn >= 40])]
    if len(top) < 3:
        continue
    mm_ = m & np.isin(comp, top)
    Xc, yc = X[mm_], comp[mm_]
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=300, C=1.0))
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    acc = float(cross_val_score(clf, Xc, yc, cv=skf).mean())
    accp = float(cross_val_score(clf, Xc, rng.permutation(yc), cv=skf).mean())
    maj = float(np.bincount(np.searchsorted(np.sort(top), yc)).max() / len(yc))
    # pairwise cos same vs different component
    n = len(yc); a = rng.integers(0, n, 40000); b = rng.integers(0, n, 40000); k = a != b
    cs = (Xc[a[k]] * Xc[b[k]]).sum(1); same = yc[a[k]] == yc[b[k]]
    r = {"n": int(n), "n_classes": int(len(top)), "cv_acc": acc, "majority": maj, "permuted_label_cv_acc": accp,
         "cos_same_comp": float(cs[same].mean()), "cos_diff_comp": float(cs[~same].mean())}
    res["per_cat"][int(c)] = r
    log("cat", int(c), r)
json.dump(res, open(f"{OUT}/o08.json", "w"), indent=1)
log("done")
