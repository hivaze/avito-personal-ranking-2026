"""03: item-embedding geometry of the categories.
Item pool: unique items with an embedding from val+test candidates and val+test histories (each item's own category), up to 20K per
category (seed 0); train candidates top up categories with < 20K items (train table). Embeddings int8 -> float32, L2-normalised.
Outputs: mean pairwise cosine within / between categories (53x53, exact from category sums), centroid cosine, kNN category
confusion (1K queries per category vs a 5K-per-category pool, k=20, self excluded), hierarchical clustering cuts.
out/emb_mats.npz, out/emb_stats.parquet, log."""
from common_cat import *
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

rng = np.random.default_rng(0)
CAP = 20_000
pool_items, pool_cat, pool_vec = [], [], []
seen = set()
for s in ["val", "test"]:
    E = emb_table(s)
    for kind in ["cand", "hist"]:
        get = cand if kind == "cand" else hist
        it = np.asarray(get(s, "item_id")).astype(np.int64); c = np.asarray(get(s, "category_id")).astype(np.int64)
        e = np.asarray(get(s, "emb_idx")).astype(np.int64)
        ok = (e >= 0) & (c > 0)
        it, c, e = it[ok], c[ok], e[ok]
        _, first = np.unique(it, return_index=True)
        it, c, e = it[first], c[first], e[first]
        new = ~np.isin(it, np.fromiter(seen, np.int64, len(seen))) if seen else np.ones(len(it), bool)
        it, c, e = it[new], c[new], e[new]
        seen.update(it.tolist())
        pool_items.append(it); pool_cat.append(c); pool_vec.append(np.asarray(E[np.sort(e)])[np.argsort(np.argsort(e))])
        log(s, kind, "new items", len(it))
items = np.concatenate(pool_items); cats = np.concatenate(pool_cat); X = np.concatenate(pool_vec).astype(np.float32)
# top up from train candidates for small categories
cnt = np.bincount(cats, minlength=NC)
need = np.maximum(CAP - cnt, 0); need[0] = 0; need[1] = 0
Etr = emb_table("train")
tr_it = np.asarray(cand("train", "item_id")[:60_000_000]).astype(np.int64)
tr_c = np.asarray(cand("train", "category_id")[:60_000_000]).astype(np.int64)
tr_e = np.asarray(cand("train", "emb_idx")[:60_000_000]).astype(np.int64)
ok = (tr_e >= 0) & (need[tr_c] > 0)
tr_it, tr_c, tr_e = tr_it[ok], tr_c[ok], tr_e[ok]
_, first = np.unique(tr_it, return_index=True)
tr_it, tr_c, tr_e = tr_it[first], tr_c[first], tr_e[first]
new = ~np.isin(tr_it, items)
tr_it, tr_c, tr_e = tr_it[new], tr_c[new], tr_e[new]
keep = []
for c in np.where(need > 0)[0]:
    w = np.where(tr_c == c)[0]
    if len(w):
        keep.append(rng.choice(w, min(len(w), need[c]), replace=False))
keep = np.sort(np.concatenate(keep))
o = np.argsort(tr_e[keep])
vec = np.empty((len(keep), 128), np.int8); vec[o] = np.asarray(Etr[tr_e[keep][o]])
items = np.concatenate([items, tr_it[keep]]); cats = np.concatenate([cats, tr_c[keep]]); X = np.concatenate([X, vec.astype(np.float32)])
log("pool after train top-up", len(items))
# cap per category
sel = []
for c in range(NC):
    w = np.where(cats == c)[0]
    if len(w):
        sel.append(w if len(w) <= CAP else rng.choice(w, CAP, replace=False))
sel = np.sort(np.concatenate(sel))
items, cats, X = items[sel], cats[sel], X[sel]
norm = np.linalg.norm(X, axis=1)
log("embedding norm quantiles", np.quantile(norm, [0, .01, .5, .99, 1]))
X /= np.maximum(norm[:, None], 1e-6)
n_c = np.bincount(cats, minlength=NC).astype(float)
log("items per category in pool", n_c[2:].astype(int).tolist())
S = np.zeros((NC, 128)); np.add.at(S, cats, X)
mean_c = S / np.maximum(n_c[:, None], 1)
Mpair = mean_c @ mean_c.T  # mean pairwise cosine between categories
within = (n_c ** 2 * np.diag(Mpair) - n_c) / np.maximum(n_c * (n_c - 1), 1)
Mpair[np.diag_indices(NC)] = within
cn = mean_c / np.maximum(np.linalg.norm(mean_c, axis=1, keepdims=True), 1e-9)
Ccos = cn @ cn.T
log("global mean pairwise cosine (all pool)", float((X.sum(0) @ X.sum(0) - len(X)) / (len(X) * (len(X) - 1))))
# kNN confusion
QN, PN, K = 1000, 5000, 20
qi, pi = [], []
for c in range(NC):
    w = np.where(cats == c)[0]
    if len(w) < 50:
        continue
    pw = rng.choice(w, min(PN, len(w)), replace=False); pi.append(pw)
    qi.append(rng.choice(pw, min(QN, len(pw)), replace=False))
qi = np.concatenate(qi); pi = np.concatenate(pi)
P = X[pi]; Pc = cats[pi]
conf = np.zeros((NC, NC))
for s0 in range(0, len(qi), 4096):
    q = qi[s0:s0 + 4096]
    sim = X[q] @ P.T
    sim[pi[None, :] == q[:, None]] = -9
    nn = np.argpartition(-sim, K, axis=1)[:, :K]
    np.add.at(conf, (np.repeat(cats[q], K), Pc[nn].ravel()), 1)
confn = conf / np.maximum(conf.sum(1, keepdims=True), 1)
# normalise by the pool composition (each category has min(PN, n) items in the pool)
pool_n = np.bincount(Pc, minlength=NC).astype(float)
log("kNN self-share (purity) per category:", {c: round(confn[c, c], 3) for c in range(NC) if pool_n[c] > 0})
np.savez(OUT / "emb_mats.npz", mean_pair=Mpair, centroid_cos=Ccos, knn_conf=conf, pool_n=pool_n, n_items=n_c, centroid=mean_c)

live = np.where(n_c >= 50)[0]
# similarity for clustering: symmetric kNN affinity normalised by the pool sizes, and normalised mean-pair cosine
A = confn[np.ix_(live, live)] / np.maximum(pool_n[live][None, :] / pool_n[live].sum(), 1e-9)  # lift over pool share
A = (A + A.T) / 2
Mn = Mpair[np.ix_(live, live)] / np.sqrt(np.outer(np.diag(Mpair)[live], np.diag(Mpair)[live]))
for nm, sim in [("centroid_cos", Ccos[np.ix_(live, live)]), ("meanpair_norm", Mn), ("knn_lift_log", np.log1p(A))]:
    D = sim.max() - sim
    np.fill_diagonal(D, 0)
    D = (D + D.T) / 2
    Z = linkage(squareform(D, checks=False), "average")
    for k in [4, 6, 8, 10, 12, 14]:
        lab = fcluster(Z, k, "maxclust")
        groups = {}
        for c, l in zip(live, lab):
            groups.setdefault(l, []).append(int(c))
        log(f"{nm} k={k}:", sorted(groups.values(), key=lambda g: g[0]))
rows = []
for c in live:
    nb = [int(live[j]) for j in np.argsort(-Ccos[c, live]) if live[j] != c][:4]
    knn_nb = [int(j) for j in np.argsort(-confn[c]) if j != c][:4]
    rows.append(dict(code=int(c), pool_items=int(n_c[c]), within_cos=float(Mpair[c, c]), knn_purity=float(confn[c, c]),
                     nn_centroid=str(nb), nn_knn=str(knn_nb), nn_knn_share=str([round(confn[c, j], 3) for j in knn_nb])))
df = pl.DataFrame(rows)
df.write_parquet(OUT / "emb_stats.parquet")
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(250); pl.Config.set_fmt_str_lengths(60)
print(df.sort("within_cos", descending=True))
log("done")
