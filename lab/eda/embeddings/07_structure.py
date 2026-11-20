"""Structure of item embeddings: category encoding (linear probe, kNN purity, ANOVA, KMeans NMI, PCA),
relation to f1..f5 (item-level vs user-item; ridge R2 from embedding), item 'quality' direction."""
import numpy as np
import polars as pl
import faiss
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import normalized_mutual_info_score
from common import feeds, arr, item_table, gather_emb
from emb_features import l2n

rng = np.random.default_rng(0)
faiss.omp_set_num_threads(40)

# ---------- unique val candidate items with category ----------
ci = np.asarray(arr('val', 'cand_item_id')); ce = np.asarray(arr('val', 'cand_emb_idx')); cc = np.asarray(arr('val', 'cand_category_id'))
df = pl.DataFrame({'item': ci, 'idx': ce, 'cat': cc}).filter(pl.col('idx') >= 0)
# is category constant per item?
g = df.group_by('item').agg(pl.col('cat').n_unique().alias('nc'))
print(f'val items with >1 category across feeds: {(g["nc"] > 1).mean():.5f}')
items = df.unique(subset='item', keep='first')
sub = items.sample(min(len(items), 300000), seed=0)
X, _ = l2n(gather_emb(item_table('val'), sub['idx'].to_numpy()))
y = sub['cat'].to_numpy()
n = len(y); tr = np.arange(n) < int(n * 0.7); te = ~tr
maj = np.bincount(y[tr]).argmax()
print(f'items {n}, categories {len(np.unique(y))}, majority-class acc {np.mean(y[te] == maj):.4f}')

# linear probe
lr = LogisticRegression(max_iter=300, C=10.0, n_jobs=40)
lr.fit(X[tr][:150000], y[tr][:150000])
print(f'linear probe (logreg) category accuracy: {lr.score(X[te], y[te]):.4f}')

# kNN purity
index = faiss.IndexFlatIP(128); index.add(X[tr])
D_, I_ = index.search(X[te][:50000], 11)
nb_y = y[tr][I_[:, :10]]
print(f'kNN(10) same-category rate {np.mean(nb_y == y[te][:50000, None]):.4f}; kNN(1) category acc {np.mean(nb_y[:, 0] == y[te][:50000]):.4f}; '
      f'chance (sum p^2) {np.sum((np.bincount(y) / n) ** 2):.4f}; mean top1 cos {D_[:, 0].mean():.3f}')

# ANOVA: fraction of total variance explained by category
mu = X.mean(0); tot = ((X - mu) ** 2).sum()
between = 0
for c in np.unique(y):
    m = y == c
    between += m.sum() * ((X[m].mean(0) - mu) ** 2).sum()
print(f'between-category variance share (R2 of category one-hot): {between / tot:.4f}')

# PCA and per-PC eta^2
Xc = X - mu
U_, S_, Vt = np.linalg.svd(Xc[:100000], full_matrices=False)
ev = S_ ** 2 / (S_ ** 2).sum()
Z = Xc @ Vt[:10].T
print('PCA explained var top10:', np.round(ev[:10], 3))
eta = []
for k in range(10):
    z = Z[:, k]; zm = z.mean(); tv = ((z - zm) ** 2).sum()
    bv = sum((y == c).sum() * (z[y == c].mean() - zm) ** 2 for c in np.unique(y))
    eta.append(bv / tv)
print('eta^2(category) for PC1..10:', np.round(eta, 3))

# KMeans vs category
for k in [53, 256]:
    km = faiss.Kmeans(128, k, niter=25, seed=1, spherical=True, verbose=False)
    km.train(X[:200000])
    _, lab = km.index.search(X, 1)
    lab = lab[:, 0]
    # purity
    pur = sum(np.bincount(y[lab == c]).max() for c in np.unique(lab)) / n
    print(f'KMeans k={k}: NMI(cluster,category)={normalized_mutual_info_score(y, lab):.3f}, purity={pur:.3f}')

# ---------- f1..f5: item-level or user-item? ----------
print('\n== f1..f5: share of variance explained by item identity (train subsample, items with >=2 occurrences)')
C = '<repo>/eda/embeddings/cache'
fidx = np.load(f'{C}/train_fidx.npy')
ft = feeds('train')[fidx]
M = ft['M'].to_numpy(); off = ft['cand_off'].to_numpy()
cidx = np.repeat(off - np.cumsum(np.r_[0, M[:-1]]), M) + np.arange(M.sum())
tdf = pl.DataFrame({'item': np.asarray(arr('train', 'cand_item_id')[cidx]),
                    'user': np.repeat(ft['user_id'].to_numpy(), M),
                    'day': np.repeat((ft['timestamp'].to_numpy() - ft['timestamp'].min()) // 86400, M),
                    **{f'f{k}': np.asarray(arr('train', f'cand_f{k}')[cidx]) for k in range(1, 6)}})
cnt = tdf.group_by('item').agg(pl.len().alias('n'))
rep = tdf.join(cnt.filter(pl.col('n') >= 2), on='item')
print(f'  rows with repeated items: {len(rep)} / {len(tdf)}')
for k in range(1, 6):
    c = f'f{k}'
    tv = rep[c].var()
    wv = rep.with_columns((pl.col(c) - pl.col(c).mean().over('item')).alias('r'))['r'].var()
    # within item & day
    wv2 = rep.with_columns((pl.col(c) - pl.col(c).mean().over(['item', 'day'])).alias('r'))
    wv2 = wv2.filter(pl.len().over(['item', 'day']) >= 2)['r'].var()
    print(f'  {c}: total var {tv:.4f}, within-item var share {wv / tv:.3f} (item explains {1 - wv / tv:.3f}); within item-day share {wv2 / tv:.3f}')

# ridge from embedding -> f_k on val candidates (item-level split by item)
print('\n== ridge R2 predicting f1..f5 from item embedding (val candidates, split by item)')
vdf = pl.DataFrame({'item': ci, 'idx': ce, **{f'f{k}': np.asarray(arr('val', f'cand_f{k}')) for k in range(1, 6)},
                    't': np.asarray(arr('val', 'cand_target')).astype(np.float32)}).filter(pl.col('idx') >= 0)
iv = vdf.group_by('item').agg(pl.col('idx').first(), *[pl.col(f'f{k}').mean() for k in range(1, 6)], pl.col('t').mean(), pl.len().alias('n'))
iv = iv.sample(min(len(iv), 300000), seed=1)
Xi, _ = l2n(gather_emb(item_table('val'), iv['idx'].to_numpy()))
cat_map = dict(zip(items['item'].to_list(), items['cat'].to_list()))
cat_i = np.array([cat_map.get(i, 0) for i in iv['item'].to_list()])
Oh = np.eye(64, dtype=np.float32)[cat_i]
n = len(Xi); tr = np.arange(n) < int(0.7 * n); te = ~tr
for k in range(1, 6):
    yk = iv[f'f{k}'].to_numpy()
    r = Ridge(alpha=1.0).fit(Xi[tr], yk[tr]); r2 = r.score(Xi[te], yk[te])
    r2c = Ridge(alpha=1.0).fit(Oh[tr], yk[tr]).score(Oh[te], yk[te])
    corr = np.corrcoef(Xi[te] @ r.coef_, yk[te])[0, 1]
    print(f'  f{k}: R2(emb)={r2:.3f}  R2(category one-hot)={r2c:.3f}  corr={corr:.3f}')
