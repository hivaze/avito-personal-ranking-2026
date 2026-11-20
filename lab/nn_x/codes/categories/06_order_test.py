"""06: is the code numbering ordered by something vertical-coherent? (tests the 'code = rank of Avito category id' claim
without using any names). Statistic: number of adjacent code pairs (c, c+1) that fall in the same DATA-derived cluster
(embedding kNN / history transitions, k=12, from 05). Null: random permutations of the code labels (100,000 draws)."""
from common_cat import *
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
em = np.load(OUT / "emb_mats.npz"); hm = np.load(OUT / "hist_mats.npz")
codes = np.arange(2, 54)
def cl(S, k=12):
    D = S.max() - S; np.fill_diagonal(D, 0); D = (D + D.T) / 2
    return fcluster(linkage(squareform(D, checks=False), "average"), k, "maxclust")
def lift(T):
    T = T[np.ix_(codes, codes)].astype(float).copy(); np.fill_diagonal(T, 0)
    L = T * T.sum() / np.maximum(T.sum(1, keepdims=True) * T.sum(0, keepdims=True), 1e-9); L = (L + L.T) / 2
    return np.log(np.maximum(L, 1e-3))
K = em["knn_conf"]; Kn = (K / np.maximum(K.sum(1, keepdims=True), 1)) / np.maximum(em["pool_n"][None, :] / em["pool_n"].sum(), 1e-9)
S_knn = np.log1p(Kn[np.ix_(codes, codes)]); S_knn = (S_knn + S_knn.T) / 2
rng = np.random.default_rng(0)
for nm, S in [("emb_centroid", em["centroid_cos"][np.ix_(codes, codes)]), ("emb_knn", S_knn), ("trans_all_TR", lift(hm["trans_all_TR"])), ("trans_all_VT", lift(hm["trans_all_VT"]))]:
    for k in [8, 12]:
        lab = cl(S, k)
        stat = int((lab[1:] == lab[:-1]).sum())
        null = np.array([(p[1:] == p[:-1]).sum() for p in (rng.permutation(lab) for _ in range(100000))])
        log(f"{nm:13s} k={k:2d}: adjacent same-cluster pairs {stat}/51, null mean {null.mean():.2f} sd {null.std():.2f}, p = {max((null >= stat).mean(), 1e-5):.5f}")
# raw similarity of adjacent codes vs random pairs (no clustering)
for nm, S in [("emb_centroid", em["centroid_cos"][np.ix_(codes, codes)]), ("trans_all_TR", lift(hm["trans_all_TR"]))]:
    adj = np.mean([S[i, i + 1] for i in range(len(codes) - 1)])
    iu = np.triu_indices(len(codes), 1); allm = S[iu].mean()
    null = []
    for _ in range(20000):
        p = rng.permutation(len(codes)); null.append(np.mean([S[p[i], p[i + 1]] for i in range(len(codes) - 1)]))
    null = np.array(null)
    log(f"{nm:13s}: mean similarity of adjacent codes {adj:.3f} vs all pairs {allm:.3f}; permutation p = {max((null >= adj).mean(), 5e-5):.5f}")
