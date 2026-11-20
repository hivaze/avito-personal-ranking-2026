"""(a) Expected-NDCG-optimal decoding. Full-list NDCG = sum_i rel_i d(pos_i) / IDCG(K): the optimal order sorts by
w_i = E[rel_i c(K)], c(K) = 1/IDCG(K) (rearrangement inequality, exact). A per-feed K model is order-invariant; only
the item-conditional E[c(K) | rel_i = 1, x_i] can re-order. Two readings of what the score estimates:
  g1: s ~ log P(rel_i)        -> key = s + beta * log E[c(K) | rel_i=1, x]
  g2: s ~ log E[rel_i / K]    -> key = s + beta * log E_{1/K}[K c(K) | rel_i=1, x]  (softmax-CE target)
g is a LightGBM regression on the POSITIVES of one user half (label-free features), applied to the other half.
beta = 1 is the principled (declared) value; other betas are information only."""
import ph, numpy as np, polars as pl, lightgbm as lgb, json, sys
D = ph.Data("val")
F = pl.read_parquet("../out/valfeat.parquet")
pr = np.load("../out/pair_val.npz")
IDCG = np.cumsum(1 / np.log2(np.arange(2, 400)))
c = 1.0 / IDCG[np.maximum(D.K.astype(int), 1) - 1]          # per feed
fid = D.fid; M = D.M
# softmax of s per feed
s = D.s
mx = np.maximum.reduceat(s, D.off); e = np.exp(s - mx[fid]); Z = np.bincount(fid, e, D.n); q = e / Z[fid]
ent = -np.bincount(fid, q * np.log(q + 1e-12), D.n)
X = np.column_stack([
    s, np.log(q), F["rank"].to_numpy() / M[fid], M[fid], F["H"].to_numpy(), ent[fid], np.maximum.reduceat(q, D.off)[fid],
    F["cat_n_in_feed"].to_numpy() / M[fid], F["cat_n_in_feed"].to_numpy(), F["cat_rank_in_feed"].to_numpy(), F["n_feed_cats"].to_numpy(),
    F["cand_maxcos"].to_numpy(), F["n_cand_dup95"].to_numpy(), F["cat_hist_share"].to_numpy(), F["h1"].to_numpy(), F["hmax"].to_numpy(),
    pr["simmass"], pr["simmass_hi"], pr["dupmass"], pr["catmass"], pr["top5cos"], F["ucos"].to_numpy(),
]).astype(np.float32)
names = "s logq relrank M H ent qmax catshare catn catrank nfeedcats maxcos ndup cathist h1 hmax simmass simmass_hi dupmass catmass top5cos ucos".split()
pos = D.y > 0
Kr = D.K[fid]
t1 = np.log(c[fid]); t2 = np.log(Kr * c[fid]); w2 = 1.0 / Kr
params = dict(objective="regression", learning_rate=0.03, num_leaves=15, min_data_in_leaf=400, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1, num_threads=12, seed=7)
res = {}
G = {}
for nm, t, w in (("g1", t1, None), ("g2", t2, w2)):
    g = np.zeros(len(s))
    for h in (0, 1):
        tr = pos & (D.halfr == h); ap = D.halfr != h
        ds = lgb.Dataset(X[tr], t[tr], weight=None if w is None else w[tr], feature_name=names)
        b = lgb.train(params, ds, num_boost_round=300)
        g[ap] = b.predict(X[ap])
        imp = sorted(zip(b.feature_importance("gain"), names), reverse=True)[:8]
        print(nm, "half", h, "top gain", [(n, round(v / sum(b.feature_importance("gain")), 3)) for v, n in imp])
    G[nm] = g
    # how much does g vary within feeds? (only within-feed variation can reorder)
    gm = np.bincount(fid, g, D.n) / M; wsd = np.sqrt(np.bincount(fid, (g - gm[fid]) ** 2, D.n) / M)
    print(nm, "within-feed sd of g: mean", wsd.mean().round(4), " total sd", g.std().round(4))
    for beta in (0.25, 0.5, 1.0, 1.5, 2.0, 3.0):
        r = D.report(s + beta * g, f"A {nm} beta {beta}")
        res[f"{nm}_b{beta}"] = r
np.savez("../out/expA_g.npz", **G)
json.dump({k: {kk: vv for kk, vv in v.items() if kk != 'line'} for k, v in res.items()}, open("../out/expA.json", "w"), indent=1)
