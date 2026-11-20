"""Cross-fitted linear head re-ranker fitted with an NDCG-weighted pairwise logistic (LambdaRank-style, one step)."""
from wc import *
from scipy.optimize import minimize
df = load(); d = features(df)
off, sz = kit.feed_layout(d)
yF = d["label"].to_numpy(); sF = d["score"].to_numpy().astype(np.float64); rkF = d["rank"].to_numpy()
fo = kit.folds(d)
nb = kit.ndcg_per_feed(sF, yF, off, sz)
disc = 1 / np.log2(np.arange(2, 400))
npos = np.add.reduceat(yF, off)

def build_pairs(K, rows_mask):
    I, J, W = [], [], []
    for fi, (o, m) in enumerate(zip(off, sz)):
        if not rows_mask[o]:
            continue
        idx = o + np.argsort(rkF[o:o + m])[:K]
        yy = yF[idx]
        if yy.sum() == 0 or yy.sum() == len(idx):
            continue
        idcg = disc[:npos[fi]].sum()
        P = np.nonzero(yy)[0]; N = np.nonzero(yy == 0)[0]
        for a in P:
            for b in N:
                I.append(idx[a]); J.append(idx[b]); W.append(abs(disc[a] - disc[b]) / idcg)
    return np.array(I), np.array(J), np.array(W)

def fit(I, J, W, X, l2=1.0):
    Z = X[I] - X[J]; g0 = sF[I] - sF[J]
    def f(th):
        u = th[0] * g0 + Z @ th[1:]
        L = (W * np.logaddexp(0, -u)).sum() + 0.5 * l2 * (th[1:] ** 2).sum()
        p = 1 / (1 + np.exp(u))
        gr = -np.r_[(W * p) @ g0, Z.T @ (W * p)] + np.r_[0, l2 * th[1:]]
        return L, gr
    r = minimize(f, np.r_[1.0, np.zeros(Z.shape[1])], jac=True, method="L-BFGS-B")
    return r.x

sets = {"cat3": ["f3", "lcatnew", "cat_hist_share"], "f1pre": ["f1", "pre2_fe_z", "pre2_new_z"],
        "six": ["f1", "pre2_fe_z", "f3", "lcatnew", "lcatn", "cat_hist_share"]}
for K in [10, 20]:
    for nm, cols in sets.items():
        X = np.nan_to_num(d.select(cols).to_numpy().astype(np.float64)); X = (X - X.mean(0)) / X.std(0)
        new = sF.copy()
        for k in range(5):
            I, J, W = build_pairs(K, fo != k)
            th = fit(I, J, W, X)
            te = (fo == k) & (rkF <= K)
            new[te] = sF[te] + X[te] @ th[1:] / th[0]
        nn = kit.ndcg_per_feed(new, yF, off, sz); dd = nn - nb
        print("K=%d %-6s oof %.5f se %.5f z %.2f  last th %s" % (K, nm, dd.mean(), dd.std(ddof=1) / np.sqrt(len(dd)), dd.mean() / (dd.std(ddof=1) / np.sqrt(len(dd))), np.round(th, 3)), flush=True)
