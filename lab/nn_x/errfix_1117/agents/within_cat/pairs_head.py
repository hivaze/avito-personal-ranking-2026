"""Head pairwise analysis: pairs (i above j) within the top-K with exactly one positive; logistic on P(j is the positive)
with offset from the score gap; LambdaRank-style |dNDCG| weights; feed-clustered SEs. Option: same-category pairs only."""
from wc import *
from scipy.optimize import minimize
K = int(sys.argv[1]); same = sys.argv[2] == "same"
df = load(); d = features(df)
F = ["f1", "f4", "f5", "ucos", "hmax", "h1", "h5", "h10", "h32", "lhidx", "xp_imp_c50", "lpop", "lpop_et2", "lpop_et4", "ldup", "cand_maxcos",
     "pre2_fe_z", "pre2_new_z", "lcv", "sdiff", "n_hdup95", "f2", "f3", "lcatn", "cat_hist_share", "lcatnew", "c_contact", "c_fav", "c_vfeed", "c_vsearch", "c_order", "c_lead"]
h = d.filter(pl.col("rank") <= K).sort(["fidx", "rank"])
fid = h["fidx"].to_numpy(); rk = h["rank"].to_numpy(); y = h["label"].to_numpy(); s = h["score"].to_numpy().astype(np.float64)
cat = h["cat"].to_numpy(); M = h["M"].to_numpy()
X = np.nan_to_num(h.select(F).to_numpy().astype(np.float64))
off, sz = kit.feed_layout(h)
I, J, W, G = [], [], [], []
# ideal DCG of each feed computed from the full feed
offF, szF = kit.feed_layout(d); yF = d["label"].to_numpy()
npos = dict(zip(d["fidx"].to_numpy()[offF], np.add.reduceat(yF, offF)))
disc = 1 / np.log2(np.arange(2, 400))
for o, m in zip(off, sz):
    yy = y[o:o + m]
    if yy.sum() == 0 or yy.sum() == m:
        continue
    idcg = disc[:npos[fid[o]]].sum()
    for a in range(m):
        for b in range(a + 1, m):
            if yy[a] == yy[b]:
                continue
            if same and cat[o + a] != cat[o + b]:
                continue
            I.append(o + a); J.append(o + b); W.append(abs(disc[a] - disc[b]) / idcg); G.append(fid[o])
I, J, W, G = map(np.array, (I, J, W, G))
t = (y[J] == 1).astype(float)           # 1 = the lower-ranked item is the positive
gap = s[I] - s[J]
print("K", K, "same" if same else "any", "pairs", len(I), "lower-is-pos rate %.3f" % t.mean(), "feeds", len(np.unique(G)))
DX = X[J] - X[I]
sdX = DX.std(0); sdX[sdX == 0] = 1
DX = DX / sdX
def fit(cols):
    Z = np.c_[-gap, DX[:, cols]] if len(cols) else -gap[:, None]
    def f(th):
        u = Z @ th
        p = 1 / (1 + np.exp(-u))
        L = -(W * (t * np.log(p + 1e-12) + (1 - t) * np.log(1 - p + 1e-12))).sum()
        g = -(Z * (W * (t - p))[:, None]).sum(0)
        return L, g
    r = minimize(f, np.r_[1.0, np.zeros(Z.shape[1] - 1)], jac=True, method="L-BFGS-B")
    th = r.x; u = Z @ th; p = 1 / (1 + np.exp(-u))
    Hm = (Z * (W * p * (1 - p))[:, None]).T @ Z
    sc = Z * (W * (t - p))[:, None]
    ug, gi = np.unique(G, return_inverse=True)
    S = np.zeros((len(ug), Z.shape[1])); np.add.at(S, gi, sc)
    Hi = np.linalg.pinv(Hm); cov = Hi @ (S.T @ S) @ Hi
    return th, np.sqrt(np.diag(cov))
th, se = fit([])
print("gap slope %.3f" % th[0])
rows = []
for k, c in enumerate(F):
    if DX[:, k].std() < 1e-9: continue
    th, se = fit([k])
    rows.append((c, th[1], th[1] / se[1]))
rows.sort(key=lambda r: -abs(r[2]))
for r in rows: print("%-15s %+.4f z=%+.2f" % r)
