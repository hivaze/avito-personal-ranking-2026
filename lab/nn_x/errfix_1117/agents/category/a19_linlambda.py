"""Pooled bound: linear LambdaRank correction on ~35 category features, ridge-penalised, cross-fitted.
new = score + X w, w fitted on 4 folds by gradient descent on the LambdaRank surrogate; NDCG on the 5th."""
import numpy as np
import cu

C = cu.Ctx()
df = C.df
c = lambda n: df[n].to_numpy().astype(np.float64)
cat = df["cat"].to_numpy()
vert = np.array([cu.VERT[x] for x in cat.tolist()])
cni = c("cat_newest_idx")
ur = c("cat_user_rank")
rk = c("rank")


def zs(x):
    x = np.asarray(x, dtype=np.float64)
    m, sd = np.nanmean(x), np.nanstd(x)
    return np.nan_to_num((x - m) / (sd if sd > 0 else 1))


cols = {
    "same_newest": c("same_cat_newest"), "absent": ur == 0, "log_cni": np.log1p(np.where(cni < 0, 300, cni)),
    "log_views": np.log1p(c("cat_views")), "log_actions": np.log1p(c("cat_actions")), "hshare": c("cat_hist_share"),
    "lcat_n": np.log(c("cat_n_in_feed")), "share": c("cat_n_in_feed") / c("M"), "lcat_rank": np.log(c("cat_rank_in_feed")),
    "cat_rank1": c("cat_rank_in_feed") == 1, "f3": c("f3"), "ur_eff": np.where(ur == 0, 30, ur),
    "dft": c("s_ft") - c("s_to"), "same_head": c("same_cat_newest") * (rk <= 5),
}
for v in ("Services", "Jobs", "RealEstate", "Personal", "Transport", "Home", "Electronics", "Hobby", "Animals"):
    m = (vert == v).astype(float)
    cols[f"v:{v}"] = m
    cols[f"v:{v} x absent"] = m * (ur == 0)
    cols[f"v:{v} x same"] = m * c("same_cat_newest")
names = list(cols)
X = np.column_stack([zs(cols[n]) for n in names])
print(len(names), "features")


from scipy.optimize import minimize

pairs = cu.lambda_pairs(C, temp=1.0)
I0, J0, L0, F0 = pairs
_, _, dn_only, _ = cu.lambda_pairs(C, temp=0.0)   # temp 0 -> sigmoid 0.5: plain |dNDCG| * 0.5
Xs = np.column_stack([C.s, X]).astype(np.float32)
DX = (Xs[I0] - Xs[J0]).astype(np.float32)
DS = (C.s[I0] - C.s[J0])
WP = dn_only * 2.0
fold_feed = C.ffold[F0]


def fit(train_pairs, lam):
    dx, ds, wp = DX[train_pairs], DS[train_pairs], WP[train_pairs]
    nfe = np.unique(F0[train_pairs]).size

    def f(w):
        m = ds + dx @ w
        l = np.logaddexp(0, -m)
        sg = -1 / (1 + np.exp(m))
        loss = (wp * l).sum() / nfe + 0.5 * lam * (w[1:] ** 2).sum()
        g = (dx * (wp * sg)[:, None]).sum(0) / nfe
        g[1:] += lam * w[1:]
        return loss, g.astype(np.float64)
    r = minimize(f, np.zeros(DX.shape[1]), jac=True, method="L-BFGS-B")
    return r.x


for lam in (0.001, 0.01, 0.1, 1.0):
    oof = np.empty(len(C.sz))
    for k in range(5):
        w = fit(fold_feed != k, lam)
        d = C.d(Xs.astype(np.float64) @ w + C.s)
        oof[C.ffold == k] = d[C.ffold == k]
    m, se, z = cu.delta_stats(oof)
    wall = fit(np.ones(len(F0), bool), lam)
    ins = C.d(Xs.astype(np.float64) @ wall + C.s).mean()
    top = np.argsort(-np.abs(wall[1:]))[:8]
    print(f"lam {lam}: crossfit {m:+.5f} se {se:.5f} z {z:+.2f} | in-sample {ins:+.5f} | score coef {wall[0]:+.3f} | top w: " +
          ", ".join(f"{names[i]}={wall[1+i]:+.3f}" for i in top), flush=True)
