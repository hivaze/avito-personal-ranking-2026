"""Across-category correction: conditional logit over the categories of a feed.

P(positive in category c | feed) ∝ exp(b1*log m_c + sum_k b_k x_ck), m_c = model mass of category c (softmax of
a0*score summed over its items). Rule: score' = score + ((b1-1)*log m_c + sum b_k x_ck) / a0 applied to every
item in category c (within-category order unchanged). Parameters fitted on 4 folds, NDCG on the 5th.
"""
import numpy as np
import polars as pl
from scipy.optimize import minimize
import cu

C = cu.Ctx()
df = C.df
s, y, fid = C.s, C.y.astype(float), C.fid
A0 = 0.98
z = A0 * s
mx = np.maximum.reduceat(z, C.off)
e = np.exp(z - mx[fid])
p = e / np.bincount(fid, weights=e)[fid]
key = df.select((pl.col("fidx") * 100 + pl.col("cat")).alias("k"))["k"].to_numpy()
uk, inv = np.unique(key, return_inverse=True)   # (feed, cat) units, sorted by feed then cat -> contiguous by feed
ncu = len(uk)
mass = np.bincount(inv, weights=p)
ypos = np.bincount(inv, weights=y)
ufeed = np.zeros(ncu, dtype=np.int64)
ufeed[inv] = fid
first = np.zeros(ncu, dtype=np.int64)
first[inv[::-1]] = np.arange(len(inv))[::-1]
c = lambda n: df[n].to_numpy().astype(np.float64)[first]
cat_u = df["cat"].to_numpy()[first]
ur = c("cat_user_rank")
cni = c("cat_newest_idx")
M = c("M")
nfc = c("cat_n_in_feed")
feats = {
    "logm": np.log(mass + 1e-12),
    "logshare": np.log(nfc / M),
    "absent": (ur == 0).astype(float),
    "newest": (cni == 0).astype(float),
    "log_cni": np.log1p(np.where(cni < 0, 300, cni)),
    "log_views": np.log1p(c("cat_views")),
    "log_actions": np.log1p(c("cat_actions")),
    "hshare": c("cat_hist_share"),
    "services": (cat_u == 50).astype(float),
}
uoff = np.r_[0, np.nonzero(ufeed[1:] != ufeed[:-1])[0] + 1]
ufold = df["fold"].to_numpy()[first]


def fit(names, mask):
    X = np.column_stack([feats[n] for n in names])
    def f(b):
        zz = X @ b
        m_ = np.maximum.reduceat(zz, uoff)
        ee = np.exp(zz - m_[ufeed])
        den = np.bincount(ufeed, weights=ee)
        P = ee / den[ufeed]
        w = ypos * mask
        ll = -(w * np.log(P + 1e-300)).sum()
        npos_f = np.bincount(ufeed, weights=w)
        g = -((w - npos_f[ufeed] * P) @ X)
        return ll, g
    b0 = np.zeros(X.shape[1]); b0[0] = 1.0
    r = minimize(f, b0, jac=True, method="L-BFGS-B")
    return r.x


def make_scores(names, b):
    X = np.column_stack([feats[n] for n in names])
    delta_u = X @ b - feats["logm"]          # new category log-mass minus old
    return s + delta_u[inv] / A0


sets = [["logm"], ["logm", "logshare"], ["logm", "absent", "newest", "log_cni"], ["logm", "logshare", "absent", "newest", "log_cni"],
        ["logm", "log_views", "log_actions", "hshare"], ["logm", "services"], ["logm", "logshare", "services", "absent", "newest"]]
for names in sets:
    ball = fit(names, np.ones(ncu))
    oof = np.empty(len(C.sz))
    for k in range(5):
        b = fit(names, (ufold != k).astype(float))
        d = C.d(make_scores(names, b))
        oof[C.ffold == k] = d[C.ffold == k]
    m, se, zz = cu.delta_stats(oof)
    ins = C.d(make_scores(names, ball)).mean()
    C.n_tried += 1
    print(f"{names}: coefs {np.round(ball, 3)} | crossfit {m:+.5f} se {se:.5f} z {zz:+.2f} | in-sample {ins:+.5f}")
print("variants", C.n_tried)
