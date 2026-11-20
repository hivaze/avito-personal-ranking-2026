"""Leak audit (same method as reports/leak_audit_features.json): single-feature NDCG on val feeds whose user has a
LATER feed (val or test), vs all other val feeds. Also: correlation matrix of KEEP features (val)."""
import sys
sys.path.insert(0, "<repo>/src")
from common import *
K = "ctx_hn_soft05,ctx_hn_rrw,ctx_hn_e1_rrw,ctx_hn_top1w,ctx_hn_margin,ctx_hn_bestrk,ctx_hs_t10,ctx_hs_d7,ctx_hs_k128,ctx_hs_e1,ctx_hs_k256,ctx_hs_e1k64,ctx_hs_k128t10,ctx_hs_mutual,ctx_nd_knn_u_dev,ctx_nd_ucent,ctx_nd_knn_h_dev,ctx_nd_knn_h,ctx_nd_max,ctx_nd_hcent,ctx_nd_knn_f1,ctx_nd_knn_u,ctx_nd_top5u,ctx_nd_top5h".split(",")
fv, Mv, offv = feed_meta("val"); ft, _, _ = feed_meta("test")
uv, tv = fv["user_id"].to_numpy(), fv["timestamp"].to_numpy()
allu = np.r_[uv, ft["user_id"].to_numpy()]; allt = np.r_[tv, ft["timestamp"].to_numpy()]
last = {}
for u, t in zip(allu, allt):
    if u != 0: last[u] = max(last.get(u, -1), t)
later = np.array([u != 0 and last[u] > t for u, t in zip(uv, tv)])
print("val feeds with a later same-user feed:", later.sum())
y = np.load(FLAT / "val/cand_target.npy").astype(np.float64)
def sub(mask):
    fs = np.where(mask)[0]; rows = np.concatenate([np.arange(offv[i], offv[i] + Mv[i]) for i in fs]); m = Mv[fs]
    return rows, m, np.r_[0, np.cumsum(m)[:-1]]
RL, ML, OL = sub(later); RO, MO, OO = sub(~later)
rng = np.random.default_rng(0)
def nd1(x, rows, m, o):
    xx = x[rows]; s = np.where(np.isnan(xx), -1e9, xx) + rng.random(len(xx)) * 1e-9
    return ndcg(s, y[rows], o, m)
print(f"{'feature':20s} later  other  excess")
for k in K + ["maxsim32", "cos_u", "f1"]:
    x = np.load(FEATS / "val" / f"{k}.npy").astype(np.float64)
    if k == "ctx_hn_bestrk": x = -x
    a, b = nd1(x, RL, ML, OL), nd1(x, RO, MO, OO)
    print(f"{k:20s} {a:.3f}  {b:.3f}  {a-b:+.3f}")
X = load_feats("val", K); g = ~np.isnan(X).any(1)
from scipy.stats import spearmanr
C = spearmanr(X[g][::5]).correlation
print("pairs with |spearman| > 0.95:")
for i in range(len(K)):
    for j in range(i + 1, len(K)):
        if abs(C[i, j]) > 0.95: print(f"  {K[i]} ~ {K[j]}: {C[i, j]:.3f}")
