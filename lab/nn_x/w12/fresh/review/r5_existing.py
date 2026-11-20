"""CHECK 5(b): how much of the fresh as-of count do the existing inputs already carry? Label-free (no click label is read
for the fits; the target of the fits is the fresh count itself). Val rows.
"""
import json
import os
import time

import numpy as np
from scipy.stats import rankdata

from rlib import OUT, ROOT, Val

t0 = time.time()
V = Val()
C = np.load(f"{OUT}/my_counts.npy")[:V.N].astype(np.float64)
na = C[:, 0]; la = np.log1p(na); ia = (na >= 1).astype(np.float64)
CVND = ["seq_cv_invd20", "seq_cv_pop", "seq_cv_norm", "it_hist_all_luo", "it_hist_et1_luo", "it_hist_et2_luo", "it_hist_et3_luo", "it_hist_et4_luo",
        "ctx_nd_hcent", "ctx_nd_knn_f1", "ctx_nd_knn_h", "ctx_nd_knn_h_dev", "ctx_nd_knn_u", "ctx_nd_knn_u_dev", "ctx_nd_max", "ctx_nd_top5h", "ctx_nd_top5u",
        "ctx_nd_ucent", "ctx_hn_top1w", "ctx_hn_rrw", "ctx_hn_bestrk", "ctx_hn_margin", "ctx_hn_soft05", "ctx_hn_e1_rrw", "ctx_hs_mutual", "ctx_hs_k128",
        "ctx_hs_e1", "ctx_hs_t10", "pre2_fe_z", "pre2_new_z", "pre2_fe", "pre2_new", "pre2_hmax32"]  # pre2_fe_s = (pre2_fe + 20) / 6: the same information
names = []; cols = []
for n in CVND + ["xp_imp_c50"]:
    p = f"{ROOT}/data/feats/val/{n}.npy"
    if not os.path.exists(p):
        print("MISSING input file", n); continue
    x = np.load(p).astype(np.float64)
    assert len(x) == V.N, n
    names.append(n); cols.append(x)
X = np.stack(cols, 1)
nan = np.isnan(X)
print(f"{len(names)} existing inputs loaded (33 of the 34 train-stage inputs + xp_imp_c50; the 34th is the has:pre2_fe mask); NaN share max {nan.mean(0).max():.3f}")
Xf = np.where(nan, 0.0, X)
Xf = np.sign(Xf) * np.log1p(np.abs(Xf))            # heavy tails
Xf = np.hstack([Xf, nan[:, nan.any(0)].astype(np.float64)])
Xf = (Xf - Xf.mean(0)) / (Xf.std(0) + 1e-9)


def r2(Xm, y):
    A = np.hstack([Xm, np.ones((len(y), 1))])
    beta = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(1 - ((y - A @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def demean(v):
    mu = np.bincount(V.fid, v, V.nf) / V.M
    return v - mu[V.fid]


res = {"n_inputs": len(names)}
res["r2_pooled_log1p_a"] = r2(Xf, la); res["r2_pooled_ind_a"] = r2(Xf, ia)
Xd = np.stack([demean(Xf[:, j]) for j in range(Xf.shape[1])], 1)
res["r2_withinfeed_log1p_a"] = r2(Xd, demean(la)); res["r2_withinfeed_ind_a"] = r2(Xd, demean(ia))
Xb = np.hstack([Xd, demean(V.b)[:, None]])
res["r2_withinfeed_log1p_a_plus_pool"] = r2(Xb, demean(la))
print(f"R^2 of log1p(n_a) on the existing inputs: pooled {res['r2_pooled_log1p_a']:.4f}; within feed (both sides de-meaned per feed) {res['r2_withinfeed_log1p_a']:.4f}; "
      f"with the pool score added {res['r2_withinfeed_log1p_a_plus_pool']:.4f} | indicator n_a >= 1: pooled {res['r2_pooled_ind_a']:.4f}, within feed {res['r2_withinfeed_ind_a']:.4f}")
for sub, m in (("cold", V.cold), ("warm", ~V.cold)):
    res[f"r2_pooled_log1p_a_{sub}"] = r2(Xf[m], la[m])
    print(f"  {sub} rows only (pooled): R^2 {res[f'r2_pooled_log1p_a_{sub}']:.4f}")


def feed_spearman(x, y):
    out = []
    for f in range(V.nf):
        o = V.off[f]; m = V.M[f]
        a = y[o:o + m]
        if a.max() == a.min():
            continue
        b = x[o:o + m]
        if np.nanmax(b) == np.nanmin(b):
            continue
        ra = rankdata(a); rb = rankdata(np.nan_to_num(b, nan=np.nanmin(b) - 1))
        out.append(np.corrcoef(ra, rb)[0, 1])
    out = np.array(out)
    return [float(out.mean()), float(out.std() / np.sqrt(len(out))), int(len(out))]


print("mean within-feed Spearman of n_a with (feeds where both vary):")
res["spearman"] = {}
for n, x in (("pool score b", V.b), ("it_hist_all_luo", X[:, names.index("it_hist_all_luo")]), ("xp_imp_c50", X[:, names.index("xp_imp_c50")]),
             ("seq_cv_pop", X[:, names.index("seq_cv_pop")]), ("pre2_fe", X[:, names.index("pre2_fe")]), ("n_a24", C[:, 1]), ("n_d (whole period)", C[:, 3])):
    s = feed_spearman(x, na); res["spearman"][n] = s
    print(f"  {n:20s} {s[0]:+.4f} +- {s[1]:.4f}  ({s[2]} feeds)")
# mean pool z by count class (label-free): does the pool rank shown items lower?
print("mean pool z by as-of count class (cold | warm):")
res["pool_z_by_count"] = {}
for lab, m in (("0", na == 0), ("1", na == 1), ("2-4", (na >= 2) & (na <= 4)), ("5+", na >= 5)):
    r = [float(V.b[m & V.cold].mean()), int((m & V.cold).sum()), float(V.b[m & ~V.cold].mean()), int((m & ~V.cold).sum())]
    res["pool_z_by_count"][lab] = r
    print(f"  count {lab:3s}: cold {r[0]:+.3f} (n {r[1]}) | warm {r[2]:+.3f} (n {r[3]})")
json.dump(res, open(f"{OUT}/r5_existing.json", "w"), indent=1)
print(f"DONE {time.time() - t0:.0f}s")
