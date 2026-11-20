"""(3b) inspect the flagged dims: bimodal (BC > 0.555 / 2 modes), lowest R^2 from the other dims, highest kurtosis.
For each: modes, eta^2(category), the single category that explains it best (r^2 of a one-vs-rest indicator), attribute correlations,
and whether the bimodality survives within categories."""
import json, numpy as np
from scipy import stats

OUT = "<repo>/nn_x/emb/structure/"
z = np.load(OUT + "cache/train_sample.npz")
Q = z["E"].astype(np.float64); cat = z["cat"].astype(np.int64)
uc = np.unique(cat)
sk = stats.skew(Q, 0); ku = stats.kurtosis(Q, 0); bc = (sk ** 2 + 1) / (ku + 3)
Xc = Q - Q.mean(0); Cv = np.cov(Q.T); P = np.linalg.inv(Cv); r2o = 1 - 1 / (np.diag(P) * np.diag(Cv))
flag = sorted(set(np.nonzero(bc > 0.555)[0].tolist()) | set(np.argsort(r2o)[:5].tolist()) | set(np.argsort(-ku)[:5].tolist()))
attrs = {"f1": z["f1"], "f2": z["f2"], "f3": z["f3"], "f4": z["f4"], "f5": z["f5"], "log_n_cand": np.log1p(z["n_cand"]),
         "log_age_h": np.log1p((z["t_feed"] - z["t_post"]) / 3600.0)}
res = {"flagged_dims": flag, "bc_gt_0555": np.nonzero(bc > 0.555)[0].tolist()}
cnt = np.bincount(cat, minlength=cat.max() + 1)
for d in flag:
    x = Q[:, d]
    h = np.bincount(x.astype(np.int64) + 128, minlength=256).astype(float)
    g = np.exp(-0.5 * (np.arange(-12, 13) / 4.0) ** 2); g /= g.sum(); s = np.convolve(h, g, "same")
    pk = [(i - 128, round(s[i] / s.max(), 3)) for i in range(1, 255) if s[i] > s[i - 1] and s[i] >= s[i + 1] and s[i] > 0.05 * s.max()]
    cm = np.bincount(cat, weights=x, minlength=cat.max() + 1) / np.maximum(cnt, 1)
    eta2 = float((cnt[uc] * (cm[uc] - x.mean()) ** 2).sum() / (x.var() * len(x)))
    best = []
    for c in uc:
        ind = (cat == c).astype(float)
        best.append((float(np.corrcoef(ind, x)[0, 1] ** 2), int(c)))
    best.sort(reverse=True)
    # bimodality inside the largest categories
    within = {}
    for c in [int(c) for c in uc[np.argsort(-cnt[uc])][:3]] + [best[0][1]]:
        xx = x[cat == c]
        within[c] = round(float((stats.skew(xx) ** 2 + 1) / (stats.kurtosis(xx) + 3)), 3)
    ac = {k: round(float(stats.spearmanr(x[np.isfinite(v)], np.asarray(v)[np.isfinite(v)])[0]), 3) for k, v in attrs.items()}
    # mean value of the dim in the two modes: which categories populate the minor mode
    res[int(d)] = dict(mean=float(x.mean()), sd=float(x.std()), skew=float(sk[d]), exkurt=float(ku[d]), bc=float(bc[d]), r2_from_others=float(r2o[d]), modes=pk,
                       eta2_cat=eta2, top_single_cat_r2=best[:3], bc_within_top_cats=within, spearman_attrs=ac,
                       cat_means_min_max=[float(cm[uc].min()), float(cm[uc].max())])
    print(d, res[int(d)], flush=True)
json.dump(res, open(OUT + "s06_special_dims.json", "w"), indent=1)
