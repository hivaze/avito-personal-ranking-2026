"""Controls for the PD LR statistic (val): (1) x = 20 quantile bins of g itself on top of the hinge calibration;
(2) x = within-feed permutation of a real input (3 perms each; null LR should be ~df); (3) LR of real inputs with
20 g-quantile dummies in BOTH the reduced and full model (removes any residual g-calibration misfit correlated with x);
(4) the label-weighted share of each input's residual that sits in feeds' top-10 by G3 (where NDCG weight is)."""
import sys, time
import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2 as CHI2
sys.path.insert(0, "/tmp/scratchpad/dpa/nonlinear")
from nlc import *  # noqa

t0 = time.time()
d = load("val")
g = d["g"]
beta, basis, p, npos = fit_calib(d, g)
fid = d["fid"]; y = d["y"].astype(np.float64)
offl = np.log(np.maximum(p, 1e-300))


def cl_fit(idx_list, sizes, lam=1e-3):
    if not idx_list:
        l = offl; mx = np.maximum.reduceat(l, d["off"]); Z = np.bincount(fid, np.exp(l - mx[fid]), d["nf"])
        return (y * l).sum() - (npos * (np.log(Z) + mx)).sum()
    starts = np.cumsum([0] + sizes[:-1]); tot = sum(sizes)

    def f(th):
        l = offl.copy()
        for j, ix in enumerate(idx_list):
            l += th[starts[j]:starts[j] + sizes[j]][ix]
        mx = np.maximum.reduceat(l, d["off"]); e = np.exp(l - mx[fid]); Z = np.bincount(fid, e, d["nf"]); q = e / Z[fid]
        ll = (y * l).sum() - (npos * (np.log(Z) + mx)).sum()
        r = npos[fid] * q - y
        return -ll + lam * (th ** 2).sum(), np.concatenate([np.bincount(ix, r, sizes[j]) for j, ix in enumerate(idx_list)]) + 2 * lam * th
    return -minimize(f, np.zeros(tot), jac=True, method="L-BFGS-B", options={"maxiter": 500}).fun


def bins_of(x, nb=10):
    q = np.unique(np.quantile(x, np.linspace(0, 1, nb + 1)[1:-1]))
    return np.unique(np.searchsorted(q, x, side="right"), return_inverse=True)[1]


LL0 = cl_fit([], [])
gb = bins_of(g, 20); ngb = gb.max() + 1
LLg = cl_fit([gb], [ngb])
print(f"(1) g itself, 20 bins on top of hinge calibration: LR {2*(LLg-LL0):.1f} df {ngb-1}")
rng = np.random.default_rng(0)
off = d["off"]; M = d["M"]


def perm_within(x):
    o = np.lexsort((rng.random(len(x)), fid))
    return x[o]


feats = {"f3": d["f3"], "fcs": d["d_fcs"], "ctx_hs_k128": d["xt_ctx_hs_k128"], "smax": d["d_smax"], "cat_last32": d["d_cat_last32"],
         "ctx_hn_margin": d["xt_ctx_hn_margin"], "f4": d["f4"], "f5": d["f5"], "pre2_fe_z": d["xt_pre2_fe_z"], "seq_cv_pop": d["xt_seq_cv_pop"]}
print("(2) within-feed permutation nulls (LR main, df):")
for n in ["f3", "fcs", "ctx_hs_k128", "smax"]:
    out = []
    for k in range(3):
        b = bins_of(perm_within(np.asarray(feats[n], np.float64))); nb = b.max() + 1
        out.append(2 * (cl_fit([b], [nb]) - LL0))
    print(f"   {n:14s} perm LR {np.round(out,1).tolist()} df {nb-1}")
print("(3) LR main with 20 g-bins in both models (df):")
# (4) top-10 share: rows in a feed's G3 top-10
o = np.lexsort((-g, fid)); st = np.r_[True, fid[o][1:] != fid[o][:-1]]
rk = np.empty(len(g)); rk[o] = np.arange(len(o)) - np.flatnonzero(st)[np.cumsum(st) - 1]
top10 = rk < 10
E = npos[fid] * p
for n, x in feats.items():
    b = bins_of(np.asarray(x, np.float64)); nb = b.max() + 1
    lr0 = 2 * (cl_fit([b], [nb]) - LL0)
    lrg = 2 * (cl_fit([gb, b], [ngb, nb]) - LLg)
    # residual mass in top-10 vs rest: sum |A-E| per bin within top10 / overall
    At = np.bincount(b[top10], y[top10], nb); Et = np.bincount(b[top10], E[top10], nb)
    Ar = np.bincount(b[~top10], y[~top10], nb); Er = np.bincount(b[~top10], E[~top10], nb)
    print(f"   {n:14s} LR {lr0:6.1f} -> with g-bins {lrg:6.1f} (df {nb-1}) | top-10 act/exp by bin {np.round(At/np.maximum(Et,1e-9),2).tolist()} | rest {np.round(Ar/np.maximum(Er,1e-9),2).tolist()}")
print(f"done {time.time()-t0:.0f}s")
