"""Monetising the PD finding directly: a nested additive bin-GAM residual in the conditional-logit (listwise) form.
score = B(g) beta + sum_j theta_j[bin_j(x_j)]  (B = hinge basis of z(G3); 10 global quantile bins per input, label-free edges)
fitted by within-feed conditional logit with ridge lam*||theta||^2 on 4 user-grouped folds, applied to the 5th (OOF NDCG vs
G3). Holdout: fit on all val, applied to the holdout (its own z(G3) through the val beta). Sets: singles, an affinity block,
a popularity block, all inputs. Usage: python gam.py <set> <lam>"""
import sys, time
import numpy as np
from scipy.optimize import minimize
sys.path.insert(0, "/tmp/scratchpad/dpa/nonlinear")
from nlc import load, folds5, ndcg, wmean_se, pmix_w, D, fpct  # noqa

t0 = time.time()
setname, lam = sys.argv[1], float(sys.argv[2])
dv = load("val"); dh = load("hold")
KN = (-1.0, 0.0, 1.0, 1.5, 2.0, 2.5, 3.0)
basis = lambda g: np.stack([g] + [np.maximum(g - k, 0) for k in KN], 1)
AFF = ["f3", "d_fcs", "d_cshare", "d_cat_last32", "xt_ctx_hs_k128", "d_smax", "xt_ctx_hn_margin", "xt_ctx_nd_knn_u", "d_lse", "f1"]
POP = ["xt_seq_cv_pop", "xt_it_hist_all_luo", "xt_it_hist_et1_luo", "f5", "f4"]
ALL = (["f1", "f2", "f3", "f4", "f5"] + [k for k in dv if k.startswith("d_") and k not in ("d_logM", "d_logH", "d_cnorm", "d_cmiss")]
       + [k for k in dv if k.startswith("xt_") and k != "xt_has_pre2_fe"])
SETS = {"aff": AFF, "pop": POP, "all": ALL, "affpop": AFF + POP}
cols = SETS.get(setname, [setname])


def binidx(d, edges):
    out = []
    for c, q in zip(cols, edges):
        out.append(np.searchsorted(q, np.asarray(d[c], np.float64), side="right"))
    return out


edges = [np.unique(np.quantile(np.asarray(dv[c], np.float64), np.linspace(0, 1, 11)[1:-1])) for c in cols]
sizes = [len(q) + 1 for q in edges]
starts = np.cumsum([0] + sizes[:-1])
Bv = basis(dv["g"].astype(np.float64)); Bh = basis(dh["g"].astype(np.float64))
IV = binidx(dv, edges); IH = binidx(dh, edges)


def fit(d, B, I, feeds):
    M = d["M"][feeds]; off0 = d["off"][feeds]
    rows = np.concatenate([np.arange(o, o + m) for o, m in zip(off0, M)])
    off = np.concatenate([[0], np.cumsum(M)[:-1]]); fid = np.repeat(np.arange(len(M)), M)
    y = d["y"][rows].astype(np.float64); Bs = B[rows]; Is = [ix[rows] for ix in I]
    npos = np.bincount(fid, y, len(M)); nb = B.shape[1]; tot = sum(sizes)

    def f(par):
        beta, th = par[:nb], par[nb:]
        l = Bs @ beta
        for j, ix in enumerate(Is):
            l = l + th[starts[j]:starts[j] + sizes[j]][ix]
        mx = np.maximum.reduceat(l, off); e = np.exp(l - mx[fid]); Z = np.bincount(fid, e, len(M)); q = e / Z[fid]
        ll = (y * l).sum() - (npos * (np.log(Z) + mx)).sum()
        r = npos[fid] * q - y
        gb = Bs.T @ r
        gt = np.concatenate([np.bincount(ix, r, sizes[j]) for j, ix in enumerate(Is)])
        return -ll + lam * (th ** 2).sum(), np.concatenate([gb, gt + 2 * lam * th])
    x0 = np.r_[1.0, np.zeros(nb - 1), np.zeros(tot)]
    r = minimize(f, x0, jac=True, method="L-BFGS-B", options={"maxiter": 1000})
    return r.x[:nb], r.x[nb:]


def score(B, I, beta, th):
    s = B @ beta
    for j, ix in enumerate(I):
        s = s + th[starts[j]:starts[j] + sizes[j]][ix]
    return s


fold = folds5(dv, seed=0)
oof = np.empty(len(dv["y"]))
for k in range(5):
    tr = np.where(fold != k)[0]
    beta, th = fit(dv, Bv, IV, tr)
    sel = fold[dv["fid"]] == k
    oof[sel] = score(Bv[sel], [ix[sel] for ix in IV], beta, th)
se = lambda x: x.std() / np.sqrt(len(x))
bv = ndcg(dv, dv["g"]); dl = ndcg(dv, oof) - bv
PW = pmix_w()["private"]; pm, ps = wmean_se(dl, PW)
M = dv["M"]
beta, th = fit(dv, Bv, IV, np.arange(dv["nf"]))
hs = score(Bh, IH, beta, th)
bh = ndcg(dh, dh["g"]); dh_ = ndcg(dh, hs) - bh
# control: the same with theta forced to 0 (calibration only) is a monotone transform of g -> identical NDCG (checked)
print(f"GAM[{setname}] lam {lam}: {len(cols)} inputs | VAL OOF {dl.mean():+.5f}+-{se(dl):.5f} (z {dl.mean()/se(dl):+.2f}) | pmix {pm:+.5f}+-{ps:.5f} | "
      f"M<50 {dl[M<50].mean():+.5f} M>=50 {dl[M>=50].mean():+.5f} | HOLD {dh_.mean():+.5f}+-{se(dh_):.5f} (z {dh_.mean()/se(dh_):+.2f}) ({time.time()-t0:.0f}s)", flush=True)
np.save(f"{D}/gam_oof_{setname}_{lam}.npy", oof.astype(np.float32))
