"""(c) item-level branch-disagreement terms and (d) slice calibration offsets, cross-fitted by user halves.
(c) key = s + g * t, t in {|zft-zto|, (zto-zft), (zpl-zbase)}; g chosen on half A from a grid (in-sample on A), applied on B.
(d) per-bin offsets = log((O+k)/(E+k)) of the final score on half A (E = K * softmax(s), slope 1.000 fitted earlier),
    applied on B with lambda 1 (declared) and 0.5; bins: listing age f4 deciles, popularity it_hist_all_luo (0 + deciles),
    exposure xp_imp_c50 deciles, new-to-user category x feed-size class, cold x feed-size class; and all of them jointly
    as a ridge conditional logit (one-hot bins, base score as fixed offset)."""
import ph, numpy as np, polars as pl, json
from scipy.optimize import minimize
D = ph.Data("val"); fid = D.fid; s = D.s
C = np.load("../out/comp_val.npz"); z = {k: ph.zf(C[k], D.off, D.M) for k in C.files}
F = pl.read_parquet("../out/valfeat.parquet", columns=["it_hist_all_luo", "xp_imp_c50", "cat_hist_share"])
f4 = np.load("<repo>/data/flat/val/cand_f4.npy").astype(np.float64)
def smax(x):
    mx = np.maximum.reduceat(x, D.off); e = np.exp(x - mx[fid]); return e / np.bincount(fid, e, D.n)[fid]
E = D.K[fid] * smax(s)
res = {}
# ---- (c)
zpl = 0.5 * (z["ft_pl"] + z["to_pl"]); zb = 0.5 * (z["ft_base"] + z["to_base"])
for nm, t in (("absdis", np.abs(D.zft - D.zto)), ("to_minus_ft", D.zto - D.zft), ("pl_minus_base", zpl - zb)):
    grid = np.round(np.arange(-0.3, 0.31, 0.05), 2)
    tab = np.stack([D.ev(s + g * t) for g in grid], 1)
    out = np.zeros(D.n)
    for h in (0, 1):
        A = D.half == h; gi = tab[A].mean(0).argmax(); out[~A] = tab[~A, gi]; print(f"(c) {nm} half {h} chose g {grid[gi]}")
    print(f"(c) {nm} in-sample curve:", " ".join(f"{g:+.2f}:{v - D.base.mean():+.5f}" for g, v in zip(grid, tab.mean(0))))
    res[f"c_{nm}"] = D.report(out, f"C nested {nm}")
# ---- (d)
Mc = np.digitize(D.M, [30, 50, 100, 200])[fid]
ih = F["it_hist_all_luo"].to_numpy().astype(np.float64); xp = F["xp_imp_c50"].to_numpy().astype(np.float64)
newcat = (np.nan_to_num(F["cat_hist_share"].to_numpy(), nan=0) <= 0).astype(int)
def qbins(x, nb=10):
    qs = np.unique(np.quantile(x, np.linspace(0, 1, nb + 1)[1:-1])); return np.digitize(x, qs)
bins = {
    "age_f4": qbins(f4),
    "pop_ih": np.where(ih <= 0, 0, 1 + qbins(np.where(ih <= 0, np.nan, ih)[ih > 0].repeat(1) if False else np.log1p(np.maximum(ih, 0)), 10)),
    "exp_xp": qbins(xp),
    "newcat_x_M": newcat * 5 + Mc,
    "cold_x_M": (ih <= 0).astype(int) * 5 + Mc,
}
bins["pop_ih"] = np.where(ih <= 0, 0, 1 + qbins(np.log1p(np.maximum(ih, 0))))
for nm, b in bins.items():
    for lam in (1.0, 0.5):
        out = np.zeros(len(s))
        for h in (0, 1):
            A = D.halfr == h
            O = np.bincount(b[A], D.y[A], b.max() + 1); Ee = np.bincount(b[A], E[A], b.max() + 1)
            off = np.log((O + 30) / (Ee + 30))
            out[~A] = s[~A] + lam * off[b[~A]]
            if lam == 1.0: print(f"(d) {nm} half {h} offsets", np.round(off, 3))
        res[f"d_{nm}_l{lam}"] = D.report(out, f"D offset {nm} lam {lam}")
# joint ridge conditional logit on one-hot bins
cols = []
for nm, b in bins.items():
    for v in np.unique(b)[1:]:
        cols.append((b == v).astype(np.float32))
X = np.column_stack(cols); print("joint design", X.shape)
tgt = D.y / D.K[fid]
for lam2 in (30.0, 300.0):
    out = np.zeros(len(s))
    for h in (0, 1):
        A = D.halfr == h; fA = fid[A]; offA = np.r_[0, np.cumsum(D.M[D.half == h])[:-1]]
        XA, sA, tA = X[A], s[A], tgt[A]
        fa = np.repeat(np.arange((D.half == h).sum()), D.M[D.half == h])
        def nll(beta):
            u = sA + XA @ beta; mx = np.maximum.reduceat(u, offA); e = np.exp(u - mx[fa]); Zs = np.bincount(fa, e)
            p = e / Zs[fa]; ll = (tA * (u - mx[fa] - np.log(Zs[fa]))).sum()
            g = XA.T @ (tA - p * np.bincount(fa, tA)[fa])
            return -ll + lam2 * beta @ beta, -g + 2 * lam2 * beta
        r = minimize(nll, np.zeros(X.shape[1]), jac=True, method="L-BFGS-B")
        out[~A] = s[~A] + X[~A] @ r.x
        print(f"joint ridge {lam2} half {h}: |beta| max {np.abs(r.x).max():.3f}")
    res[f"d_joint_r{lam2}"] = D.report(out, f"D joint ridge {lam2}")
json.dump({k: {kk: vv for kk, vv in v.items() if kk != 'line'} for k, v in res.items()}, open("../out/expCD.json", "w"), indent=1)
