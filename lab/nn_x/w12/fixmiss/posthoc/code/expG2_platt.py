"""Platt check of the per-feed group mass (2 parameters, cross-fitted): is the model's group mass calibrated?"""
import ph, numpy as np, polars as pl
from scipy.optimize import minimize
D = ph.Data("val"); fid = D.fid; s = D.s; n = D.n
F = pl.read_parquet("../out/valfeat.parquet", columns=["it_hist_all_luo", "cat_hist_share", "cat"])
mx = np.maximum.reduceat(s, D.off); e = np.exp(s - mx[fid]); q = e / np.bincount(fid, e, n)[fid]
cat = F["cat"].to_numpy(); top = np.array([cat[o + np.argmax(s[o:o + m])] for o, m in zip(D.off, D.M)])
groups = {"newcat": (np.nan_to_num(F["cat_hist_share"].to_numpy(), nan=0) <= 0), "cold": (F["it_hist_all_luo"].to_numpy() <= 0), "offtopcat": (cat != top[fid])}
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
ce = lambda a, p: -(a * np.log(np.clip(p, 1e-9, 1)) + (1 - a) * np.log(np.clip(1 - p, 1e-9, 1))).mean()
for gname, g in groups.items():
    g = g.astype(float); mf = np.bincount(fid, q * g, n); af = np.bincount(fid, D.y * g, n) / D.K; share = np.bincount(fid, g, n) / D.M
    ok = (share > 0) & (share < 1)
    for h in (0, 1):
        A = (D.half == h) & ok; B = (D.half != h) & ok
        r = minimize(lambda p: ce(af[A], 1 / (1 + np.exp(-(p[0] + p[1] * lg(mf[A]))))), [0, 1], method="Nelder-Mead")
        print(f"{gname} half {h}: Platt a {r.x[0]:+.3f} b {r.x[1]:.3f} | CE on B: Platt {ce(af[B], 1 / (1 + np.exp(-(r.x[0] + r.x[1] * lg(mf[B]))))):.5f} identity {ce(af[B], mf[B]):.5f} constant {ce(af[B], np.full(B.sum(), af[A].mean())):.5f}")
print("--- NDCG of the Platt group shift (cross-fitted, 2 params per group) ---")
for gname, g in groups.items():
    g = g.astype(float); mf = np.bincount(fid, q * g, n); af = np.bincount(fid, D.y * g, n) / D.K; share = np.bincount(fid, g, n) / D.M
    ok = (share > 0) & (share < 1); out = s.copy()
    for h in (0, 1):
        A = (D.half == h) & ok; B = (D.half != h) & ok
        r = minimize(lambda p: ce(af[A], 1 / (1 + np.exp(-(p[0] + p[1] * lg(mf[A]))))), [0, 1], method="Nelder-Mead")
        delta = np.zeros(n); delta[B] = r.x[0] + (r.x[1] - 1) * lg(mf[B])
        rr = D.halfr != h; out[rr] = s[rr] + delta[fid][rr] * g[rr]
    D.report(out, f"G2 Platt shift {gname}")
