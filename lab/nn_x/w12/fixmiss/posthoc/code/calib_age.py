"""Calibration of the final score, the FT and the TO branch by listing age (f4) and by val day (lag), O/E with
expected = K_f * softmax(a s) (a fitted globally per score by listwise CE; 1 parameter)."""
import ph, numpy as np, polars as pl
from scipy.optimize import minimize_scalar
D = ph.Data("val"); fid = D.fid
f4 = np.load("<repo>/data/flat/val/cand_f4.npy").astype(np.float64)
print("f4 quantiles", np.quantile(f4, [0, .01, .1, .25, .5, .75, .9, .99, 1]).round(3))
ts = D.f["timestamp"].to_numpy(); day = (ts - ts.min()) // 86400; print("val days", np.bincount(day))
def smax(x):
    mx = np.maximum.reduceat(x, D.off); e = np.exp(x - mx[fid]); return e / np.bincount(fid, e, D.n)[fid]
def fit_a(x):
    t = D.y / D.K[fid]
    r = minimize_scalar(lambda a: -(t * np.log(smax(a * x) + 1e-300)).sum(), bounds=(0.2, 3), method="bounded"); return r.x
qs = np.quantile(f4, [0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 0.95])
b = np.digitize(f4, qs)
dayr = day[fid]
for nm, x in (("final", D.s), ("FT", D.zft), ("TO", D.zto)):
    a = fit_a(x); E = D.K[fid] * smax(a * x)
    print(f"{nm}: slope a {a:.3f}")
    print("  f4 bin    rows   O/E all   O/E day0     O/E day2")
    for k in range(len(qs) + 1):
        m = b == k
        o, e = D.y[m].sum(), E[m].sum()
        se = np.sqrt(o) / e
        m1, m2 = m & (dayr <= 1), m & (dayr >= 2)
        print(f"  {k:2d} f4<={np.r_[qs, f4.max()][k]:8.3f} {m.sum():8d}  {o / e:.3f}+-{se:.3f}   {D.y[m1].sum() / E[m1].sum():.3f}   {D.y[m2].sum() / E[m2].sum():.3f}")
