"""Control for gam.py: calibration-only (theta = 0) refit per fold must reproduce G3's NDCG (monotone transform check)."""
import sys
import numpy as np
sys.path.insert(0, "/tmp/scratchpad/dpa/nonlinear")
from nlc import load, folds5, ndcg, fit_calib
dv = load("val"); fold = folds5(dv, 0)
KN = (-1.0, 0.0, 1.0, 1.5, 2.0, 2.5, 3.0)
B = np.stack([dv["g"]] + [np.maximum(dv["g"] - k, 0) for k in KN], 1).astype(np.float64)
beta, basis, p, npos = fit_calib(dv, dv["g"])
s = B @ beta
gg = np.linspace(-4, 6, 1001); bb = np.stack([gg] + [np.maximum(gg - k, 0) for k in KN], 1) @ beta
print("full-val beta", np.round(beta, 3).tolist(), "monotone on [-4,6]:", bool(np.all(np.diff(bb) > 0)))
d = ndcg(dv, s) - ndcg(dv, dv["g"])
print(f"calibration-only score vs G3: NDCG delta {d.mean():+.6f} (max |per-feed| {np.abs(d).max():.2e})")
