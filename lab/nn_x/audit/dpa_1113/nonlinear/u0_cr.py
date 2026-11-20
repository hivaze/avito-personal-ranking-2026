"""CR (label top/bottom decile lift over calibrated-model top/bottom) split by user0 (guest) vs logged-in feeds, val and holdout."""
import sys
import numpy as np
sys.path.insert(0, "/tmp/scratchpad/dpa/nonlinear")
from nlc import load, fit_calib
for split in ["val", "hold"]:
    d = load(split); fid = d["fid"]; y = d["y"].astype(np.float64); M = d["M"]
    beta, basis, p, npos = fit_calib(d, d["g"]); E = npos[fid] * p; E0 = npos[fid] / M[fid]
    u0 = d["u0"][fid]
    print(f"[{split}] user0 feeds {d['u0'].mean():.3f}, rows {u0.mean():.3f}")
    for n in ["f3", "d_cat_last32", "xt_ctx_nd_knn_f1", "d_cos_u", "d_fcs"]:
        x = np.asarray(d[n], np.float64)
        q = np.unique(np.quantile(x, np.linspace(0, 1, 11)[1:-1])); b = np.unique(np.searchsorted(q, x, side="right"), return_inverse=True)[1]; nb = b.max() + 1
        out = []
        for m in [u0, ~u0]:
            A = np.bincount(b[m], y[m], nb); Eb = np.bincount(b[m], E[m], nb); E0b = np.bincount(b[m], E0[m], nb)
            ll, ml = A / np.maximum(E0b, 1e-12), Eb / np.maximum(E0b, 1e-12)
            out.append(((ll[-1] / ll[0]) / (ml[-1] / ml[0]), ll[0], ml[0], ll[-1], ml[-1]))
        print(f"  {n:18s} guest CR {out[0][0]:.3f} (bottom L/M {out[0][1]:.2f}/{out[0][2]:.2f}, top {out[0][3]:.2f}/{out[0][4]:.2f}) | logged-in CR {out[1][0]:.3f} (bottom {out[1][1]:.2f}/{out[1][2]:.2f}, top {out[1][3]:.2f}/{out[1][4]:.2f})")
