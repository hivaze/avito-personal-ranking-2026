"""web_shift probe s5: residual conditional logit of stored val pools on the lag-deflation delta (PREDECLARED.txt addendum)."""
import sys
from pathlib import Path
import numpy as np
import polars as pl
ROOT = Path("<repo>"); HERE = ROOT / "reports/TRICKS_1115/web_shift_probe"
f = pl.read_parquet(ROOT / "data/flat/val/feeds.parquet")
M = f["M"].to_numpy().astype(np.int64); n = len(M); fid = np.repeat(np.arange(n), M)
y = np.load(ROOT / "data/flat/val/cand_target.npy").astype(np.float64)
cnt = np.load(ROOT / "data/feats/val/it_hist_all_luo.npy"); cold = (cnt <= 0).astype(np.float64)
dv = np.load(HERE / "cf/delta_val.npy").astype(np.float64)
npos = np.bincount(fid, y, n); keep = npos[fid] > 0; w = np.where(keep, y / np.maximum(npos[fid], 1), 0.0)
def z(s):
    s = s.astype(np.float64); mu = np.bincount(fid, s, n) / M
    sd = np.sqrt(np.maximum(np.bincount(fid, s * s, n) / M - mu ** 2, 0)) + 1e-9
    return (s - mu[fid]) / sd[fid]
def fit(X):
    b = np.zeros(X.shape[1]); b[0] = 1.0
    for it in range(50):
        l = X @ b; l -= np.maximum.reduceat(l, np.r_[0, np.cumsum(M)[:-1]])[fid]
        e = np.exp(l); p = e / np.bincount(fid, e, n)[fid]; p = np.where(keep, p, 0.0)
        g = X.T @ (w - p)
        Xp = X * p[:, None]; mu = np.stack([np.bincount(fid, Xp[:, j], n) for j in range(X.shape[1])], 1)
        Hm = X.T @ Xp - mu.T @ (mu * (npos > 0)[:, None])
        step = np.linalg.solve(Hm, g); b = b + step
        if np.abs(step).max() < 1e-8:
            break
    R = (X * (w - p)[:, None]); G = np.stack([np.bincount(fid, R[:, j], n) for j in range(X.shape[1])], 1)
    Hi = np.linalg.inv(Hm); V = Hi @ (G.T @ G) @ Hi
    return b, np.sqrt(np.diag(V))
for tag in sys.argv[1:]:
    for br in ("to", "ft"):
        p = ROOT / "preds" / f"ens_{tag}_{br}_val.npy"
        if not p.exists():
            print(f"{tag} {br}: no file"); continue
        s = z(np.load(p))
        b, se = fit(np.stack([s, cold, dv], 1))
        b0, se0 = fit(np.stack([s, cold], 1))
        print(f"{tag:12s} {br}: g(delta) {b[2]:+.4f} +- {se[2]:.4f} (z {b[2]/se[2]:+.2f}) | cold dummy {b[1]:+.4f} +- {se[1]:.4f} (without delta {b0[1]:+.4f} +- {se0[1]:.4f}) | slope on z {b[0]:.3f}")
