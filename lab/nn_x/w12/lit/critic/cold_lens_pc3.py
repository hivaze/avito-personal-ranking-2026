"""critic cold_lens_pc3: BLOCK 7 information. Positive control of cold_lens_pc.py, rank form of the column."""
import os, sys
sys.dont_write_bytecode = True
for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS"]:
    os.environ[k] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from pathlib import Path
import numpy as np, polars as pl
from scipy.optimize import minimize
from scipy.stats import chi2
ROOT = Path("<repo>"); FLAT = ROOT / "data/flat"; FEATS = ROOT / "data/feats"
f = pl.read_parquet(FLAT / "val/feeds.parquet")
M = f["M"].to_numpy().astype(np.int64); coff = f["cand_off"].to_numpy().astype(np.int64); nF = len(M); fid = np.repeat(np.arange(nF), M)
y = np.load(FLAT / "val/cand_target.npy").astype(np.float64)
coldm = np.asarray(np.load(FEATS / "val/it_hist_all_luo.npy"), np.float64) == 0
def zf(x):
    mu = np.add.reduceat(x, coff) / M; d = x - mu[fid]
    sd = np.sqrt(np.add.reduceat(d * d, coff) / M); return d / np.maximum(sd[fid], 1e-9)
X = zf(np.mean([zf(np.load(ROOT / "nn_x/abl/preds" / f"nn_abl_e10_xdrop_pre2_s{s}_trainonly_val.npy").astype(np.float64)) for s in (951, 952)], 0))
c = np.nan_to_num(np.asarray(np.load(FEATS / "val/pre2_fe_noid.npy"), np.float64))
idx = np.nonzero(coldm)[0]; cf = fid[idx]; nc = np.bincount(cf, minlength=nF).astype(float)
def zc(v):
    mu = np.bincount(cf, weights=v, minlength=nF) / np.maximum(nc, 1); dd = v - mu[cf]
    sd = np.sqrt(np.bincount(cf, weights=dd * dd, minlength=nF) / np.maximum(nc, 1)); return dd / np.maximum(sd[cf], 1e-9)
def crank(v):
    o = np.lexsort((v, cf)); r = np.empty(len(v)); r[o] = np.arange(len(v))
    start = np.concatenate([[0], np.cumsum(nc.astype(np.int64))[:-1]])
    return ((r - start[cf] + 0.5) / nc[cf] - 0.5) * np.sqrt(12.0)
yq = y[idx]; cpos = np.bincount(cf, weights=yq, minlength=nF); has_pair = (cpos > 0) & (nc - cpos > 0)
zX = zc(X[idx]); cols = {"z": zc(c[idx]), "rank": crank(c[idx])}
selF = np.load(ROOT / "nn_x/w12/lit/int_gnn/s5_cold_val.npz")["selF"]
for name, fm in [("sample A", np.isin(np.arange(nF), selF)), ("all val feeds", np.ones(nF, bool))]:
    keep = (fm & has_pair)[cf]; _, inv = np.unique(cf[keep], return_inverse=True)
    off = np.concatenate([[0], np.cumsum(np.bincount(inv))[:-1]]); yy = yq[keep]; npos = np.bincount(inv, weights=yy)
    def fit(Fm):
        def obj(th):
            s = Fm @ th; mx = np.maximum.reduceat(s, off); e = np.exp(s - mx[inv]); den = np.add.reduceat(e, off)
            return -(yy * s).sum() + (npos * (mx + np.log(den))).sum(), Fm.T @ (-yy + npos[inv] * e / den[inv])
        th0 = np.zeros(Fm.shape[1]); th0[0] = 1.0
        o = minimize(obj, th0, jac=True, method="L-BFGS-B", options=dict(maxiter=2000, ftol=1e-13, gtol=1e-9)); return o.x, o.fun
    _, l0 = fit(zX[keep][:, None])
    for cn, col in cols.items():
        t1, l1 = fit(np.stack([zX[keep], col[keep]], 1)); lr = 2 * (l0 - l1)
        print(f"[{name}] positive control, {cn} form: LR {lr:.2f} df 1 p {chi2.sf(max(lr, 0), 1):.3g} | coefs {t1[0]:+.3f} {t1[1]:+.3f}", flush=True)
