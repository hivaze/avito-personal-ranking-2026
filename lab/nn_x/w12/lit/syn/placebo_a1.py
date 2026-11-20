"""syn placebo_a1: amendment A6 (information only): local-permutation placebo of the rank-residual AUC."""
import os, sys
sys.dont_write_bytecode = True
for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS"]:
    os.environ[k] = "12"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from pathlib import Path
import numpy as np, polars as pl
from scipy.stats import rankdata
ROOT = Path("<repo>"); FLAT = ROOT / "data/flat"
def feed_z(x, fid, nF):
    cnt = np.maximum(np.bincount(fid, minlength=nF), 1)
    mu = np.bincount(fid, weights=x, minlength=nF) / cnt
    d = x - mu[fid]
    sd = np.sqrt(np.bincount(fid, weights=d * d, minlength=nF) / cnt)
    return d / np.maximum(sd[fid], 1e-9)
Z = np.load(sys.argv[1])
qrows, nv, s1, s2, dg = Z["qrows"], Z["nv"], Z["s1"], Z["s2"], Z["dg"]
f = pl.read_parquet(FLAT / "val/feeds.parquet")
M = f["M"].to_numpy().astype(np.int64); nF = len(M)
fid = np.repeat(np.arange(nF), M)
y = np.load(FLAT / "val/cand_target.npy").astype(np.float64)
wgt = np.where(nv >= 0.90, nv, 0.0); ws = np.maximum(wgt.sum(1), 1e-9)
cols = {"x1": (wgt * s1).sum(1) / ws, "x2": (wgt * s2 / np.maximum(dg, 1) ** 0.5).sum(1) / ws}
_, qf = np.unique(fid[qrows], return_inverse=True); nS = qf.max() + 1
yq = y[qrows]
ncold = np.bincount(qf, minlength=nS); cpos = np.bincount(qf, weights=yq, minlength=nS)
has_pair = (cpos > 0) & (ncold - cpos > 0)
qstart = np.concatenate([[0], np.cumsum(ncold)[:-1]]).astype(np.int64)
def crank(v):
    r = np.zeros(len(v))
    for ff in range(nS):
        k = ncold[ff]; a = qstart[ff]
        r[a:a + k] = (rankdata(v[a:a + k]) - 0.5) / k - 0.5
    return r
def auc_cold(v):
    out = []
    for ff in np.nonzero(has_pair)[0]:
        a, k = qstart[ff], ncold[ff]
        r = rankdata(v[a:a + k]); yy = yq[a:a + k] > 0; n1 = yy.sum()
        out.append((r[yy].sum() - n1 * (n1 + 1) / 2) / (n1 * (k - n1)))
    return np.array(out)
def a1(rx, rp):
    b = (rp @ rx) / (rp @ rp)
    au = auc_cold(rx - b * rp)
    return au.mean(), au.std() / np.sqrt(len(au))
zf = lambda v: feed_z(np.asarray(v, np.float64), fid, nF)
for pn, a, b in [("F", "ens_FD16P20kd_ft_val.npy", "ens_FD16P20kd_to_val.npy"), ("W", "ens_W16kd_ft_val.npy", "ens_W16kd_to_val.npy")]:
    p = zf(0.5 * zf(np.load(ROOT / "preds" / a)) + 0.5 * zf(np.load(ROOT / "preds" / b)))[qrows]
    rp = crank(p)
    # position of every cold row in the pool order of its feed (0 = best)
    o = np.lexsort((-p, qf)); pos = np.empty(len(p), np.int64); pos[o] = np.arange(len(p)) - qstart[qf[o]]
    for n, x in cols.items():
        real, se = a1(crank(x), rp)
        line = f"[A6 pool {pn}] {n}: real A1(i) {real:.4f}±{se:.4f}"
        for B in (4, 8):
            blk = pos // B
            pl_ = []
            for sd in range(10):
                rng = np.random.default_rng(sd)
                oo = np.lexsort((rng.random(len(p)), blk, qf))      # rows grouped by feed, block; random inside
                src = np.lexsort((pos, blk, qf))                    # rows grouped by feed, block; pool order inside
                xp = np.empty_like(x); xp[src] = x[oo]
                pl_.append(a1(crank(xp), rp)[0])
            line += f" | placebo B={B}: mean {np.mean(pl_):.4f} sd {np.std(pl_):.4f} max {np.max(pl_):.4f} -> real minus placebo {real - np.mean(pl_):+.4f}"
        print(line, flush=True)
