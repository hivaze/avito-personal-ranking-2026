"""critic lr_B: BLOCK 4 of DECLARATION.txt. Copy of nn_x/w12/lit/syn/lr_read.py (same columns, same conditional logit over cold rows)
with a feed-subset argument: argv[1] = npz of s5_coldsignal.py, argv[2] = ALL | NEW | OLD (NEW = feeds outside sample A of int_gnn)."""
import os, sys
sys.dont_write_bytecode = True
for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS"]:
    os.environ[k] = "6"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from pathlib import Path
import numpy as np, polars as pl
from scipy.optimize import minimize
from scipy.stats import rankdata, chi2
ROOT = Path("<repo>"); FLAT = ROOT / "data/flat"

def feed_z(x, fid, nF):
    cnt = np.maximum(np.bincount(fid, minlength=nF), 1)
    mu = np.bincount(fid, weights=x, minlength=nF) / cnt
    d = x - mu[fid]
    sd = np.sqrt(np.bincount(fid, weights=d * d, minlength=nF) / cnt)
    return d / np.maximum(sd[fid], 1e-9)

Z = np.load(sys.argv[1])
selF, qrows, nv, s1, s2, dg = Z["selF"], Z["qrows"], Z["nv"], Z["s1"], Z["s2"], Z["dg"]
SUB = sys.argv[2]
f0 = pl.read_parquet(FLAT / "val/feeds.parquet"); M0 = f0["M"].to_numpy().astype(np.int64); fid0 = np.repeat(np.arange(len(M0)), M0)
selA = np.load(ROOT / "nn_x/w12/lit/int_gnn/s5_cold_val.npz")["selF"]; inA = np.zeros(len(M0), bool); inA[selA] = True
keepq = np.ones(len(qrows), bool) if SUB == "ALL" else (~inA[fid0[qrows]] if SUB == "NEW" else inA[fid0[qrows]])
qrows, nv, s1, s2, dg = qrows[keepq], nv[keepq], s1[keepq], s2[keepq], dg[keepq]
print(f"== subset {SUB}: cold query rows {len(qrows)}, feeds {len(np.unique(fid0[qrows]))}", flush=True)
f = pl.read_parquet(FLAT / "val/feeds.parquet")
M = f["M"].to_numpy().astype(np.int64); nF = len(M)
fid = np.repeat(np.arange(nF), M)
y = np.load(FLAT / "val/cand_target.npy").astype(np.float64)
wgt = np.where(nv >= 0.90, nv, 0.0); ws = np.maximum(wgt.sum(1), 1e-9)
imp = [(wgt * s1).sum(1) / ws, (wgt * s2 / np.maximum(dg, 1) ** 0.5).sum(1) / ws, np.where(wgt > 0, s1, 0).max(1),
       (wgt * s2 / np.maximum(dg, 1)).sum(1) / ws, ((wgt > 0) & (s1 > 0)).sum(1).astype(np.float64), (wgt * np.log1p(dg)).sum(1) / ws, nv[:, 0]]
logt = [True, True, True, True, True, False, False]
qf_all = fid[qrows]
ug, qf = np.unique(qf_all, return_inverse=True); nS = len(ug)
yq = y[qrows]
ncold = np.bincount(qf, minlength=nS); cpos = np.bincount(qf, weights=yq, minlength=nS)
pair = ((cpos > 0) & (ncold - cpos > 0))[qf]
qstart = np.concatenate([[0], np.cumsum(ncold)[:-1]]).astype(np.int64)
zc = lambda v: feed_z(np.asarray(v, np.float64), qf, nS)
def crank(v):
    r = np.zeros(len(v))
    for ff in range(nS):
        k = ncold[ff]; a = qstart[ff]
        r[a:a + k] = (rankdata(v[a:a + k]) - 0.5) / k - 0.5
    return r
cols = []
for v, lt in zip(imp, logt):
    v = np.asarray(v, np.float64)
    cols.append(zc(np.log1p(v / np.median(v[v > 0]))) if lt else zc(v))
    cols.append(crank(v) * np.sqrt(12.0))
ind = (imp[0] > 0).astype(np.float64)
cols.append(ind - (np.bincount(qf, weights=ind, minlength=nS) / ncold)[qf])
X = np.stack(cols, 1)
zf = lambda v: feed_z(np.asarray(v, np.float64), fid, nF)
def nll(F):
    Fr, yr = F[pair], yq[pair]
    _, gi = np.unique(qf[pair], return_inverse=True); ng = gi.max() + 1
    npos = np.bincount(gi, weights=yr, minlength=ng)
    def obj(th):
        s = Fr @ th
        mx = np.full(ng, -np.inf); np.maximum.at(mx, gi, s)
        e = np.exp(s - mx[gi]); den = np.bincount(gi, weights=e, minlength=ng)
        loss = -(yr * s).sum() + (npos * (mx + np.log(den))).sum()
        r = -yr + npos[gi] * e / den[gi]
        return loss, Fr.T @ r
    th0 = np.zeros(F.shape[1]); th0[0] = 1.0
    o = minimize(obj, th0, jac=True, method="L-BFGS-B", options=dict(maxiter=2000, ftol=1e-13, gtol=1e-9))
    return o.fun, o.x
for pn, a, b in [("F", "ens_FD16P20kd_ft_val.npy", "ens_FD16P20kd_to_val.npy"), ("W", "ens_W16kd_ft_val.npy", "ens_W16kd_to_val.npy")]:
    p = zf(0.5 * zf(np.load(ROOT / "preds" / a)) + 0.5 * zf(np.load(ROOT / "preds" / b)))
    pz = zc(p[qrows]); pr = crank(p[qrows]) * np.sqrt(12.0)
    l0, t0 = nll(pz[:, None])
    print(f"[A5 pool {pn}] feeds {int(len(np.unique(qf[pair])))}, cold rows {int(pair.sum())}, cold positives {int(yq[pair].sum())}; pool alone: -logL {l0:.2f}, coef {t0[0]:+.3f}")
    for name, F, df in [("+ z(x1)", np.column_stack([pz, X[:, 0]]), 1), ("+ z(x2)", np.column_stack([pz, X[:, 2]]), 1),
                        ("+ rank(x1)", np.column_stack([pz, X[:, 1]]), 1), ("+ rank(x2)", np.column_stack([pz, X[:, 3]]), 1),
                        ("+ all 15 extras", np.column_stack([pz, X]), 15),
                        ("+ pool z^2 + pool rank (no neighbour column)", np.column_stack([pz, pz ** 2 - (np.bincount(qf, weights=pz ** 2, minlength=nS) / ncold)[qf], pr]), 2)]:
        l1, t1 = nll(F)
        lr = 2 * (l0 - l1)
        print(f"[A5 pool {pn}] pool {name}: LR {lr:7.2f} df {df} p {chi2.sf(max(lr, 0), df):.4f} | coefs " + " ".join(f"{t:+.3f}" for t in t1[:4]))
    lb, tb = nll(np.column_stack([pz, pz ** 2 - (np.bincount(qf, weights=pz ** 2, minlength=nS) / ncold)[qf], pr]))
    for name, c in [("z(x1)", 0), ("z(x2)", 2), ("rank(x1)", 1), ("rank(x2)", 3)]:
        l1, t1 = nll(np.column_stack([pz, pz ** 2 - (np.bincount(qf, weights=pz ** 2, minlength=nS) / ncold)[qf], pr, X[:, c]]))
        lr = 2 * (lb - l1)
        print(f"[A5 pool {pn}] flexible pool (z, z^2, rank) + {name}: LR {lr:7.2f} df 1 p {chi2.sf(max(lr, 0), 1):.4f} | coef {t1[3]:+.3f}")
