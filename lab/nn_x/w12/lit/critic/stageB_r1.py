"""critic stageB_r1: BLOCK 5 (information). Additive cold-row residual of the neighbour-completed columns on all val feeds."""
import os, sys
sys.dont_write_bytecode = True
for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS"]:
    os.environ[k] = "4"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from pathlib import Path
import numpy as np, polars as pl
from sklearn.model_selection import GroupKFold
ROOT = Path("<repo>"); FLAT = ROOT / "data/flat"
sys.path.insert(0, str(ROOT / "src"))
from metrics import ndcg_per_feed
import pmix
Z = np.load(sys.argv[1])
qrows, nv, s1, s2, dg = Z["qrows"], Z["nv"], Z["s1"], Z["s2"], Z["dg"]
f = pl.read_parquet(FLAT / "val/feeds.parquet")
M = f["M"].to_numpy().astype(np.int64); coff = f["cand_off"].to_numpy().astype(np.int64); user = f["user_id"].to_numpy().astype(np.int64)
nF = len(M); fid = np.repeat(np.arange(nF), M)
y = np.load(FLAT / "val/cand_target.npy").astype(np.float64)
def zf(x):
    mu = np.add.reduceat(x, coff) / M; d = x - mu[fid]
    sd = np.sqrt(np.add.reduceat(d * d, coff) / M); return d / np.maximum(sd[fid], 1e-9)
wgt = np.where(nv >= 0.90, nv, 0.0); ws = np.maximum(wgt.sum(1), 1e-9)
cols = {"x1": (wgt * s1).sum(1) / ws, "x2": (wgt * s2 / np.maximum(dg, 1) ** 0.5).sum(1) / ws}
qf = fid[qrows]; nc = np.bincount(qf, minlength=nF).astype(float)
def zc_full(v):
    v = np.log1p(v / np.median(v[v > 0]))
    mu = np.bincount(qf, weights=v, minlength=nF) / np.maximum(nc, 1); dd = v - mu[qf]
    sd = np.sqrt(np.bincount(qf, weights=dd * dd, minlength=nF) / np.maximum(nc, 1))
    out = np.zeros(len(y)); out[qrows] = dd / np.maximum(sd[qf], 1e-9); return out
grid = [0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
grp = np.where(user == 0, -1 - np.arange(nF), user); W = pmix.weights()
def wm(d, w):
    m = np.average(d, weights=w); neff = w.sum() ** 2 / (w ** 2).sum(); return m, np.sqrt(np.average((d - m) ** 2, weights=w) / neff)
for pn, a, b in [("F", "ens_FD16P20kd_ft_val.npy", "ens_FD16P20kd_to_val.npy"), ("W", "ens_W16kd_ft_val.npy", "ens_W16kd_to_val.npy")]:
    p = zf(0.5 * zf(np.load(ROOT / "preds" / a).astype(np.float64)) + 0.5 * zf(np.load(ROOT / "preds" / b).astype(np.float64)))
    base = ndcg_per_feed(p, y, coff, M)
    for cn, v in cols.items():
        c = zc_full(np.asarray(v, np.float64))
        D = np.stack([ndcg_per_feed(p + w * c, y, coff, M) - base for w in grid], 1)
        nd = np.zeros(nF); wsel = []
        for tr, va in GroupKFold(n_splits=5).split(np.arange(nF), groups=grp):
            k = int(np.argmax(D[tr].mean(0))); wsel.append(grid[k]); nd[va] = D[va, k]
        se = nd.std() / np.sqrt(nF); mr, sr = wm(nd, W["private"])
        print(f"[R1 pool {pn}] {cn}: nested {nd.mean():+.5f} ± {se:.5f} w {wsel} | pmix-private {mr:+.5f}±{sr:.5f} | in-sample " + " ".join(f"{w}:{D[:, j].mean():+.5f}" for j, w in enumerate(grid) if w > 0), flush=True)
