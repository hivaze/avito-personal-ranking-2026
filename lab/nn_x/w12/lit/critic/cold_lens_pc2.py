"""critic cold_lens_pc2: BLOCK 3 (information). Additive residual forms of the same positive control as cold_lens_pc.py."""
import os, sys
sys.dont_write_bytecode = True
for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"]:
    os.environ[k] = "6"
os.environ["POLARS_MAX_THREADS"] = "2"; os.environ["CUDA_VISIBLE_DEVICES"] = ""
from pathlib import Path
import numpy as np, polars as pl
from sklearn.model_selection import GroupKFold
ROOT = Path("<repo>"); FLAT = ROOT / "data/flat"; FEATS = ROOT / "data/feats"
sys.path.insert(0, str(ROOT / "src"))
from metrics import ndcg_per_feed
f = pl.read_parquet(FLAT / "val/feeds.parquet")
M = f["M"].to_numpy().astype(np.int64); coff = f["cand_off"].to_numpy().astype(np.int64); user = f["user_id"].to_numpy().astype(np.int64)
nF = len(M); fid = np.repeat(np.arange(nF), M)
y = np.load(FLAT / "val/cand_target.npy").astype(np.float64)
coldm = np.asarray(np.load(FEATS / "val/it_hist_all_luo.npy"), np.float64) == 0
def zf(x):
    mu = np.add.reduceat(x, coff) / M; d = x - mu[fid]
    sd = np.sqrt(np.add.reduceat(d * d, coff) / M); return d / np.maximum(sd[fid], 1e-9)
def ld(p): return np.load(p).astype(np.float64)
def pool(paths): return zf(np.mean([zf(ld(p)) for p in paths], 0))
X = pool([ROOT / "nn_x/abl/preds" / f"nn_abl_e10_xdrop_pre2_s{s}_trainonly_val.npy" for s in (951, 952)])
c = np.nan_to_num(np.asarray(np.load(FEATS / "val/pre2_fe_noid.npy"), np.float64))
idx = np.nonzero(coldm)[0]; cf = fid[idx]; nc = np.bincount(cf, minlength=nF).astype(float)
v = c[idx]; mu = np.bincount(cf, weights=v, minlength=nF) / np.maximum(nc, 1); dd = v - mu[cf]
sd = np.sqrt(np.bincount(cf, weights=dd * dd, minlength=nF) / np.maximum(nc, 1)); zc = np.zeros(len(X)); zc[idx] = dd / np.maximum(sd[cf], 1e-9)
zall = zf(c); nX = ndcg_per_feed(X, y, coff, M)
grid = [0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
grp = np.where(user == 0, -1 - np.arange(nF), user)
selF = np.load(ROOT / "nn_x/w12/lit/int_gnn/s5_cold_val.npz")["selF"]
for nm, col in [("R1 cold rows only, additive", zc), ("R2 all rows, additive", zall)]:
    D = np.stack([ndcg_per_feed(X + w * col, y, coff, M) - nX for w in grid], 1)
    for sn, ii in [("sample A", np.sort(selF)), ("all val", np.arange(nF))]:
        nd = np.zeros(len(ii)); ws = []
        for tr, va in GroupKFold(n_splits=5).split(ii, groups=grp[ii]):
            k = int(np.argmax(D[ii[tr]].mean(0))); ws.append(grid[k]); nd[va] = D[ii[va], k]
        se = nd.std() / np.sqrt(len(ii))
        print(f"[{nm}] {sn}: nested {nd.mean():+.5f} ± {se:.5f} w {ws} | in-sample " + " ".join(f"{w}:{D[ii, j].mean():+.5f}" for j, w in enumerate(grid) if w > 0), flush=True)
