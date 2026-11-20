"""critic cold_lens_pc: BLOCK 2 of DECLARATION.txt. Positive control for the cold-row lens of SYNTHESIS 2.1 (LR test in a
conditional logit over cold rows; slot-preserving cold re-order): the encoder score pre2_fe_noid on members trained without it.
CPU only, read-only on the project."""
import os, sys
sys.dont_write_bytecode = True
for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"]:
    os.environ[k] = "6"
os.environ["POLARS_MAX_THREADS"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import json, time
from pathlib import Path
import numpy as np, polars as pl
from threadpoolctl import threadpool_limits
threadpool_limits(limits=6)
from scipy.optimize import minimize
from scipy.stats import chi2
from sklearn.model_selection import GroupKFold
ROOT = Path("<repo>"); FLAT = ROOT / "data/flat"; FEATS = ROOT / "data/feats"; HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from metrics import ndcg_per_feed
T0 = time.time()
def log(*a): print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)

f = pl.read_parquet(FLAT / "val/feeds.parquet")
M = f["M"].to_numpy().astype(np.int64); coff = f["cand_off"].to_numpy().astype(np.int64); user = f["user_id"].to_numpy().astype(np.int64)
nF = len(M); N = int(M.sum()); fid = np.repeat(np.arange(nF), M)
y = np.load(FLAT / "val/cand_target.npy").astype(np.float64)
luo = np.asarray(np.load(FEATS / "val/it_hist_all_luo.npy"), np.float64)
coldm = luo == 0
def feed_z(x):
    mu = np.add.reduceat(x, coff) / M; d = x - mu[fid]
    sd = np.sqrt(np.add.reduceat(d * d, coff) / M); return d / np.maximum(sd[fid], 1e-9)
zf = feed_z
def ld(p): return np.load(p).astype(np.float64)
def pool(paths): return zf(np.mean([zf(ld(p)) for p in paths], 0))
X = pool([ROOT / "nn_x/abl/preds" / f"nn_abl_e10_xdrop_pre2_s{s}_trainonly_val.npy" for s in (951, 952)])
B = pool([ROOT / "nn_x/w5/preds" / f"nn_w5_e10_base_s{s}_trainonly_val.npy" for s in (951, 952)])
c_raw = np.nan_to_num(np.asarray(np.load(FEATS / "val/pre2_fe_noid.npy"), np.float64))
nX = ndcg_per_feed(X, y, coff, M); nB = ndcg_per_feed(B, y, coff, M)
d = nB - nX
log(f"pool X (xdrop_pre2 x2) val {nX.mean():.5f} | pool B (base x2) val {nB.mean():.5f} | B - X {d.mean():+.5f} ± {d.std() / np.sqrt(nF):.5f}")

idx = np.nonzero(coldm)[0]; cf = fid[idx]                      # cold rows, feed-sorted
ncold = np.bincount(cf, minlength=nF).astype(np.float64)
def zc(v):                                                     # z among the cold rows of the feed
    s1 = np.bincount(cf, weights=v, minlength=nF); mu = s1 / np.maximum(ncold, 1)
    dd = v - mu[cf]; sd = np.sqrt(np.bincount(cf, weights=dd * dd, minlength=nF) / np.maximum(ncold, 1))
    return dd / np.maximum(sd[cf], 1e-9)
yq = y[idx]; cpos = np.bincount(cf, weights=yq, minlength=nF)
has_pair = (cpos > 0) & (ncold - cpos > 0)
zX = zc(X[idx]); zcol = zc(c_raw[idx]); zB = zc(B[idx])
order_pool = np.lexsort((-X[idx], cf)); sorted_pool = X[idx][order_pool]
def reorder(key):                                              # slot-preserving: k-th best key gets the k-th best X score
    o = np.lexsort((-key, cf)); s = X.copy(); s[idx[o]] = sorted_pool; return s
def auc_cold(v, feeds):
    out = []
    o = np.lexsort((v, cf)); r = np.empty(len(v)); r[o] = np.arange(len(v))
    start = np.concatenate([[0], np.cumsum(np.bincount(cf, minlength=nF))[:-1]])
    rk = r - start[cf] + 1                                     # rank within feed (no tie averaging; scores are continuous)
    s1 = np.bincount(cf, weights=rk * (yq > 0), minlength=nF); n1 = cpos; n0 = ncold - cpos
    a = (s1 - n1 * (n1 + 1) / 2) / np.maximum(n1 * n0, 1)
    return a[feeds & has_pair]
grid = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
D = np.stack([ndcg_per_feed(reorder(zX + w * zcol), y, coff, M) - nX for w in grid], 1)   # per feed x grid
assert abs(D[:, 0]).max() < 1e-12
DT = ndcg_per_feed(reorder(zB), y, coff, M) - nX              # trained reference: B's order among cold rows
grp = np.where(user == 0, -1 - np.arange(nF), user)

def clogit_lr(feeds):
    keep = (feeds & has_pair)[cf]
    cfk = cf[keep]; u, inv = np.unique(cfk, return_inverse=True); nf = len(u)
    off = np.concatenate([[0], np.cumsum(np.bincount(inv))[:-1]]); yy = yq[keep]; npos = np.bincount(inv, weights=yy)
    def fit(Fm):
        def obj(th):
            s = Fm @ th; mx = np.maximum.reduceat(s, off); e = np.exp(s - mx[inv]); den = np.add.reduceat(e, off)
            loss = -(yy * s).sum() + (npos * (mx + np.log(den))).sum()
            g = Fm.T @ (-yy + npos[inv] * e / den[inv]); return loss, g
        th0 = np.zeros(Fm.shape[1]); th0[0] = 1.0
        o = minimize(obj, th0, jac=True, method="L-BFGS-B", options=dict(maxiter=500)); return o.x, o.fun
    F0 = zX[keep][:, None]; F1 = np.stack([zX[keep], zcol[keep]], 1)
    t0, l0 = fit(F0); t1, l1 = fit(F1); lr = 2 * (l0 - l1)
    return nf, int(keep.sum()), int(yy.sum()), lr, chi2.sf(max(lr, 0), 1), t1

selF = np.load(ROOT / "nn_x/w12/lit/int_gnn/s5_cold_val.npz")["selF"]
SETS = {"sample A (3,000 feeds)": np.isin(np.arange(nF), selF), "all val feeds": np.ones(nF, bool)}
res = {}
for name, fm in SETS.items():
    n = int(fm.sum()); ii = np.nonzero(fm)[0]
    nf, nr, npz, lr, p, t1 = clogit_lr(fm)
    aX = auc_cold(X[idx], fm); aC = auc_cold(c_raw[idx], fm); aB = auc_cold(B[idx], fm)
    log(f"[{name}] feeds {n}; pair feeds {nf}, cold rows {nr}, cold positives {npz} | AUC among cold rows: X {aX.mean():.4f}, column {aC.mean():.4f}, B {aB.mean():.4f}")
    log(f"[{name}] L1 LR zc(X) vs zc(X)+zc(c): LR {lr:.1f} df 1 p {p:.3g} | coefs {t1[0]:+.3f} {t1[1]:+.3f}")
    folds = list(GroupKFold(n_splits=5).split(ii, groups=grp[ii]))
    nd = np.zeros(n); ws = []
    for tr, va in folds:
        k = int(np.argmax(D[ii[tr]].mean(0))); ws.append(grid[k]); nd[va] = D[ii[va], k]
    se = nd.std() / np.sqrt(n)
    cur = " ".join(f"{w}:{D[ii, j].mean():+.5f}" for j, w in enumerate(grid) if w > 0)
    log(f"[{name}] L2 nested {nd.mean():+.5f} ± {se:.5f} (z {nd.mean() / se:+.2f}) w {ws} | in-sample {cur}")
    dt = DT[ii]; db = (nB - nX)[ii]
    log(f"[{name}] L3 trained reference: cold rows of X in B's order {dt.mean():+.5f} ± {dt.std() / np.sqrt(n):.5f} | B - X {db.mean():+.5f} ± {db.std() / np.sqrt(n):.5f}")
    res[name] = dict(pair_feeds=nf, lr=float(lr), p=float(p), nested=float(nd.mean()), se=float(se), w=ws,
                     insample={str(w): float(D[ii, j].mean()) for j, w in enumerate(grid)}, trained_cold=float(dt.mean()),
                     trained_cold_se=float(dt.std() / np.sqrt(n)), b_minus_x=float(db.mean()))
(HERE / "cold_lens_pc.json").write_text(json.dumps(res, indent=1))
log("DONE")
