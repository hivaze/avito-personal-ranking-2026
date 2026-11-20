"""W25 column-level probe on val (CPU), same protocol as nn_x/audit/altapproach/probe_residuals.py:
pairwise AUC of the within-feed z of each W25 column on all (pos, neg) pairs / same-category pairs / the INSIGHTS warm-beater cell
(P8 gate: cell AUC >= 0.53), and the nested (user-grouped, 2 seeds) linear residual on G3 at w in {0,.05,.1,.2,.3,.5}.
Leak-free val feeds only (eda/errors/synthesis/qt.leak_mask). Val labels are used only inside the nested OOF scheme (no fitting)."""
import os
for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "POLARS_MAX_THREADS", "NUMBA_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
    os.environ[k] = "12"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import json, sys, time
from pathlib import Path
import numpy as np, polars as pl, numba as nb
sys.path.insert(0, "<repo>/src"); sys.path.insert(0, "<repo>/eda/errors/synthesis")
from metrics import ndcg_per_feed  # noqa: E402
from stack_val import feed_rank_z  # noqa: E402
from qt import leak_mask  # noqa: E402
from pmix import weights as pmix_weights, wmean_se  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
ROOT = Path("<repo>"); FLAT = ROOT / "data/flat/val"; P = ROOT / "preds"; ERR = ROOT / "nn_x/insights/errors/out"
WD = ROOT / "nn_x/w25/data"; OUT = ROOT / "nn_x/w25/out"
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:6.1f}s]", *a, flush=True)


f = pl.read_parquet(FLAT / "feeds.parquet")
M = f["M"].to_numpy().astype(np.int64); off = f["cand_off"].to_numpy().astype(np.int64); n = len(M)
fid = np.repeat(np.arange(n), M)
y = np.load(FLAT / "cand_target.npy").astype(np.float64)
cat = np.load(FLAT / "cand_category_id.npy").astype(np.int64)
users = f["user_id"].to_numpy().copy(); users[users == 0] = -np.arange(1, (users == 0).sum() + 1)
keep = ~leak_mask(); log(f"val feeds {n}, leak-free {keep.sum()}")
wpriv = pmix_weights()["private"]
z = lambda x: feed_rank_z(off, M, np.asarray(x, np.float32))[:, 1].astype(np.float64)
zG3 = 0.5 * z(np.load(P / "ens_G3_ft_val.npy")) + 0.5 * z(np.load(P / "ens_G3_to_val.npy"))
cold = np.load(ERR / "cold_luo.npy").astype(bool); minh = np.load(ERR / "cand_cat_minh.npy")
base_per = ndcg_per_feed(zG3, y, off, M)
log("G3 base", round(float(base_per[keep].mean()), 5))


@nb.njit(cache=True)
def feed_z_nan(off, M, x):
    out = np.zeros(len(x))
    for g in range(len(M)):
        o, m = off[g], M[g]
        s = 0.0; s2 = 0.0; c = 0
        for k in range(m):
            v = x[o + k]
            if v == v:
                s += v; s2 += v * v; c += 1
        if c >= 2:
            mu = s / c; sd = np.sqrt(max(s2 / c - mu * mu, 0.0))
            if sd > 0:
                for k in range(m):
                    v = x[o + k]
                    if v == v:
                        out[o + k] = (v - mu) / sd
    return out


pi, ni = [], []
for g in np.nonzero(keep)[0]:
    s, m = off[g], M[g]; yy = y[s:s + m]
    p = np.nonzero(yy == 1)[0] + s; q = np.nonzero(yy == 0)[0] + s
    if len(p) and len(q):
        pi.append(np.repeat(p, len(q))); ni.append(np.tile(q, len(p)))
pi = np.concatenate(pi); ni = np.concatenate(ni)
same = cat[pi] == cat[ni]
ck = fid * 64 + cat; uk, inv = np.unique(ck, return_inverse=True)
lse = np.log(np.bincount(inv, np.exp(zG3)))
kf = uk // 64; o = np.lexsort((-lse, kf)); cr = np.empty(len(uk), np.int64); cr[o] = np.arange(len(uk)) - np.searchsorted(kf[o], kf[o])
pcr = cr[inv]
cell = same & (~cold[pi]) & (~cold[ni]) & (minh[pi] < 8) & (pcr[pi] == 0) & (zG3[ni] > zG3[pi])
log(f"pairs {len(pi):,} same-cat {same.mean():.3f} cell {cell.sum():,}")
auc = lambda d: float(((d > 0) + 0.5 * (d == 0)).mean()) if len(d) else float("nan")
WG = [0, .05, .1, .2, .3, .5]
folds = {seed: list(GroupKFold(5, shuffle=True, random_state=seed).split(np.arange(n), groups=users)) for seed in range(2)}
G = json.loads((WD / "groups.json").read_text())
names = [nm for g in sys.argv[1:] or [k for k in G if k != "G_excl"] for nm in G[g]]
seen = set(); names = [nm for nm in names if not (nm in seen or seen.add(nm))]
names += ["tag:" + t for t in os.environ.get("W25_TAGS", "").split(",") if t]
rows = []
print(f"{'column':26s} {'cov':>5s} | {'AUCall':>6s} {'same':>6s} {'cell':>6s} | nested±SE  pmix±SE  w*", flush=True)
for nm in names:
    x = np.load(OUT / f"{nm[4:]}_val.npy") if nm.startswith("tag:") else np.load(WD / f"val/{nm}.npy")
    x = np.asarray(x, np.float64); dfn = np.isfinite(x)
    zx = feed_z_nan(off, M, np.where(dfn, x, np.nan))
    d = zx[pi] - zx[ni]; both = dfn[pi] & dfn[ni]
    a_all, a_same, a_cell = auc(d[both]), auc(d[both & same]), auc(d[both & cell])
    pers = {w: ndcg_per_feed(zG3 + w * zx, y, off, M) for w in WG}
    deltas = []
    for seed in range(2):
        out = np.zeros(n)
        for trn, va in folds[seed]:
            trk = trn[keep[trn]]
            wbest = max(WG, key=lambda w: pers[w][trk].mean())
            out[va] = pers[wbest][va] - base_per[va]
        deltas.append(out)
    dn = np.mean(deltas, 0); m_, se_ = dn[keep].mean(), dn[keep].std() / np.sqrt(keep.sum())
    pm, ps = wmean_se(dn[keep], wpriv[keep]); wfull = max(WG, key=lambda w: pers[w][keep].mean())
    rows.append(dict(column=nm, cov=float(dfn.mean()), auc_all=a_all, auc_same=a_same, auc_cell=a_cell, nested=m_, nested_se=se_, pmix=pm, pmix_se=ps, wfull=wfull))
    print(f"{nm:26s} {dfn.mean():5.3f} | {a_all:6.4f} {a_same:6.4f} {a_cell:6.4f} | {m_:+.5f}±{se_:.5f} {pm:+.5f}±{ps:.5f} w{wfull:.2f}", flush=True)
pl.DataFrame(rows).write_csv(OUT / "cell_w25.csv")
