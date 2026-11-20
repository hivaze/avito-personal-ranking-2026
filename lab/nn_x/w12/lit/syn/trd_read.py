"""syn trd_read: amendment A8 of DECLARATION.txt (information only). Residual-lens read of the 10 fresh-density columns
(lit_gnn S2 family i) on the current file: conditional logit over all rows of a feed, nested by user, with placebo.
CPU only, read-only on the project."""
import os, sys
sys.dont_write_bytecode = True
for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS"]:
    os.environ[k] = "12"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import json, time
from pathlib import Path
import numpy as np, polars as pl
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
cold = (np.asarray(np.load(FEATS / "val/it_hist_all_luo.npy"), np.float64) == 0).astype(np.float64)
npos = np.add.reduceat(y, coff)
def feed_z(x):
    mu = np.add.reduceat(x, coff) / M; d = x - mu[fid]
    sd = np.sqrt(np.add.reduceat(d * d, coff) / M); return d / np.maximum(sd[fid], 1e-9)
def prank(x):  # within-feed centred percentile rank (average ties)
    o = np.lexsort((x, fid)); r = np.empty(N); r[o] = np.arange(N) - coff[fid[o]]
    # average ties: group equal values inside a feed
    xs = x[o]; fs = fid[o]; new = np.ones(N, bool); new[1:] = (xs[1:] != xs[:-1]) | (fs[1:] != fs[:-1])
    g = np.cumsum(new) - 1; mean_r = np.bincount(g, weights=r[o]) / np.bincount(g)
    r[o] = mean_r[g]
    return (r + 0.5) / M[fid] - 0.5
NAMES = [f"trd_{a}_{w}" for a in ["kde", "n85", "n95", "skde", "sn95"] for w in ["sday", "s6h"]]
RAW = np.stack([np.nan_to_num(np.asarray(np.load(FEATS / f"val/{n}.npy"), np.float64)) for n in NAMES], 1)
log("columns", NAMES, "share > 0:", np.round((RAW > 0).mean(0), 3).tolist())
lm = (np.log(M) - np.log(M)[fid].mean())[fid]
def build(raw):
    cols = [raw / np.maximum(raw.std(0), 1e-12)] + [np.stack([prank(raw[:, j]) for j in range(raw.shape[1])], 1) * np.sqrt(12.0)]
    return np.concatenate(cols, 1)
def control(b):
    C = np.stack([b, b * lm, cold, cold * b, cold * lm], 1); C[:, 1:] /= C[:, 1:].std(0); return C
def fit(F, feeds, lam):
    rows = feeds[fid]; Fr = F[rows]; yr = y[rows]
    Mr = M[feeds]; offr = np.concatenate([[0], np.cumsum(Mr)[:-1]]); fr = np.repeat(np.arange(len(Mr)), Mr); npr = npos[feeds]
    keep = npr > 0
    def obj(th):
        s = Fr @ th
        mx = np.maximum.reduceat(s, offr); e = np.exp(s - mx[fr]); den = np.add.reduceat(e, offr)
        loss = -(yr * s).sum() + (npr * (mx + np.log(den)))[keep].sum() + 0.5 * lam * (th[1:] ** 2).sum()
        r = -yr + (npr * keep)[fr] * e / den[fr]
        g = Fr.T @ r; g[1:] += lam * th[1:]
        return loss, g
    th0 = np.zeros(F.shape[1]); th0[0] = 1.0
    o = minimize(obj, th0, jac=True, method="L-BFGS-B", options=dict(maxiter=300))
    return o.x, o.fun
def nested(Fc, Fa, seed):
    rng = np.random.default_rng(seed)
    grp = np.where(user == 0, -1 - np.arange(nF), user)
    ug, gi = np.unique(grp, return_inverse=True); perm = rng.permutation(len(ug)); g2 = perm[gi]
    sc = np.zeros(N); sa = np.zeros(N)
    for trf, vaf in GroupKFold(n_splits=5).split(np.arange(nF), groups=g2):
        mtr = np.zeros(nF, bool); mtr[trf] = True; mva = ~mtr; rv = mva[fid]
        tc, _ = fit(Fc, mtr, 10.0); ta, _ = fit(Fa, mtr, 10.0)
        sc[rv] = Fc[rv] @ tc; sa[rv] = Fa[rv] @ ta
    return ndcg_per_feed(sc, y, coff, M), ndcg_per_feed(sa, y, coff, M)
sys.path.insert(0, str(ROOT / "eda/errors/segments"))
import pmix
W = pmix.weights()
def wm(d, w):
    m = np.average(d, weights=w); neff = w.sum() ** 2 / (w ** 2).sum(); return m, np.sqrt(np.average((d - m) ** 2, weights=w) / neff)
zf = feed_z
res = {}
X = build(RAW)
for pn, a, b_ in [("F", "ens_FD16P20kd_ft_val.npy", "ens_FD16P20kd_to_val.npy"), ("W", "ens_W16kd_ft_val.npy", "ens_W16kd_to_val.npy")]:
    b = zf(0.5 * zf(np.load(ROOT / "preds" / a).astype(np.float64)) + 0.5 * zf(np.load(ROOT / "preds" / b_).astype(np.float64)))
    base = ndcg_per_feed(b, y, coff, M)
    Fc = control(b); Fa = np.concatenate([Fc, X], 1)
    allf = np.ones(nF, bool)
    _, l0 = fit(Fc, allf, 0.0); ta, l1 = fit(Fa, allf, 0.0)
    lr = 2 * (l0 - l1)
    log(f"[A8 pool {pn}] pool NDCG {base.mean():.5f}; in-sample LR arm vs control {lr:.2f} df 20 p {chi2.sf(max(lr, 0), 20):.4g}")
    ds = []; dc = []
    for seed in (0, 1, 2):
        nc, na = nested(Fc, Fa, seed); ds.append(na - nc); dc.append(nc - base)
        log(f"[A8 pool {pn}] fold seed {seed}: arm - control {ds[-1].mean():+.5f}; control - pool {dc[-1].mean():+.5f}")
    d = np.mean(ds, 0); se = d.std() / np.sqrt(nF)
    mp, sp = wm(d, W["public"]); mr, sr = wm(d, W["private"])
    log(f"[A8 pool {pn}] NESTED arm - control {d.mean():+.5f} ± {se:.5f} (z {d.mean() / se:+.2f}) | pmix public {mp:+.5f}±{sp:.5f} private {mr:+.5f}±{sr:.5f} | control - pool {np.mean(dc):+.5f}")
    res[pn] = dict(gain=float(d.mean()), se=float(se), lr=float(lr), pub=float(mp), priv=float(mr), control_minus_pool=float(np.mean(dc)))
    if pn == "F":
        np.save(HERE / "trd_delta_F.npy", d)
        pl_ = []
        for sd in range(10):
            rng = np.random.default_rng(sd)
            o = np.lexsort((rng.random(N), cold, fid)); src = np.lexsort((np.arange(N), cold, fid))
            Rp = np.empty_like(RAW); Rp[src] = RAW[o]
            Fp = np.concatenate([Fc, build(Rp)], 1)
            nc, na = nested(Fc, Fp, 0)
            pl_.append(float((na - nc).mean()))
            log(f"[A8 placebo {sd}] {pl_[-1]:+.5f}")
        log(f"[A8 pool F] placebo (fold seed 0, 10 draws): mean {np.mean(pl_):+.5f} sd {np.std(pl_):.5f} min {np.min(pl_):+.5f} max {np.max(pl_):+.5f}; real fold seed 0 {ds[0].mean():+.5f}")
        res["placebo"] = pl_; res["real_seed0"] = float(ds[0].mean())
(HERE / "trd_read.json").write_text(json.dumps(res, indent=1))
log("DONE")
