"""critic stale_trd_read: BLOCK 1 of DECLARATION.txt. Residual-lens read of the fresh-density columns (G2) on the STALE-val pool,
on the popularity-light pool and on their 0.8 / 0.2 mixes, with a positive control (fresh counters) and placebos.
Lens copied from nn_x/w12/lit/syn/trd_read.py (same objective, ridge, folds). CPU only, read-only on the project."""
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
cold = (luo == 0).astype(np.float64)
npos = np.add.reduceat(y, coff)
def feed_z(x):
    mu = np.add.reduceat(x, coff) / M; d = x - mu[fid]
    sd = np.sqrt(np.add.reduceat(d * d, coff) / M); return d / np.maximum(sd[fid], 1e-9)
zf = feed_z
def prank(x):
    o = np.lexsort((x, fid)); r = np.empty(N); r[o] = np.arange(N) - coff[fid[o]]
    xs = x[o]; fs = fid[o]; new = np.ones(N, bool); new[1:] = (xs[1:] != xs[:-1]) | (fs[1:] != fs[:-1])
    g = np.cumsum(new) - 1; mean_r = np.bincount(g, weights=r[o]) / np.bincount(g)
    r[o] = mean_r[g]
    return (r + 0.5) / M[fid] - 0.5
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
grp = np.where(user == 0, -1 - np.arange(nF), user)
ug, gi = np.unique(grp, return_inverse=True)
def folds(seed):
    rng = np.random.default_rng(seed); perm = rng.permutation(len(ug)); g2 = perm[gi]
    return list(GroupKFold(n_splits=5).split(np.arange(nF), groups=g2))
FOLDS = {s: folds(s) for s in (0, 1, 2)}
def nested_score(Fx, seed):
    s = np.zeros(N)
    for trf, vaf in FOLDS[seed]:
        mtr = np.zeros(nF, bool); mtr[trf] = True; rv = (~mtr)[fid]
        t, _ = fit(Fx, mtr, 10.0); s[rv] = Fx[rv] @ t
    return ndcg_per_feed(s, y, coff, M)
import pmix
W = pmix.weights()
def wm(d, w):
    m = np.average(d, weights=w); neff = w.sum() ** 2 / (w ** 2).sum(); return m, np.sqrt(np.average((d - m) ** 2, weights=w) / neff)

def ld(p): return np.load(p).astype(np.float64)
def pool(paths): return zf(np.mean([zf(ld(p)) for p in paths], 0))
SEEDS = range(951, 959)
MEM = [f"nn_w5_e10_{a}_s{s}" for a in ("kd05w192", "kd05t25w192") for s in SEEDS]
WL = pool([ROOT / "nn_x/w5/preds" / f"{m}@last_trainonly_val.npy" for m in MEM])
S = pool([ROOT / "nn_x/w12/stale/hz2n_review/preds_stale_lt3" / f"{m}@last_staleval.npy" for m in MEM])
P = pool([ROOT / "preds" / f"nn_abl_e10_pop1kd05w192_s{s}_trainonly_val.npy" for s in range(951, 955)])
Fp = zf(0.5 * zf(ld(ROOT / "preds/ens_FD16P20kd_ft_val.npy")) + 0.5 * zf(ld(ROOT / "preds/ens_FD16P20kd_to_val.npy")))
POOLS = {"F": Fp, "WL": WL, "S": S, "P": P, "WP": zf(0.8 * WL + 0.2 * P), "SP": zf(0.8 * S + 0.2 * P)}
w16 = zf(ld(ROOT / "preds/ens_W16kd_to_val.npy"))
log("check: corr(WL, stored ens_W16kd_to_val) per row", round(float(np.corrcoef(WL, w16)[0, 1]), 5))
base = {k: ndcg_per_feed(v, y, coff, M) for k, v in POOLS.items()}
for k in POOLS:
    mr, sr = wm(base[k], W["private"])
    log(f"pool {k}: val NDCG {base[k].mean():.5f} | pmix-private {mr:.5f}")
d = base["S"] - base["WL"]; log(f"stale cost S - WL: {d.mean():+.5f} ± {d.std() / np.sqrt(nF):.5f}")
d = base["SP"] - base["S"]; log(f"admixture under stale base SP - S: {d.mean():+.5f} ± {d.std() / np.sqrt(nF):.5f}")
d = base["WP"] - base["WL"]; log(f"admixture under fresh base WP - WL: {d.mean():+.5f} ± {d.std() / np.sqrt(nF):.5f}")

N10 = [f"trd_{a}_{w}" for a in ["kde", "n85", "n95", "skde", "sn95"] for w in ["sday", "s6h"]]
N4 = ["trd_kde_sday", "trd_skde_sday", "trd_n95_sday", "trd_sn95_sday"]
def raw(names): return np.stack([np.nan_to_num(np.asarray(np.load(FEATS / f"val/{n}.npy"), np.float64)) for n in names], 1)
R10 = raw(N10); R4 = raw(N4)
RPC = np.stack([np.log1p(np.maximum(luo, 0)), np.log1p(np.maximum(np.nan_to_num(np.asarray(np.load(FEATS / "val/seq_cv_pop.npy"), np.float64)), 0))], 1)
ARMS = {"T10": build(R10), "T4": build(R4), "PC": build(RPC)}
res = {}
PLAN = [("F", ["T10"]), ("WL", ["T10", "T4", "PC"]), ("S", ["T10", "T4", "PC"]), ("SP", ["T10", "T4", "PC"]), ("P", ["T10", "T4", "PC"]), ("WP", ["T10", "T4"])]
allf = np.ones(nF, bool)
seed0 = {}
for pn, arms in PLAN:
    Fc = control(POOLS[pn])
    nc = {s: nested_score(Fc, s) for s in (0, 1, 2)}
    _, l0 = fit(Fc, allf, 0.0)
    for an in arms:
        Fa = np.concatenate([Fc, ARMS[an]], 1)
        ds = [nested_score(Fa, s) - nc[s] for s in (0, 1, 2)]
        dd = np.mean(ds, 0); se = dd.std() / np.sqrt(nF)
        mr, sr = wm(dd, W["private"]); mp, sp = wm(dd, W["public"])
        _, l1 = fit(Fa, allf, 0.0); lr = 2 * (l0 - l1); df = ARMS[an].shape[1]
        log(f"[{an} on {pn}] NESTED arm - control {dd.mean():+.5f} ± {se:.5f} (z {dd.mean() / se:+.2f}) | per fold seed {' '.join(f'{x.mean():+.5f}' for x in ds)} | pmix public {mp:+.5f}±{sp:.5f} private {mr:+.5f}±{sr:.5f} | in-sample LR {lr:.1f} df {df} p {chi2.sf(max(lr, 0), df):.3g} | control - pool {np.mean([nc[s].mean() for s in nc]) - base[pn].mean():+.5f}")
        res[f"{an}|{pn}"] = dict(gain=float(dd.mean()), se=float(se), seeds=[float(x.mean()) for x in ds], priv=float(mr), priv_se=float(sr), pub=float(mp), lr=float(lr), df=int(df))
        seed0[(an, pn)] = float(ds[0].mean())
        np.save(HERE / f"delta_{an}_{pn}.npy", dd.astype(np.float32))
    if pn in ("S", "P"):
        pl_ = []
        for sd in range(10):
            rng = np.random.default_rng(sd)
            o = np.lexsort((rng.random(N), cold, fid)); src = np.lexsort((np.arange(N), cold, fid))
            Rp = np.empty_like(R4); Rp[src] = R4[o]
            pl_.append(float((nested_score(np.concatenate([Fc, build(Rp)], 1), 0) - nc[0]).mean()))
        r0 = seed0[("T4", pn)]
        log(f"[T4 on {pn}] placebo (fold seed 0, 10 draws): mean {np.mean(pl_):+.5f} sd {np.std(pl_):.5f} min {np.min(pl_):+.5f} max {np.max(pl_):+.5f} | real fold seed 0 {r0:+.5f} | above {sum(r0 > x for x in pl_)} of 10 | real minus placebo mean {r0 - np.mean(pl_):+.5f}")
        res[f"placebo_T4|{pn}"] = dict(draws=pl_, real_seed0=r0)
(HERE / "stale_trd_read.json").write_text(json.dumps(res, indent=1))
log("DONE")
