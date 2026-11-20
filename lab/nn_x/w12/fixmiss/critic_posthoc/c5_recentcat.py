"""Critic check 5 (CPU): the stream's 'newcat' group = category never in history (6.5% of positives), not the
'outside recent categories' cell (pass 3: 28% of positives outside the last 8 events). Re-run the group-mass test
(Platt calibration + cross-fitted 2-parameter shift, and a feed-descriptor LightGBM) on groups 'not in last 8' and
'not in last 32', with 4 random user splits."""
import os
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS"):
    os.environ[k] = "12"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys; sys.path.insert(0, "<repo>/src")
import numpy as np, polars as pl, lightgbm as lgb
from scipy.optimize import minimize
from common import feed_meta, FLAT
from metrics import ndcg_per_feed
import pmix
P = "<repo>/preds"
fv, M, off = feed_meta("val"); n = len(M); fid = np.repeat(np.arange(n), M)
y = np.load(FLAT / "val/cand_target.npy").astype(np.float64)
def z(x):
    x = np.asarray(x, np.float64); mu = np.bincount(fid, x, n) / M
    sd = np.sqrt(np.maximum(np.bincount(fid, x * x, n) / M - mu ** 2, 0)) + 1e-9
    return (x - mu[fid]) / sd[fid]
s = 0.5 * z(np.load(f"{P}/ens_FD16P20_m04_02kd_ft_val.npy")) + 0.5 * z(np.load(f"{P}/ens_FD16P20_m04_02kd_to_val.npy"))
ev = lambda k: ndcg_per_feed(k, y, off, M); base = ev(s); W = pmix.weights(); big = M >= 50
mx = np.maximum.reduceat(s, off); e = np.exp(s - mx[fid]); q = e / np.bincount(fid, e, n)[fid]
K = np.bincount(fid, y, n)
F = pl.read_parquet("<repo>/lab/nn_x/w12/fixmiss/posthoc/out/valfeat.parquet",
                    columns=["cat_newest_idx", "H", "newest_type", "newest_src", "n_hist_cats32", "hour", "weekday", "guest"])
ci = F["cat_newest_idx"].to_numpy().astype(np.int64)
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
ce = lambda a, p: -(a * np.log(np.clip(p, 1e-9, 1)) + (1 - a) * np.log(np.clip(1 - p, 1e-9, 1))).mean()
fx = np.column_stack([F[c].to_numpy()[off].astype(float) for c in ["H", "newest_type", "newest_src", "n_hist_cats32", "hour", "weekday", "guest"]] + [np.log(M)])
u = fv["user_id"].to_numpy().astype(np.int64)
f = lambda d, w=None: "{:+.5f}+-{:.5f}".format(*pmix.wmean_se(d, w))
for gname, g in (("notlast8", (ci < 0) | (ci >= 8)), ("notlast32", (ci < 0) | (ci >= 32))):
    g = g.astype(float)
    mf = np.bincount(fid, q * g, n); af = np.bincount(fid, y * g, n) / np.maximum(K, 1); share = np.bincount(fid, g, n) / M
    ok = (share > 0) & (share < 1) & (K > 0)
    print(f"[{gname}] positives in group {(y * g).sum() / y.sum():.3f}; feeds with both {ok.mean():.3f}; model mass {mf[ok].mean():.4f} vs actual {af[ok].mean():.4f}")
    rng = np.random.default_rng(5)
    for form in ("platt", "lgbm"):
        ds = []
        for r in range(4):
            uu = np.unique(u); hv = dict(zip(uu, rng.integers(0, 2, len(uu)))); half = np.array([hv[x] for x in u])
            half[u == 0] = rng.integers(0, 2, (u == 0).sum())
            out = s.copy(); ces = []
            for h in (0, 1):
                A = (half == h) & ok; B = (half != h) & ok
                if form == "platt":
                    rr = minimize(lambda p: ce(af[A], 1 / (1 + np.exp(-(p[0] + p[1] * lg(mf[A]))))), [0, 1], method="Nelder-Mead")
                    ah = 1 / (1 + np.exp(-(rr.x[0] + rr.x[1] * lg(mf[B]))))
                else:
                    X = np.column_stack([lg(mf), share, fx])
                    b = lgb.train(dict(objective="cross_entropy", learning_rate=0.03, num_leaves=7, min_data_in_leaf=300, lambda_l2=10.0, verbose=-1, num_threads=12, seed=3),
                                  lgb.Dataset(X[A], af[A]), num_boost_round=150)
                    ah = b.predict(X[B])
                ces.append((ce(af[B], ah), ce(af[B], mf[B])))
                delta = np.zeros(n); delta[B] = lg(ah) - lg(mf[B])
                rows = half[fid] != h; out[rows] = s[rows] + delta[fid][rows] * g[rows]
            d = ev(out) - base; ds.append(d)
            if r == 0: print(f"   {form} split0 OOF CE model {np.mean([c[0] for c in ces]):.5f} vs own mass {np.mean([c[1] for c in ces]):.5f}")
            print(f"   {gname} {form} split{r}: d {f(d)} | pub {f(d, W['public'])} | priv {f(d, W['private'])} | M<50 {f(d[~big])} | M>=50 {f(d[big])}")
        dm = np.mean(ds, 0); print(f"   {gname} {form} mean over 4 splits: d {f(dm)}")
