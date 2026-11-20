"""hzcv screen (b): incremental value over the current NN mix on val (pre-declared, no tuning).
Folds: EXACTLY nn/finetune.py:180-183: GroupKFold(n_splits=5).split(np.arange(n_feeds), groups=val user_id).
Base b = within-feed z of 0.5 z(nn_final4_val_oof) + 0.5 z(nn_final4_trainonly_val).
Residual models fitted on 4 folds, scored on the held-out fold (setup copied from the featrev inventory
c07_resid_new.py):
  GBM : LightGBM lambdarank, init_score = beta_k * b (beta_k = listwise-logit slope of b on the training folds),
        inputs [b, x, z(x) ...], 60 rounds, 7 leaves, min_data_in_leaf 2000, l2 50; control = same on [b] only.
  LIN : listwise softmax s = beta*b + sum_j (w1 z(x_j) + w2 (pctrank(x_j) - 0.5)); control s = beta*b.
Groups: ALL6 = seq_cvlag_{invd20,pop,norm} (== seq_cv_* on val) + hzcv_{expo_slog,censored,stale_d};
        OFF3 = the 3 hzcv offsets only (the NN already gets seq_cv_* in both stages).
Paired per-feed delta vs control with SE over 14,423 val feeds, plus Tue/Thu and M<50/M>=50 splits.
"""
import sys, json, time
from pathlib import Path
import numpy as np, numba as nb, lightgbm as lgb, polars as pl
from scipy.optimize import minimize
from sklearn.model_selection import GroupKFold

sys.path.insert(0, "<repo>/src")
from metrics import ndcg_per_feed  # noqa: E402

ROOT = Path("<repo>")
OUT = Path("/tmp/scratchpad/featrev/hzcv")
T0 = time.time()
THR = 16
f = pl.read_parquet(ROOT / "data/flat/val/feeds.parquet")
M = f["M"].to_numpy().astype(np.int64)
off = f["cand_off"].to_numpy().astype(np.int64)
user = f["user_id"].to_numpy()
y = np.load(ROOT / "data/flat/val/cand_target.npy").astype(np.float64)
nF = len(M)
fid = np.repeat(np.arange(nF), M)
npos_f = np.bincount(fid, weights=y, minlength=nF)
day = ((f["timestamp"].to_numpy().astype(np.int64) + 10800) // 86400)


@nb.njit(cache=True)
def feed_z(x, off, M):
    out = np.zeros_like(x)
    for i in range(len(M)):
        o = off[i]; m = M[i]
        v = x[o:o + m]
        mu = v.mean(); sd = v.std()
        if sd > 1e-12:
            for k in range(m):
                out[o + k] = (v[k] - mu) / sd
    return out


@nb.njit(cache=True)
def pctrank(x, off, M):
    out = np.zeros_like(x)
    for f_ in range(len(M)):
        o = off[f_]; m = M[f_]
        if m <= 1:
            continue
        v = x[o:o + m]
        order = np.argsort(v)
        i = 0
        while i < m:
            j = i
            while j + 1 < m and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = 0.5 * (i + j) / (m - 1) - 0.5
            for k in range(i, j + 1):
                out[o + order[k]] = avg
            i = j + 1
    return out


P = ROOT / "preds"
zz = lambda a: feed_z(np.asarray(a, np.float64), off, M)
b = zz(0.5 * zz(np.load(P / "nn_final4_val_oof.npy")) + 0.5 * zz(np.load(P / "nn_final4_trainonly_val.npy")))
folds = list(GroupKFold(n_splits=5).split(np.arange(nF), groups=user))
rowfold = np.empty(len(y), np.int64)
for k, (trf, vaf) in enumerate(folds):
    rowfold[np.isin(fid, vaf)] = k


def nll(theta, G, rows):
    s = G @ theta
    ff = fid[rows]
    e = np.exp(np.clip(s, -50, 20.0))
    Z = np.bincount(ff, weights=e, minlength=nF)
    ok = Z > 0
    lZ = np.log(np.where(ok, Z, 1.0))
    yy = y[rows]
    L = -(yy * s).sum() + (npos_f * lZ).sum()
    p = e / np.where(ok, Z, 1.0)[ff]
    w = npos_f[ff] * p - yy
    return L, G.T @ w


def fit_lin(G, tr):
    th0 = np.zeros(G.shape[1]); th0[0] = 0.94
    rows = np.flatnonzero(tr)
    r = minimize(lambda t: nll(t, G[tr], rows), th0, jac=True, method="L-BFGS-B", options=dict(maxiter=200))
    return r.x


def cv_lin(G):
    pred = np.empty(len(y))
    for k in range(5):
        tr = rowfold != k
        pred[~tr] = G[~tr] @ fit_lin(G, tr)
    return pred


betas = [fit_lin(b[:, None], rowfold != k)[0] for k in range(5)]
PARAMS = dict(objective="lambdarank", metric="ndcg", learning_rate=0.05, num_leaves=7, min_data_in_leaf=2000,
              lambda_l2=50.0, feature_fraction=1.0, bagging_fraction=1.0, lambdarank_truncation_level=1000,
              max_bin=63, verbose=-1, num_threads=THR, deterministic=True, force_row_wise=True, seed=0)


def cv_gbm(F):
    pred = np.empty(len(y))
    for k, (trf, vaf) in enumerate(folds):
        tr = rowfold != k
        init = betas[k] * b
        ds = lgb.Dataset(F[tr], y[tr], group=M[np.sort(trf)], init_score=init[tr])
        bst = lgb.train(PARAMS, ds, 60)
        pred[~tr] = init[~tr] + bst.predict(F[~tr])
    return pred


def nd(s):
    return ndcg_per_feed(s, y, off, M)


def rep(name, d):
    se = d.std() / np.sqrt(len(d))
    out = dict(delta=float(d.mean()), se=float(se))
    line = f"{name:10s} {d.mean():+.5f} ({se:.5f}) z {d.mean()/se:+.2f}"
    for sn, m in (("tue", day <= day.min() + 1), ("thu", day >= day.min() + 2), ("M<50", M < 50), ("M>=50", M >= 50)):
        s2 = d[m].std() / np.sqrt(m.sum())
        out[sn] = dict(delta=float(d[m].mean()), se=float(s2))
        line += f" | {sn} {d[m].mean():+.5f} ({s2:.5f})"
    print(line, flush=True)
    return out


ALL6 = ["seq_cvlag_invd20", "seq_cvlag_pop", "seq_cvlag_norm", "hzcv_expo_slog", "hzcv_censored", "hzcv_stale_d"]
OFF3 = ALL6[3:]
X = {n: np.load(ROOT / "data/feats_new/val" / f"{n}.npy").astype(np.float64) for n in ALL6}
assert not any(np.isnan(v).any() for v in X.values())
c_lin = nd(cv_lin(b[:, None]))
c_gbm = nd(cv_gbm(b[:, None].astype(np.float32)))
print(f"control: base {nd(b).mean():.5f} lin {c_lin.mean():.5f} gbm {c_gbm.mean():.5f} betas {np.round(betas, 3)}",
      flush=True)
res = {}
for gname, names in (("ALL6", ALL6), ("OFF3", OFF3)):
    gl, fl = [b], [b]
    for n in names:
        x = X[n]
        fl.append(x)
        if n != "hzcv_stale_d":  # feed-constant: z and rank are identically 0
            xz = np.clip(feed_z(x, off, M), -5, 5)
            fl.append(xz)
            gl += [xz, pctrank(x, off, M)]
        else:
            gl.append(x - 0.35)  # feed-level value; centred constant (cancels in the softmax, kept for symmetry)
    d_lin = nd(cv_lin(np.stack(gl, 1))) - c_lin
    d_gbm = nd(cv_gbm(np.stack(fl, 1).astype(np.float32))) - c_gbm
    res[gname] = {"lin": rep(gname + " lin", d_lin), "gbm": rep(gname + " gbm", d_gbm)}
(OUT / "resid_summary.json").write_text(json.dumps(res, indent=1))
print("done", f"{time.time()-T0:.0f}s")
