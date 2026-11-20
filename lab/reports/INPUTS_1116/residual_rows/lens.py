"""INVENTORY B, uniform lens (CPU only, read-only on every project file).

One read for every candidate column, on the CURRENT held-out wide pool, on val AND on the in-period holdout:
  conditional logit of the click on [z(pool), column]  (listwise softmax per feed, every positive one observation)
    LR      = 2 * (loglik with the column - loglik of the pool alone)        (in-sample likelihood ratio, 1 df per column)
    zrob    = coefficient / user-clustered sandwich SE                        (robust to model misspecification)
    nats    = loglik gain per feed
    nested  = NDCG(pool + w * column) - NDCG(pool), coefficients fitted on 4 user-grouped folds, read on the 5th
Forms: z = within-feed z over present rows (log1p first for counts with max > 20), NaN -> 0, clipped at +-5;
       rk = within-feed percentile rank - 0.5 (NaN -> 0);  nan = within-feed z of the missing flag (only if 0.5% < NaN share < 99.5%).
The lens is calibrated in s03 on real positive controls (input blocks the ranker reads, offered to pools trained without them)
and on label-free placebo columns (N(0,1) and within-feed permutations of real columns).
"""
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS"):
    os.environ[k] = "12"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["NUMBA_CACHE_DIR"] = "<repo>/reports/INPUTS_1116/residual_rows/out/numba_cache"
sys.dont_write_bytecode = True
sys.path.insert(0, "<repo>/src")
import numpy as np, numba as nb, polars as pl
from pathlib import Path

ROOT = Path("<repo>"); FLAT = ROOT / "data/flat"; W5 = ROOT / "nn_x/w5/preds"; AB = ROOT / "nn_x/abl/preds"; P = ROOT / "preds"
OUT = ROOT / "reports/INPUTS_1116/residual_rows/out"
W16 = [f"w5_e10_{a}_s{s}" for a in ("kd05w192", "kd05t25w192") for s in range(951, 959)]


# ---------------------------------------------------------------- numba kernels
@nb.njit(parallel=True, cache=True)
def feed_z(x, off, M, clip):
    """within-feed z over present rows; NaN -> 0; constant feeds -> 0"""
    out = np.zeros(len(x), np.float32)
    for f in nb.prange(len(M)):
        o = off[f]; m = M[f]; s = 0.0; s2 = 0.0; c = 0
        for k in range(m):
            v = x[o + k]
            if v == v:
                s += v; s2 += v * v; c += 1
        if c < 2:
            continue
        mu = s / c; var = s2 / c - mu * mu
        if var <= 1e-18:
            continue
        sd = np.sqrt(var)
        for k in range(m):
            v = x[o + k]
            if v == v:
                z = (v - mu) / sd
                if z > clip: z = clip
                if z < -clip: z = -clip
                out[o + k] = z
    return out


@nb.njit(parallel=True, cache=True)
def feed_rk(x, off, M):
    """within-feed percentile rank (average ties) of present rows minus 0.5; NaN -> 0"""
    out = np.zeros(len(x), np.float32)
    for f in nb.prange(len(M)):
        o = off[f]; m = M[f]
        v = x[o:o + m].astype(np.float64)
        ok = v == v; c = ok.sum()
        if c < 2:
            continue
        idx = np.nonzero(ok)[0]; vv = v[idx]; order = np.argsort(vv)
        r = np.empty(c); i = 0
        while i < c:
            j = i
            while j + 1 < c and vv[order[j + 1]] == vv[order[i]]:
                j += 1
            a = 0.5 * (i + j)
            for t in range(i, j + 1):
                r[order[t]] = a
            i = j + 1
        for t in range(c):
            out[o + idx[t]] = r[t] / (c - 1) - 0.5
    return out


@nb.njit(parallel=True, cache=True)
def clogit(X, y, off, M, theta, use):
    """per-feed loglik, score (gradient) and information of the conditional logit. X [rows, K] float32, use = feed mask."""
    n = len(M); K = X.shape[1]
    ll = np.zeros(n); G = np.zeros((n, K)); H = np.zeros((n, K, K))
    for f in nb.prange(n):
        if not use[f]:
            continue
        o = off[f]; m = M[f]; npos = 0.0
        for i in range(m):
            npos += y[o + i]
        if npos == 0.0:
            continue
        s = np.empty(m); mx = -1e30
        for i in range(m):
            v = 0.0
            for k in range(K):
                v += X[o + i, k] * theta[k]
            s[i] = v
            if v > mx: mx = v
        Z = 0.0
        for i in range(m):
            s[i] = np.exp(s[i] - mx); Z += s[i]
        mu = np.zeros(K); S2 = np.zeros((K, K)); l = 0.0
        for i in range(m):
            p = s[i] / Z; yi = y[o + i]
            if yi > 0:
                l += yi * np.log(p)
            for k in range(K):
                xk = X[o + i, k]
                mu[k] += p * xk
                if yi > 0:
                    G[f, k] += yi * xk
                for q in range(k, K):
                    S2[k, q] += p * xk * X[o + i, q]
        ll[f] = l
        for k in range(K):
            G[f, k] -= npos * mu[k]
            for q in range(k, K):
                h = npos * (S2[k, q] - mu[k] * mu[q]); H[f, k, q] = h; H[f, q, k] = h
    return ll, G, H


@nb.njit(parallel=True, cache=True)
def _ndcg(s, y, off, M):
    n = len(M); out = np.ones(n, np.float64)
    for f in nb.prange(n):
        o, m = off[f], M[f]
        sv = s[o:o + m]; yv = y[o:o + m]
        npos = 0
        for k in range(m):
            if yv[k] > 0: npos += 1
        if npos == 0: continue
        order = np.argsort(-sv.astype(np.float64) + 0.0, kind='mergesort')
        dcg = 0.0; i = 0
        while i < m:
            j = i
            while j + 1 < m and sv[order[j + 1]] == sv[order[i]]: j += 1
            if j == i:
                if yv[order[i]] > 0: dcg += 1.0 / np.log2(i + 2.0)
            else:
                cp = 0
                for k in range(i, j + 1):
                    if yv[order[k]] > 0: cp += 1
                for k in range(j - cp + 1, j + 1): dcg += 1.0 / np.log2(k + 2.0)
            i = j + 1
        idcg = 0.0
        for k in range(npos): idcg += 1.0 / np.log2(k + 2.0)
        out[f] = dcg / idcg
    return out


def ndcg(s, D):
    """copy of nn_x/w12/recheck/pool_composition/pc_common._ndcg (pessimistic ties = src/metrics.ndcg_per_feed)"""
    return _ndcg(np.ascontiguousarray(s, np.float32), D["y"], D["off"], D["M"])


# ---------------------------------------------------------------- data
def hold_feeds():
    """the 30,000 holdout feeds of nn/train.py --holdout 30000 (same construction as forensics_skeptic/hx.py hold_feeds)"""
    f = pl.read_parquet(FLAT / "train/feeds.parquet"); Mtr = f["M"].to_numpy().astype(np.int64)
    pr = np.load(ROOT / "eda/labels_position/out/train_prandom.npz"); assert (pr["fidx"] == np.arange(f.height)).all()
    p = np.nan_to_num(pr["p_random"].astype(np.float32), nan=0.0); p[Mtr < 50] = 0.0
    cand = np.where((p > 0.9) & (Mtr >= 50))[0]
    hold = np.sort(np.random.default_rng(12345).choice(cand, 30000, replace=False))
    return f, hold


def zpool(files, off, M):
    return feed_z(np.mean([feed_z(np.load(f).astype(np.float32), off, M, 1e9) for f in files], 0).astype(np.float32), off, M, 1e9)


def folds_by_user(uid, k=5, seed=20261116):
    rng = np.random.default_rng(seed); uu = np.unique(uid); g = rng.integers(0, k, len(uu))
    fold = g[np.searchsorted(uu, uid)]; z0 = uid == 0; fold[z0] = rng.integers(0, k, int(z0.sum()))
    return fold


def cluster_ids(uid):
    c = uid.astype(np.int64).copy(); z0 = np.nonzero(uid == 0)[0]; c[z0] = -1 - np.arange(len(z0))
    _, inv = np.unique(c, return_inverse=True)
    return inv


def load_split(sp, base="TO16"):
    """sp 'val' | 'hold'. base: 'TO16' = train-only pool of the 16 held-out wide students; 'FILE' (val only) = FD16P20 (0.5 FT + 0.5 TO);
    or a list of prediction files."""
    if sp == "val":
        f = pl.read_parquet(FLAT / "val/feeds.parquet"); M = f["M"].to_numpy().astype(np.int64); off = np.concatenate([[0], np.cumsum(M)[:-1]]).astype(np.int64)
        assert (f["cand_off"].to_numpy() == off).all()
        y = np.load(FLAT / "val/cand_target.npy").astype(np.float32); rows = None; uid = f["user_id"].to_numpy()
    else:
        ft, hold = hold_feeds(); Mh, oh = np.load(W5 / "holdout/holdout_meta.npy"); M = Mh.astype(np.int64); off = oh.astype(np.int64)
        f = ft[hold]; assert (f["M"].to_numpy() == M).all()
        y = np.load(W5 / "holdout/holdout_y.npy").astype(np.float32); uid = f["user_id"].to_numpy()
        co = f["cand_off"].to_numpy().astype(np.int64)
        rows = np.repeat(co - off, M) + np.arange(int(M.sum()), dtype=np.int64)
        yt = np.load(FLAT / "train/cand_target.npy", mmap_mode="r"); assert (np.asarray(yt[rows[:300000]]) == y[:300000]).all()
    D = dict(split=sp, M=M, off=off, y=y, rows=rows, uid=uid, n=len(M), ts=f["timestamp"].to_numpy(), fold=folds_by_user(uid), clus=cluster_ids(uid))
    D["base"] = get_base(D, base)
    D["ndcg0"] = ndcg(D["base"], D)
    return D


def get_base(D, base):
    sp, off, M = D["split"], D["off"], D["M"]
    if isinstance(base, (list, tuple)):
        return zpool(base, off, M)
    if base == "TO16":
        fl = [W5 / (f"nn_{v}_trainonly_val.npy" if sp == "val" else f"holdout/nn_{v}_holdout.npy") for v in W16]
        return zpool(fl, off, M)
    if base == "FILE":
        assert sp == "val"
        ft = feed_z(np.load(P / "ens_FD16P20kd_ft_val.npy").astype(np.float32), off, M, 1e9); to = feed_z(np.load(P / "ens_FD16P20kd_to_val.npy").astype(np.float32), off, M, 1e9)
        return feed_z((0.5 * ft + 0.5 * to).astype(np.float32), off, M, 1e9)
    raise ValueError(base)


# ---------------------------------------------------------------- transforms
def forms(x, D, log=None):
    """x float array aligned with the split's candidate rows. Returns dict of forms and stats."""
    x = np.array(x, np.float64); x[~np.isfinite(x)] = np.nan  # copy: sources may be read-only memmaps
    nanshare = float(np.isnan(x).mean())
    if log is None:
        with np.errstate(all="ignore"):
            log = bool(np.nanmin(x) >= 0 and np.nanmax(x) > 20) if nanshare < 1 else False
    if log: x = np.log1p(x)
    z = feed_z(x, D["off"], D["M"], 5.0); rk = feed_rk(x, D["off"], D["M"])
    out = dict(z=z, rk=rk, nanshare=nanshare, log=log, varshare=float((np.maximum.reduceat(np.abs(z), D["off"]) > 0).mean()))
    if 0.005 < nanshare < 0.995:
        out["nan"] = feed_z(np.isnan(x).astype(np.float64), D["off"], D["M"], 5.0)
    return out


# ---------------------------------------------------------------- fits
def fit(D, cols, use=None, theta0=None, iters=30):
    """Newton fit of the conditional logit on [base] + cols. Returns theta, loglik, per-feed ll / G, H."""
    X = np.ascontiguousarray(np.stack([D["base"]] + list(cols), 1).astype(np.float32)); K = X.shape[1]
    use = np.ones(D["n"], np.bool_) if use is None else use
    th = np.zeros(K) if theta0 is None else np.asarray(theta0, np.float64).copy()
    if theta0 is None: th[0] = 1.0
    ll, G, H = clogit(X, D["y"], D["off"], D["M"], th, use); L = ll.sum()
    for _ in range(iters):
        g = G.sum(0); Hs = H.sum(0) + 1e-9 * np.eye(K)
        step = np.linalg.solve(Hs, g); t = 1.0
        while True:
            th2 = th + t * step; ll2, G2, H2 = clogit(X, D["y"], D["off"], D["M"], th2, use); L2 = ll2.sum()
            if L2 >= L - 1e-9 or t < 1e-4: break
            t *= 0.5
        done = abs(L2 - L) < 1e-7 * max(1.0, abs(L))
        th, ll, G, H, L = th2, ll2, G2, H2, L2
        if done: break
    return dict(theta=th, L=L, ll=ll, G=G, H=H.sum(0), X=X)


def robust_z(D, r):
    K = len(r["theta"]); Hi = np.linalg.inv(r["H"] + 1e-9 * np.eye(K))
    Gc = np.zeros((D["clus"].max() + 1, K)); np.add.at(Gc, D["clus"], r["G"])
    V = Hi @ (Gc.T @ Gc) @ Hi
    return r["theta"] / np.sqrt(np.maximum(np.diag(V), 1e-30)), r["theta"] / np.sqrt(np.maximum(np.diag(Hi), 1e-30))


def base_fit(D):
    if "fit0" not in D:
        D["fit0"] = fit(D, [])
        D["fit0_fold"] = [fit(D, [], use=D["fold"] != k, theta0=D["fit0"]["theta"]) for k in range(5)]
    return D["fit0"]


def nested(D, cols, theta_full):
    """5 user-grouped folds: coefficients on 4, NDCG read on the 5th, against the pool alone."""
    d = np.zeros(D["n"])
    X = np.ascontiguousarray(np.stack([D["base"]] + list(cols), 1).astype(np.float32))
    for k in range(5):
        tr = D["fold"] != k
        r = fit(D, cols, use=tr, theta0=theta_full, iters=8)
        th = r["theta"]
        if th[0] <= 0:  # degenerate: keep the base
            continue
        s = (X @ th.astype(np.float32)).astype(np.float32)
        e = ndcg(s, D); te = ~tr; d[te] = e[te] - D["ndcg0"][te]
    return d


def read_column(D, x=None, F=None, do_nested=True, do_rk=True):
    """the full read of one column; returns a flat dict"""
    f0 = base_fit(D)
    if F is None: F = forms(x, D)
    res = dict(nanshare=F["nanshare"], log=F["log"], varshare=F["varshare"])
    if F["varshare"] < 0.01:
        res["skip"] = "constant within feeds"; return res
    r = fit(D, [F["z"]], theta0=[f0["theta"][0], 0.0]); zr, zm = robust_z(D, r)
    res.update(LR_z=2 * (r["L"] - f0["L"]), coef_z=r["theta"][1], zrob_z=zr[1], zmod_z=zm[1], nats_z=(r["L"] - f0["L"]) / D["n"])
    # per-feed loglik gain -> paired SE of the nats gain
    dl = r["ll"] - f0["ll"]; res["nats_se"] = dl.std() / np.sqrt(D["n"])
    if do_rk:
        r2 = fit(D, [F["rk"]], theta0=[f0["theta"][0], 0.0]); zr2, _ = robust_z(D, r2)
        res.update(LR_rk=2 * (r2["L"] - f0["L"]), coef_rk=r2["theta"][1], zrob_rk=zr2[1])
    if "nan" in F:
        r3 = fit(D, [F["z"], F["nan"]], theta0=[r["theta"][0], r["theta"][1], 0.0]); zr3, _ = robust_z(D, r3)
        res.update(LR_znan=2 * (r3["L"] - f0["L"]), coef_nan=r3["theta"][2], zrob_nan=zr3[2])
    if do_nested:
        d = nested(D, [F["z"]], r["theta"]); res.update(nest=d.mean(), nest_se=d.std() / np.sqrt(len(d)))
        if "wpriv" in D:
            w = D["wpriv"]; mm = np.average(d, weights=w); neff = w.sum() ** 2 / (w ** 2).sum()
            res.update(nest_pmix=mm, nest_pmix_se=np.sqrt(np.average((d - mm) ** 2, weights=w) / neff))
    return res


def fmt(d):
    return f"{d.mean():+.5f} +- {d.std() / np.sqrt(len(d)):.5f}"
