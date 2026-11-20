"""Harness for the forensics re-check (CPU only, 12 threads).
load_val() / load_hold(): base score (W8), labels, feed meta.
resid_cv(): LightGBM lambdarank RESIDUAL on top of the base (init_score = a * z(base)), user-split 2-fold;
            ALL tuning (scale a, tree config, number of rounds incl. 0 = 'leave the base alone') is done inside the fit half
            with an inner user-grouped 3-fold; the other half is touched once.
"""
import os
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "POLARS_MAX_THREADS", "NUMBA_NUM_THREADS"): os.environ.setdefault(k, "12")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys, time; sys.path.insert(0, "<repo>/src")
import numpy as np, polars as pl, lightgbm as lgb
from pathlib import Path
from common import feed_meta, FLAT, FEATS
from metrics import ndcg_per_feed
R = Path("<repo>"); W = R / "nn_x/w5/preds"; P = R / "preds"; OUT = R / "nn_x/w12/recheck/forensics_skeptic"
W8 = [f"w5_e10_{a}_s{s}" for a in ("kd05w192", "kd05t25w192") for s in (951, 952, 953, 954)]
NT = 12

def zfeed(x, off, M):
    x = np.asarray(x, np.float64); fid = np.repeat(np.arange(len(M)), M)
    mu = np.bincount(fid, x) / M; sd = np.sqrt(np.maximum(np.bincount(fid, x * x) / M - mu * mu, 0)) + 1e-9
    return (x - mu[fid]) / sd[fid]

def rankpct(x, off, M):
    fid = np.repeat(np.arange(len(M)), M); o = np.lexsort((x, fid)); r = np.empty(len(x)); r[o] = np.arange(len(x)) - np.repeat(off, M)
    return r / np.maximum(np.repeat(M, M) - 1, 1)

def se(d): return d.std() / np.sqrt(len(d))
def fmt(d): return f"{d.mean():+.5f} +- {se(d):.5f}"

def load_val():
    f, M, off = feed_meta("val"); y = np.load(FLAT / "val/cand_target.npy").astype(np.float32)
    ft = zfeed(np.load(P / "ens_W8kd_ft_val.npy"), off, M); to = zfeed(np.load(P / "ens_W8kd_to_val.npy"), off, M)
    base = zfeed(0.5 * ft + 0.5 * to, off, M)
    return dict(split="val", f=f, M=M, off=off, y=y, uid=f["user_id"].to_numpy(), ts=f["timestamp"].to_numpy(), H=f["H"].to_numpy(), base=base, ft=ft, to=to,
                rows=None, n=len(M), fid=np.repeat(np.arange(len(M)), M))

def hold_feeds():
    f = pl.read_parquet(FLAT / "train/feeds.parquet"); Mtr = f["M"].to_numpy().astype(np.int64)
    pr = np.load(R / "eda/labels_position/out/train_prandom.npz"); assert (pr["fidx"] == np.arange(f.height)).all()
    p = np.nan_to_num(pr["p_random"].astype(np.float32), nan=0.0); p[Mtr < 50] = 0.0
    cand = np.where((p > 0.9) & (Mtr >= 50))[0]
    hold = np.sort(np.random.default_rng(12345).choice(cand, 30000, replace=False))
    return f, hold

def load_hold():
    f, hold = hold_feeds(); Mh, oh = np.load(W / "holdout/holdout_meta.npy"); Mh = Mh.astype(np.int64); oh = oh.astype(np.int64)
    fh = f[hold]; assert (fh["M"].to_numpy() == Mh).all()
    y = np.load(W / "holdout/holdout_y.npy").astype(np.float32)
    rows = np.concatenate([np.arange(o, o + m) for o, m in zip(fh["cand_off"].to_numpy(), Mh)])
    yt = np.load(FLAT / "train/cand_target.npy", mmap_mode="r"); assert (np.asarray(yt[rows[:200000]]) == y[:200000]).all()
    base = zfeed(np.mean([zfeed(np.load(W / f"holdout/nn_{m}_holdout.npy"), oh, Mh) for m in W8], 0), oh, Mh)
    return dict(split="hold", f=fh, M=Mh, off=oh, y=y, uid=fh["user_id"].to_numpy(), ts=fh["timestamp"].to_numpy(), H=fh["H"].to_numpy(), base=base,
                rows=rows, hold=hold, n=len(Mh), fid=np.repeat(np.arange(len(Mh)), Mh), ftrain=f)

def user_half(uid, seed=0):
    rng = np.random.default_rng(seed); uu = np.unique(uid); h = rng.random(len(uu)) < 0.5
    half = h[np.searchsorted(uu, uid)]; z0 = uid == 0; half[z0] = rng.random(z0.sum()) < 0.5
    return half

def user_folds(uid, k, seed):
    rng = np.random.default_rng(seed); uu = np.unique(uid); g = rng.integers(0, k, len(uu))
    fold = g[np.searchsorted(uu, uid)]; z0 = uid == 0; fold[z0] = rng.integers(0, k, z0.sum())
    return fold

def nested_lin(D, s, ws=(-0.3, -0.2, -0.1, -0.05, -0.02, 0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5), seed=0):
    """the forensics report's protocol (linear, additive), same user split"""
    y, off, M = D["y"], D["off"], D["M"]; p0 = ndcg_per_feed(D["base"], y, off, M)
    Pm = np.stack([ndcg_per_feed(D["base"] + w * s, y, off, M) for w in ws]); half = user_half(D["uid"], seed); d = np.zeros(len(M)); wc = []
    for a in (True, False):
        tr = half == a; k = int(np.argmax(Pm[:, tr].mean(1))); wc.append(ws[k]); d[~tr] = Pm[k, ~tr] - p0[~tr]
    return d, wc

def _sub(D, fmask):
    """row index + new offsets for a feed subset"""
    M, off = D["M"], D["off"]; idx = np.nonzero(fmask)[0]; Ms = M[idx]
    rows = np.concatenate([np.arange(off[i], off[i] + M[i]) for i in idx]); return rows, Ms, np.r_[0, np.cumsum(Ms)[:-1]]

def _fit_scale(z, y, M, off, grid=(0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0)):
    fid = np.repeat(np.arange(len(M)), M); npos = np.bincount(fid, y, len(M)); best = None
    for a in grid:
        s = a * z; mx = np.maximum.reduceat(s, off); e = np.exp(s - mx[fid]); lse = np.log(np.bincount(fid, e, len(M))) + mx
        ll = (np.bincount(fid, y * s, len(M)) - npos * lse)[npos > 0].mean()
        if best is None or ll > best[0]: best = (ll, a)
    return best[1]

CFGS = [dict(num_leaves=4, min_data_in_leaf=1500, learning_rate=0.03), dict(num_leaves=7, min_data_in_leaf=600, learning_rate=0.05),
        dict(num_leaves=15, min_data_in_leaf=300, learning_rate=0.05)]
CKPT = (0, 10, 20, 40, 80, 150, 250)
# v2 (after the v1 unconstrained run, kept as v1_unconstrained_*): monotone +1 constraints on the base columns, so a model without a
# signal cannot re-order a feed (context is constant inside a feed) and every order change has to come from the signal columns.

def _params(cfg, seed, mono=None):
    p = dict(objective="lambdarank", label_gain=[0, 1], lambdarank_truncation_level=400, verbose=-1, num_threads=NT, seed=seed, feature_fraction=0.7,
             bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, min_sum_hessian_in_leaf=1e-3, max_bin=63, **cfg)
    if mono is not None: p.update(monotone_constraints=list(mono), monotone_constraints_method="intermediate", feature_fraction=1.0 if len(mono) < 8 else 0.7)
    return p

def _train(X, y, M, init, cfg, rounds, seed, mono=None):
    ds = lgb.Dataset(X, label=y, group=M, init_score=init, free_raw_data=False)
    return lgb.train(_params(cfg, seed, mono), ds, num_boost_round=rounds)

def resid_cv(D, X, names, tag, seed=0, cfgs=CFGS, ckpt=CKPT, bag=2, log=print, feed_mask=None, inner_k=3, mono="auto"):
    """returns per-feed delta (NaN outside feed_mask), details. X float32 [rows, k]; base D['base'] already per-feed z."""
    y, M, off, uid, base = D["y"], D["M"], D["off"], D["uid"], D["base"]; n = len(M)
    fm = np.ones(n, bool) if feed_mask is None else feed_mask
    if isinstance(mono, str): mono = [1 if nm in ("base_z", "base_rk") else 0 for nm in names]
    half = user_half(uid, seed); p0 = ndcg_per_feed(base, y, off, M); d = np.full(n, np.nan); det = []; imp = np.zeros(X.shape[1]); t0 = time.time()
    newscore = base.copy()
    for a in (True, False):
        fit = (half == a) & fm; ev = (half != a) & fm
        rf, Mf, of = _sub(D, fit); re, Me, oe = _sub(D, ev)
        Xf, yf, zf = X[rf], y[rf], base[rf]; uf = uid[fit]
        sc = _fit_scale(zf, yf, Mf, of)
        # inner user-grouped CV for (cfg, rounds)
        fold = user_folds(uf, inner_k, seed + 17); nf = len(Mf); pf0 = ndcg_per_feed(zf, yf, of, Mf)
        gain = np.zeros((len(cfgs), len(ckpt)))
        for ci, cfg in enumerate(cfgs):
            for k in range(inner_k):
                itr = fold != k; ite = ~itr
                Dn = dict(M=Mf, off=of); r1, M1, o1 = _sub(Dn, itr); r2, M2, o2 = _sub(Dn, ite)
                b = _train(Xf[r1], yf[r1], M1, sc * zf[r1], cfg, max(ckpt), seed, mono)
                for qi, q in enumerate(ckpt):
                    s = sc * zf[r2] + (b.predict(Xf[r2], num_iteration=q, raw_score=True, num_threads=NT) if q else 0.0)
                    gain[ci, qi] += (ndcg_per_feed(s, yf[r2], o2, M2) - pf0[ite]).sum() / nf
        ci, qi = np.unravel_index(np.argmax(gain), gain.shape); q = ckpt[qi]
        if q:
            pr = np.zeros(len(re))
            for bsd in range(bag):
                b = _train(Xf, yf, Mf, sc * zf, cfgs[ci], q, seed + 100 * bsd, mono); pr += b.predict(X[re], raw_score=True, num_threads=NT) / bag
                imp += b.feature_importance("gain")
            s = sc * base[re] + pr
        else:
            s = sc * base[re]
        newscore[re] = s / sc
        d[ev] = ndcg_per_feed(s, y[re], oe, Me) - p0[ev]
        det.append(dict(fit_half=bool(a), scale=sc, cfg=ci, rounds=q, inner_gain=float(gain[ci, qi]), inner_tab=np.round(gain * 1e5).astype(int).tolist(), delta=float(d[ev].mean()), se=float(se(d[ev])), n=int(ev.sum())))
        log(f"  [{tag}] fit half {int(a)}: scale {sc}, cfg {ci}, rounds {q}, inner gain {gain[ci, qi]:+.5f} -> other half {fmt(d[ev])} (n {ev.sum()}) [{time.time() - t0:.0f}s]")
        log(f"      inner table x1e5 (rows cfg, cols rounds {ckpt}): {np.round(gain * 1e5).astype(int).tolist()}")
    dd = d[fm]; top = sorted(zip(imp, names), reverse=True)[:8]
    log(f"  [{tag}] BOTH halves: {fmt(dd)} (z {dd.mean() / max(se(dd), 1e-12):+.2f}) over {fm.sum()} feeds; all-feed mean {np.nansum(d) / n:+.5f}; top gain: " + ", ".join(f"{nm} {v / max(imp.sum(), 1e-9):.2f}" for v, nm in top))
    return d, det, newscore

def ctx_cols(D):
    """label-free feed-level context broadcast to rows + base views"""
    M, off, base = D["M"], D["off"], D["base"]; fid = D["fid"]
    cols = dict(base_z=base, base_rk=rankpct(base, off, M), M=np.repeat(M, M).astype(np.float64), H=np.repeat(D["H"], M).astype(np.float64),
                guest=np.repeat((D["uid"] == 0), M).astype(np.float64), hour=np.repeat(((D["ts"] + 3 * 3600) % 86400) / 3600.0, M))
    return cols

if __name__ == "__main__":
    V = load_val(); print("W8 val", ndcg_per_feed(V["base"], V["y"], V["off"], V["M"]).mean(), "ft", ndcg_per_feed(V["ft"], V["y"], V["off"], V["M"]).mean(), "to", ndcg_per_feed(V["to"], V["y"], V["off"], V["M"]).mean(), "rows", len(V["y"]))
    Hd = load_hold(); print("W8 TO pool holdout", ndcg_per_feed(Hd["base"], Hd["y"], Hd["off"], Hd["M"]).mean(), "rows", len(Hd["y"]), "users", len(np.unique(Hd["uid"])), "guest feeds", (Hd["uid"] == 0).sum())

def nested_bucket(D, s, bucket, ws=(-0.3, -0.2, -0.1, -0.05, -0.02, 0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5), seed=0):
    """interaction test with low variance: one weight PER feed bucket (label-free feed-level partition), each chosen on the fit half.
    returns per-feed delta of the bucketed rule, of the global-weight rule, and the chosen weights"""
    y, off, M = D["y"], D["off"], D["M"]; p0 = ndcg_per_feed(D["base"], y, off, M)
    Pm = np.stack([ndcg_per_feed(D["base"] + w * s, y, off, M) for w in ws]); half = user_half(D["uid"], seed)
    db = np.zeros(len(M)); dg = np.zeros(len(M)); chosen = []
    for a in (True, False):
        tr = half == a; k = int(np.argmax(Pm[:, tr].mean(1))); dg[~tr] = Pm[k, ~tr] - p0[~tr]; cw = []
        for b in np.unique(bucket):
            m = bucket == b; kb = int(np.argmax(Pm[:, tr & m].mean(1))) if (tr & m).sum() else list(ws).index(0); cw.append(ws[kb])
            db[(~tr) & m] = Pm[kb, (~tr) & m] - p0[(~tr) & m]
        chosen.append(cw)
    return db, dg, chosen
