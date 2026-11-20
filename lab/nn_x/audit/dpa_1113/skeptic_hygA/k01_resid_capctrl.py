"""Input pipeline audit, step 3: nested residual tests on G3 (val, user-grouped 5-fold) + holdout check.

Base val  b  = z(0.5 z(ens_G3_ft_val) + 0.5 z(ens_G3_to_val))  (G3 submission score, val 0.40032)
Base hold bh = member-equal per-feed z-mean of the 20 G3 members' holdout preds (train-only; no FT on the holdout)
LIN: listwise softmax  s = beta b + sum_k w_k g_k  (fold-fit); control s = beta b.  Holdout: weights fit on FULL val.
GBM: LightGBM lambdarank on [b, features] with init_score beta b (60 rounds, 7 leaves, min_data 2000, l2 50);
     control = same on [b].  Holdout: model fit on FULL val, init beta_full * bh.
Reported: paired per-feed delta vs control (SE over feeds), slices, private-mix (src/pmix weights), holdout delta + z.
CPU only.  python s03_resid.py <test names comma-separated | all>
"""
import os, sys, json, time
for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS"]:
    os.environ[k] = "12"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pathlib import Path
import numpy as _np0, torch as _t0
ROOT = Path("<repo>"); BOUND = 1790497920
OUT = Path("/tmp/scratchpad/dpa/pipeline/out")  # read-only inputs
MYOUT = Path("/tmp/scratchpad/dpa/skeptic_hygA/out")
_CK = _t0.load(ROOT / "nn_x/r6/ckpt/r6_cd_m02_r101.pt", map_location="cpu", weights_only=False)
XT = _CK["cfg"]["xt_names"]; NAMES = None
def load(sub):
    X = _np0.load(OUT / f"{sub}_X.npy", mmap_mode="r"); R = _np0.load(OUT / f"{sub}_raw.npy", mmap_mode="r")
    return X, R, dict(_np0.load(OUT / f"{sub}_meta.npz"))

import numpy as np, numba as nb, polars as pl
import lightgbm as lgb
from scipy.optimize import minimize
from sklearn.model_selection import GroupKFold
sys.path.insert(0, "<repo>/src")
from metrics import ndcg_per_feed
import pmix

LOG = open(MYOUT / ("k01.log" if not os.environ.get("BASE_MEMBER") else f"k01_base_{os.environ['BASE_MEMBER'].split(',')[0]}.log"), "a")
RES = MYOUT / "k01_results.jsonl"
T0 = time.time()


def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.write(s + "\n"); LOG.flush()


@nb.njit(parallel=True, cache=True)
def feed_z(x, off, M):
    out = np.zeros(len(x))
    for i in nb.prange(len(M)):
        o = off[i]; m = M[i]; s = 0.0
        for k in range(m):
            s += x[o + k]
        mu = s / m; v = 0.0
        for k in range(m):
            v += (x[o + k] - mu) ** 2
        sd = np.sqrt(v / m)
        for k in range(m):
            out[o + k] = (x[o + k] - mu) / (sd + 1e-9) if sd > 1e-12 else 0.0
    return out


@nb.njit(parallel=True, cache=True)
def feed_rank(x, off, M):
    out = np.full(len(x), 0.5)
    for i in nb.prange(len(M)):
        o = off[i]; m = M[i]
        if m < 2:
            continue
        for k in range(m):
            less = 0; eq = 0
            for j in range(m):
                if x[o + j] < x[o + k]:
                    less += 1
                elif x[o + j] == x[o + k]:
                    eq += 1
            out[o + k] = (less + 0.5 * (eq - 1)) / (m - 1)
    return out


@nb.njit(parallel=True, cache=True)
def feed_mean(x, off, M):
    out = np.zeros(len(M))
    for i in nb.prange(len(M)):
        s = 0.0
        for k in range(M[i]):
            s += x[off[i] + k]
        out[i] = s / M[i]
    return out


@nb.njit(parallel=True, cache=True)
def feed_std(x, off, M):
    out = np.zeros(len(M))
    for i in nb.prange(len(M)):
        s = 0.0; s2 = 0.0
        for k in range(M[i]):
            v = x[off[i] + k]; s += v; s2 += v * v
        mu = s / M[i]; out[i] = np.sqrt(max(s2 / M[i] - mu * mu, 0.0))
    return out


@nb.njit(parallel=True, cache=True)
def cat_entropy(c, off, M, ncat):
    ent = np.zeros(len(M)); nd = np.zeros(len(M)); mx = np.zeros(len(M))
    for i in nb.prange(len(M)):
        cnt = np.zeros(ncat)
        for k in range(M[i]):
            cnt[c[off[i] + k]] += 1
        e = 0.0; d = 0; best = 0.0
        for j in range(ncat):
            if cnt[j] > 0:
                p = cnt[j] / M[i]; e -= p * np.log(p); d += 1
                if p > best:
                    best = p
        ent[i] = e; nd[i] = d; mx[i] = best
    return ent, nd, mx


@nb.njit(parallel=True, cache=True)
def warm_rank(x, off, M):
    out = np.full(len(x), -1.0)
    for i in nb.prange(len(M)):
        o = off[i]; m = M[i]; nw = 0
        for k in range(m):
            nw += x[o + k] > 0
        for k in range(m):
            v = x[o + k]
            if v <= 0:
                continue
            if nw == 1:
                out[o + k] = 0.5; continue
            less = 0; eq = 0
            for j in range(m):
                u = x[o + j]
                if u <= 0:
                    continue
                if u < v:
                    less += 1
                elif u == v:
                    eq += 1
            out[o + k] = (less + 0.5 * (eq - 1)) / (nw - 1)
    return out


# ------------------------------------------------------------------------------------------------ data
def zz_np(a, off, M):
    return feed_z(np.asarray(a, np.float64), off, M)


Xv, Rv, mv = load("val"); Xh, Rh, mh = load("hold")
Xv = np.asarray(Xv); Rv = np.asarray(Rv); Xh = np.asarray(Xh); Rh = np.asarray(Rh)
Mv, offv, yv = mv["M"], mv["off"], mv["y"].astype(np.float64)
Mh, offh, yh = mh["M"], mh["off"], mh["y"].astype(np.float64)
hm = np.load(ROOT / "preds/holdout/holdout_meta.npy"); assert (hm[0] == Mh).all() and (hm[1] == offh).all()
assert np.allclose(np.load(ROOT / "preds/holdout/holdout_y.npy"), yh)
user = mv["user"]; nF = len(Mv); fid = np.repeat(np.arange(nF), Mv); npos_f = np.bincount(fid, weights=yv, minlength=nF)
nFh = len(Mh); fidh = np.repeat(np.arange(nFh), Mh); npos_h = np.bincount(fidh, weights=yh, minlength=nFh)
b = zz_np(0.5 * zz_np(np.load(ROOT / "preds/ens_G3_ft_val.npy"), offv, Mv) + 0.5 * zz_np(np.load(ROOT / "preds/ens_G3_to_val.npy"), offv, Mv), offv, Mv)
MEM = [f"v23_gru_pre2s_s50{i}" for i in range(1, 7)] + [f"r6_cd_m02_r10{i}" for i in range(1, 8)] + [f"r6_cd_m02_p70k_r10{i}" for i in range(1, 8)]
bh = np.mean([zz_np(np.load(ROOT / f"preds/holdout/nn_{m}_holdout.npy"), offh, Mh) for m in MEM], 0)
bh = zz_np(bh, offh, Mh)
BASE = os.environ.get("BASE_MEMBER")  # positive control: a single member (or comma list) instead of G3
if BASE:
    mems = BASE.split(",")
    b = zz_np(np.mean([zz_np(np.load(ROOT / f"preds/nn_{m}_trainonly_val.npy"), offv, Mv) for m in mems], 0), offv, Mv)
    bh = zz_np(np.mean([zz_np(np.load(ROOT / f"preds/holdout/nn_{m}_holdout.npy"), offh, Mh) for m in mems], 0), offh, Mh)
    RES = MYOUT / f"k01_results_base_{mems[0]}.jsonl"
day = (mv["ts"].astype(np.int64) + 10800) // 86400
xi = {n: j for j, n in enumerate(XT)}
luo_v = np.nan_to_num(Rv[:, xi["it_hist_all_luo"]]); cold = luo_v == 0
coldpos_f = np.bincount(fid, weights=yv * cold, minlength=nF)
SL = {"M<50": Mv < 50, "M>=50": Mv >= 50, "tue": day <= day.min() + 1, "thu": day >= day.min() + 2,
      "coldpos": (coldpos_f == npos_f) & (npos_f > 0), "warmpos": (coldpos_f == 0) & (npos_f > 0)}
WPRIV = pmix.weights()["private"]
folds = list(GroupKFold(n_splits=5).split(np.arange(nF), groups=user))
rowfold = np.empty(len(yv), np.int64)
for k, (trf, vaf) in enumerate(folds):
    rowfold[np.isin(fid, vaf)] = k
ndv = lambda s: ndcg_per_feed(s, yv, offv, Mv)
ndh = lambda s: ndcg_per_feed(s, yh, offh, Mh)


def nll(theta, G, rows, ff_all, y_all, npos_all, nFF):
    s = G @ theta; ff = ff_all[rows]
    e = np.exp(np.clip(s, -50, 20.0)); Z = np.bincount(ff, weights=e, minlength=nFF)
    ok = Z > 0; lZ = np.log(np.where(ok, Z, 1.0)); yy = y_all[rows]
    npos_used = np.bincount(ff, weights=yy, minlength=nFF)
    L = -(yy * s).sum() + (npos_used * lZ).sum()
    p = e / np.where(ok, Z, 1.0)[ff]; w = npos_used[ff] * p - yy
    return L, G.T @ w


def fit_lin(G, tr):
    th0 = np.zeros(G.shape[1]); th0[0] = 0.94
    rows = np.flatnonzero(tr)
    r = minimize(lambda t: nll(t, G[tr], rows, fid, yv, npos_f, nF), th0, jac=True, method="L-BFGS-B", options=dict(maxiter=300))
    return r.x


PARAMS = dict(objective="lambdarank", metric="ndcg", learning_rate=0.05, num_leaves=7, min_data_in_leaf=2000,
              lambda_l2=50.0, feature_fraction=1.0, bagging_fraction=1.0, lambdarank_truncation_level=1000,
              max_bin=63, verbose=-1, num_threads=12, deterministic=True, force_row_wise=True, seed=0)
betas = [fit_lin(b[:, None], rowfold != k)[0] for k in range(5)]
beta_full = fit_lin(b[:, None], np.ones(len(yv), bool))[0]


def run_lin(Gv, Gh):
    """Gv/Gh exclude the base column; returns per-feed val delta, holdout delta, weights."""
    G = np.column_stack([b, Gv]); pred = np.empty(len(yv))
    for k in range(5):
        tr = rowfold != k; pred[~tr] = G[~tr] @ fit_lin(G, tr)
    th = fit_lin(G, np.ones(len(yv), bool))
    ph = np.column_stack([bh, Gh]) @ th
    return ndv(pred) - C["lin_v"], ndh(ph) - C["lin_h"], th


def run_gbm(Fv, Fh, rounds=60, params=None, cat_idx=None, ckey=None):
    P = dict(PARAMS, **(params or {}))
    cf = [i + 1 for i in cat_idx] if cat_idx else "auto"
    F = np.column_stack([b, Fv]).astype(np.float32); pred = np.empty(len(yv))
    for k, (trf, vaf) in enumerate(folds):
        tr = rowfold != k; init = betas[k] * b
        ds = lgb.Dataset(F[tr], yv[tr], group=Mv[np.sort(trf)], init_score=init[tr], categorical_feature=cf)
        bst = lgb.train(P, ds, rounds); pred[~tr] = init[~tr] + bst.predict(F[~tr])
    ds = lgb.Dataset(F, yv, group=Mv, init_score=beta_full * b, categorical_feature=cf)
    bst = lgb.train(P, ds, rounds)
    Fhh = np.column_stack([bh, Fh]).astype(np.float32)
    ph = beta_full * bh + bst.predict(Fhh)
    kv = ckey or f"gbm_v{rounds}"; kh = (ckey.replace("_v", "_h") if ckey else f"gbm_h{rounds}")
    return ndv(pred) - C.get(kv, 0), ndh(ph) - C.get(kh, 0), bst


C = {}
C["base_v"] = ndv(b).mean(); C["base_h"] = ndh(bh).mean()
pv = np.empty(len(yv))
for k in range(5):
    pv[rowfold != k] = 0  # placeholder
G0 = b[:, None]; pred = np.empty(len(yv))
for k in range(5):
    tr = rowfold != k; pred[~tr] = G0[~tr] @ fit_lin(G0, tr)
C["lin_v"] = ndv(pred); C["lin_h"] = ndh(beta_full * bh)
for r in (60, 150):
    dv, dh, _ = run_gbm(np.zeros((len(yv), 0)), np.zeros((len(yh), 0)), r)
    C[f"gbm_v{r}"] = dv; C[f"gbm_h{r}"] = dh  # absolute (C empty for this key at call time)
log(f"== base: val G3 {C['base_v']:.5f}, holdout G3(TO 20) {C['base_h']:.5f}; LIN control val {C['lin_v'].mean():.5f} "
    f"hold {C['lin_h'].mean():.5f}; GBM60 control val {C['gbm_v60'].mean():.5f} hold {C['gbm_h60'].mean():.5f}; "
    f"GBM150 val {C['gbm_v150'].mean():.5f} hold {C['gbm_h150'].mean():.5f}; betas {np.round(betas, 3)} full {beta_full:.3f}  {time.time()-T0:.0f}s")


def report(name, kind, dv, dh, extra=""):
    se = dv.std() / np.sqrt(len(dv)); seh = dh.std() / np.sqrt(len(dh))
    pm, pse = pmix.wmean_se(dv, WPRIV)
    out = dict(name=name, kind=kind, val=float(dv.mean()), val_se=float(se), val_z=float(dv.mean() / se),
               pmix=float(pm), pmix_se=float(pse), hold=float(dh.mean()), hold_se=float(seh), hold_z=float(dh.mean() / seh))
    line = f"{name:34s} {kind:6s} val {dv.mean():+.5f} ({se:.5f}) z {dv.mean()/se:+.2f} | pmix {pm:+.5f} ({pse:.5f}) | hold {dh.mean():+.5f} ({seh:.5f}) z {dh.mean()/seh:+.2f}"
    for sn, m in SL.items():
        out[sn] = float(dv[m].mean()); line += f" | {sn} {dv[m].mean():+.5f}"
    log(line + extra)
    with open(RES, "a") as fh:
        fh.write(json.dumps(out) + "\n")
    return out


# ------------------------------------------------------------------------------------------------ feature builders
def col(X, name):
    return X[:, NAMES.index(name)].astype(np.float64)


def raw(R, name):
    return np.nan_to_num(R[:, xi[name]].astype(np.float64), nan=0.0)


def both(fn):
    return fn(Xv, Rv, offv, Mv, "val"), fn(Xh, Rh, offh, Mh, "hold")


MASS = ["ctx_hn_soft05", "ctx_hs_k128", "ctx_hs_e1", "ctx_hs_t10"]
NOREL = ["xt:ctx_nd_knn_h", "xt:ctx_nd_knn_u", "xt:ctx_nd_hcent", "xt:ctx_nd_top5h", "xt:ctx_nd_top5u", "xt:ctx_nd_knn_f1",
         "xt:ctx_nd_max", "xt:ctx_nd_ucent", "smax_et1", "smax8", "smax32", "smean", "xt:ctx_hn_rrw", "xt:ctx_hn_margin",
         "xt:ctx_hn_e1_rrw", "xt:ctx_hn_soft05", "xt:ctx_hs_k128", "xt:ctx_hs_e1", "xt:ctx_hs_t10", "xt:it_hist_et1_luo",
         "xt:seq_cv_pop", "fcs", "first_inv"]


def f_mass_log(X, R, off, M, s):
    return np.column_stack([zz_np(np.log1p(raw(R, n)), off, M) for n in MASS])


def f_mass_exc(X, R, off, M, s):
    return np.column_stack([np.log(np.maximum(raw(R, n), 10.0) / 10.0) for n in MASS])


def f_mass_gbm(X, R, off, M, s):
    return np.column_stack([np.log1p(raw(R, n)) for n in MASS] + [feed_rank(raw(R, n), off, M) for n in MASS])


def f_norel_z(X, R, off, M, s):
    return np.column_stack([zz_np(col(X, n), off, M) for n in NOREL])


def f_norel_gbm(X, R, off, M, s):
    return np.column_stack([zz_np(col(X, n), off, M) for n in NOREL] + [feed_rank(col(X, n), off, M) for n in NOREL])


POP = ["it_hist_all_luo", "seq_cv_invd20", "it_hist_et1_luo", "seq_cv_pop"]


def f_warm(X, R, off, M, s):
    out = []
    for n in POP:
        w = warm_rank(raw(R, n), off, M); out += [np.where(w < 0, 0.0, w), (w < 0).astype(np.float64)]
    return np.column_stack(out)


def feed_aggs(X, R, off, M, s):
    luo = raw(R, "it_hist_all_luo"); cv = raw(R, "seq_cv_invd20")
    cat = np.asarray(np.load(ROOT / f"data/flat/{'val' if s == 'val' else 'train'}/cand_category_id.npy", mmap_mode="r"))
    if s == "hold":
        fe = np.load(OUT / "hold_meta.npz")["fidx"]
        f = pl.read_parquet(ROOT / "data/flat/train/feeds.parquet")
        co = f["cand_off"].to_numpy()[fe]; Mm = f["M"].to_numpy()[fe]
        cat = cat[np.concatenate([np.arange(o, o + m) for o, m in zip(co, Mm)])]
    cat = cat.astype(np.int64)
    ent, nd, mx = cat_entropy(cat, off, M, 64)
    agg = [feed_mean((luo == 0).astype(np.float64), off, M), feed_mean(np.log1p(luo), off, M), feed_std(np.log1p(luo), off, M),
           feed_mean((cv == 0).astype(np.float64), off, M), feed_mean(col(X, "cos_u"), off, M), feed_std(col(X, "cos_u"), off, M),
           feed_mean(col(X, "smax"), off, M), feed_std(col(X, "smax"), off, M), feed_mean(col(X, "f1"), off, M),
           feed_mean(col(X, "f4"), off, M), feed_mean(col(X, "f5"), off, M), ent, nd, mx,
           feed_mean(raw(R, "pre2_fe_s"), off, M), feed_mean(col(X, "ccnt_share"), off, M)]
    return np.column_stack([np.repeat(a, M) for a in agg])


KEY = ["f1_zf", "f3_zf", "f4_zf", "f5_zf", "cos_u_zf", "smax_zf", "xt:it_hist_all_luo", "xt:it_hist_all_luo_zf",
       "xt:pre2_fe_s_zf", "xt:ctx_hs_t10", "xt:ctx_nd_knn_h", "xt:seq_cv_pop", "fcs", "log1p_ccnt", "logM", "log1pH"]


def f_key(X, R, off, M, s):
    return np.column_stack([col(X, n) for n in KEY])


def f_key_agg(X, R, off, M, s):
    return np.column_stack([f_key(X, R, off, M, s), feed_aggs(X, R, off, M, s)])


def f_all104(X, R, off, M, s):
    return X.astype(np.float64)


def f_all104_agg(X, R, off, M, s):
    return np.column_stack([X.astype(np.float64), feed_aggs(X, R, off, M, s)])


def cand_cat(s, M_, off_):
    if s == "val":
        return np.load(ROOT / "data/flat/val/cand_category_id.npy").astype(np.float64)
    fe = np.load(OUT / "hold_meta.npz")["fidx"]
    f = pl.read_parquet(ROOT / "data/flat/train/feeds.parquet")
    co = f["cand_off"].to_numpy()[fe]; Mm = f["M"].to_numpy()[fe]
    a = np.load(ROOT / "data/flat/train/cand_category_id.npy", mmap_mode="r")
    return np.asarray(a[np.concatenate([np.arange(o, o + m) for o, m in zip(co, Mm)])]).astype(np.float64)


def f_key_cat(X, R, off, M, s):
    return np.column_stack([cand_cat(s, M, off), f_key(X, R, off, M, s)])


def f_flags(X, R, off, M, s):
    f1, f2, f3, f5 = col(X, "f1"), col(X, "f2"), col(X, "f3"), col(X, "f5")
    return np.column_stack([(f1 <= 0.0179).astype(float), (f2 == 0).astype(float), (np.abs(f3 - 1 / 62) < 1e-6).astype(float),
                            (f5 == 0).astype(float), (np.abs(f5 - 1 / 62) < 1e-6).astype(float)])


def f_big(X, R, off, M, s):
    return np.column_stack([cand_cat(s, M, off), X.astype(np.float64), feed_aggs(X, R, off, M, s)])


BIG = dict(num_leaves=31, min_data_in_leaf=500, learning_rate=0.03, lambda_l2=10.0, feature_fraction=0.5, bagging_fraction=0.8,
           bagging_freq=1, deterministic=True)


rng = np.random.default_rng(0)
NZ = {"v": rng.standard_normal((len(yv), 2)), "h": rng.standard_normal((len(yh), 2))}

TESTS = {
    "noise2_lin": ("lin", lambda: (NZ["v"], NZ["h"])),
    "noise2_gbm": ("gbm", lambda: (NZ["v"], NZ["h"])),
    "T1a_mass_log_z_lin": ("lin", lambda: both(f_mass_log)),
    "T1b_mass_capexcess_lin": ("lin", lambda: both(f_mass_exc)),
    "T1c_mass_uncapped_gbm": ("gbm", lambda: both(f_mass_gbm)),
    "T2a_norel_z_lin": ("lin", lambda: both(f_norel_z)),
    "T2b_norel_zrk_gbm": ("gbm", lambda: both(f_norel_gbm)),
    "T3a_warmrank_lin": ("lin", lambda: both(f_warm)),
    "T3b_warmrank_gbm": ("gbm", lambda: both(f_warm)),
    "T4a_key_gbm": ("gbm", lambda: both(f_key)),
    "T4b_key_feedagg_gbm": ("gbm", lambda: both(f_key_agg)),
    "T0a_all104_gbm": ("gbm", lambda: both(f_all104)),
    "T0b_all104_feedagg_gbm": ("gbm", lambda: both(f_all104_agg)),
    "T0c_all104_gbm150": ("gbm150", lambda: both(f_all104)),
    "T0d_all104_feedagg_gbm150": ("gbm150", lambda: both(f_all104_agg)),
    "T5_key_cat_gbm": ("gbmcat", lambda: both(f_key_cat)),
    "T6_massflags_lin": ("lin", lambda: both(f_flags)),
    "T7a_big_noise121": ("gbmbig", lambda: (np.random.default_rng(1).standard_normal((len(yv), 121)), np.random.default_rng(2).standard_normal((len(yh), 121)))),
    "T7b_big_all": ("gbmbigcat", lambda: both(f_big)),
}

def f_mass_gbm_cap(X, R, off, M, s):
    return np.column_stack([np.log1p(np.minimum(raw(R, n), 10.0)) for n in MASS] + [feed_rank(np.minimum(raw(R, n), 10.0), off, M) for n in MASS])

def f_mass_lin_cap(X, R, off, M, s):
    return np.column_stack([zz_np(np.log1p(np.minimum(raw(R, n), 10.0)), off, M) for n in MASS])

TESTS["K1_mass_capped_gbm"] = ("gbm", lambda: both(f_mass_gbm_cap))
TESTS["K2_mass_capped_log_z_lin"] = ("lin", lambda: both(f_mass_lin_cap))

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    names = list(TESTS) if which == "all" else which.split(",")
    for n in names:
        kind, fn = TESTS[n]
        t = time.time(); Fv, Fh = fn()
        if kind == "lin":
            dv, dh, th = run_lin(Fv, Fh); extra = f" | w {np.round(th, 3)}"
        elif kind == "gbmcat":
            dv, dh, bst = run_gbm(Fv, Fh, 60, cat_idx=[0]); extra = ""
        elif kind.startswith("gbmbig"):
            if "gbmbig_v" not in C:
                cv_, ch_, _ = run_gbm(np.zeros((len(yv), 0)), np.zeros((len(yh), 0)), 300, BIG, ckey="gbmbig_v")
                C["gbmbig_v"] = cv_; C["gbmbig_h"] = ch_
                log(f"  big GBM control (300 rounds, 31 leaves): val {cv_.mean():.5f} hold {ch_.mean():.5f}")
            dv, dh, bst = run_gbm(Fv, Fh, 300, BIG, cat_idx=[0] if kind == "gbmbigcat" else None, ckey="gbmbig_v"); extra = ""
        else:
            r = 150 if kind == "gbm150" else 60
            dv, dh, bst = run_gbm(Fv, Fh, r); extra = ""
        report(n, kind, dv, dh, extra + f"  ({time.time()-t:.0f}s)")
        np.savez(MYOUT / f"k01_{n}_{os.environ.get('BASE_MEMBER','G3').split(',')[0]}.npz", dv=dv, dh=dh)
    log(f"DONE {time.time()-T0:.0f}s")
