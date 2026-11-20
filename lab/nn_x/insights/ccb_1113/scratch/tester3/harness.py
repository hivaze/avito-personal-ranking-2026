"""CCB-1113 tester3 harness: nested user-grouped LIN residual on G3 (PREDECLARED_ccb.txt B1-B8), CPU only.
Copy of the s04_resid_probe nll / fit_lin (unchanged) + the B2 holdout base + late/early + pmix.
Read-only on the project; writes only under this directory."""
import os
import sys
sys.dont_write_bytecode = True
for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS"]:
    os.environ[k] = "12"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import json
import time
from pathlib import Path
import numpy as np
import polars as pl
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from lib_av import feed_z_nanmin, pctrank, feed_auc, rows_of  # noqa: E402  (local copy: numba cache stays here)
sys.path.insert(0, "<repo>/src")
from metrics import ndcg_per_feed  # noqa: E402
import pmix  # noqa: E402

ROOT = Path("<repo>")
FLAT = ROOT / "data/flat"
CACHE = HERE / "cache"
OUT = HERE / "out"
HOLD_MEMBERS = [f"v23_gru_pre2s_s50{i}" for i in range(1, 7)] + [f"r6_cd_m02_r10{i}" for i in range(1, 8)] + \
    [f"r6_cd_m02_p70k_r10{i}" for i in range(1, 8)]
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)


# ---------------- val (B1, B3) ----------------
fv = pl.read_parquet(FLAT / "val/feeds.parquet")
Mv = fv["M"].to_numpy().astype(np.int64)
ov = fv["cand_off"].to_numpy().astype(np.int64)
user = fv["user_id"].to_numpy()
yv = np.load(FLAT / "val/cand_target.npy").astype(np.float64)
nF = len(Mv)
fid = np.repeat(np.arange(nF), Mv)
npos_f = np.bincount(fid, weights=yv, minlength=nF)
vday = (fv["timestamp"].to_numpy().astype(np.int64) + 10800) // 86400
zzv = lambda a: feed_z_nanmin(np.asarray(a, np.float64), ov, Mv)  # noqa: E731
b = zzv(0.5 * zzv(np.load(ROOT / "preds/ens_G3_ft_val.npy")) + 0.5 * zzv(np.load(ROOT / "preds/ens_G3_to_val.npy")))
ndv = lambda s: ndcg_per_feed(s, yv, ov, Mv)  # noqa: E731
BASE_V = ndv(b).mean()
assert abs(BASE_V - 0.40032) <= 0.00002, BASE_V
folds = list(GroupKFold(n_splits=5).split(np.arange(nF), groups=user))
rowfold = np.empty(len(yv), np.int64)
for k_, (trf, vaf) in enumerate(folds):
    rowfold[np.isin(fid, vaf)] = k_
luo = np.load(ROOT / "data/feats/val/it_hist_all_luo.npy")
cold = luo == 0
coldpos_f = np.bincount(fid, weights=yv * cold, minlength=nF)
SL = {"M<50": Mv < 50, "M>=50": Mv >= 50, "tue": vday <= vday.min() + 1, "thu": vday >= vday.min() + 2,
      "coldpos": (coldpos_f == npos_f) & (npos_f > 0), "warmpos": (coldpos_f == 0) & (npos_f > 0)}
PMW = pmix.weights()["private"]

# ---------------- holdout (B2) ----------------
_ft = pl.read_parquet(FLAT / "train/feeds.parquet", columns=["M", "timestamp", "user_id", "cand_off", "hist_off", "H"])
_Mt = _ft["M"].to_numpy().astype(np.int64)
_p = np.nan_to_num(np.load(ROOT / "eda/labels_position/out/train_prandom.npz")["p_random"].astype(np.float32), nan=0.0)
_p[_Mt < 50] = 0.0
HOLD = np.sort(np.random.default_rng(12345).choice(np.where((_p > 0.9) & (_Mt >= 50))[0], 30000, replace=False))
Mh, oh = np.load(ROOT / "preds/holdout/holdout_meta.npy")
Mh = Mh.astype(np.int64)
oh = oh.astype(np.int64)
assert np.array_equal(_Mt[HOLD], Mh), "holdout feed ids do not reproduce holdout_meta"
yh = np.load(ROOT / "preds/holdout/holdout_y.npy").astype(np.float64)
HOLD_ROWS = rows_of(_ft["cand_off"].to_numpy().astype(np.int64), _Mt, HOLD)
_yt = np.load(FLAT / "train/cand_target.npy", mmap_mode="r")
assert np.array_equal(np.asarray(_yt[HOLD_ROWS], np.float64), yh), "holdout rows != holdout_y"
hday_abs = (_ft["timestamp"].to_numpy()[HOLD].astype(np.int64) + 10800) // 86400
hday = hday_abs - hday_abs.min()
H_LATE, H_EARLY = hday >= 11, hday <= 2
assert H_LATE.sum() == 8279 and H_EARLY.sum() == 7823, (H_LATE.sum(), H_EARLY.sum())
zzh = lambda a: feed_z_nanmin(np.asarray(a, np.float64), oh, Mh)  # noqa: E731
bh = zzh(np.mean([zzh(np.load(ROOT / f"preds/holdout/nn_{k}_holdout.npy")) for k in HOLD_MEMBERS], 0))
ndh = lambda s: ndcg_per_feed(s, yh, oh, Mh)  # noqa: E731
BASE_H_PF = ndh(bh)
BASE_H = BASE_H_PF.mean()
assert abs(BASE_H - 0.32379) <= 0.00002, BASE_H
fidh = np.repeat(np.arange(len(Mh)), Mh)
log(f"bases: val {BASE_V:.5f} (14,423 feeds, {len(yv)} rows) | holdout {BASE_H:.5f} ({len(Mh)} feeds, {len(yh)} rows); "
    f"late {H_LATE.sum()} early {H_EARLY.sum()}")


# ---------------- LIN (s04 nll / fit_lin, unchanged) ----------------
def nll(theta, G, rows, ff_all, yy_all, nposf, nf):
    s = G @ theta
    ff = ff_all[rows]
    e = np.exp(np.clip(s, -50, 20.0))
    Z = np.bincount(ff, weights=e, minlength=nf)
    ok = Z > 0
    lZ = np.log(np.where(ok, Z, 1.0))
    yy = yy_all[rows]
    L = -(yy * s).sum() + (nposf * lZ).sum()
    p = e / np.where(ok, Z, 1.0)[ff]
    w = nposf[ff] * p - yy
    return L, G.T @ w


def fit_lin(G, tr):
    th0 = np.zeros(G.shape[1])
    th0[0] = 0.94
    rows = np.flatnonzero(tr)
    # npos over the training rows only (identical to s04: feeds outside tr have Z = 0 and contribute lZ = 0)
    r = minimize(lambda t: nll(t, G[tr], rows, fid, yv, npos_f, nF), th0, jac=True, method="L-BFGS-B",
                 options=dict(maxiter=200))
    return r.x


def cols_of(x, zz, off, M):
    x = np.asarray(x, np.float64)
    return zz(x), pctrank(x, off, M), np.isnan(x).astype(np.float64)


def design(xs, side, nan_cols=None):
    """xs: list of raw columns. side 'v' or 'h'. nan_cols: which isnan terms to include (decided on val)."""
    zz, off, M, base = (zzv, ov, Mv, b) if side == "v" else (zzh, oh, Mh, bh)
    G = [base]
    use_nan = []
    for j, x in enumerate(xs):
        z, rk, isn = cols_of(x, zz, off, M)
        G += [z, rk]
        has = bool(isn.any()) if nan_cols is None else nan_cols[j]
        if has:
            G.append(isn)
        use_nan.append(has)
    return np.stack(G, 1), use_nan


_CTRL = {}


def control():
    if "cv" not in _CTRL:
        pred = np.empty(len(yv))
        G = b[:, None]
        for k in range(5):
            tr = rowfold != k
            pred[~tr] = G[~tr] @ fit_lin(G, tr)
        _CTRL["cv"] = ndv(pred)
        _CTRL["betas"] = [float(fit_lin(G, rowfold != k)[0]) for k in range(5)]
        _CTRL["full_beta"] = float(fit_lin(G, np.ones(len(yv), bool))[0])
        log(f"control LIN cv {_CTRL['cv'].mean():.5f} betas {np.round(_CTRL['betas'], 3)} full beta {_CTRL['full_beta']:.3f}")
    return _CTRL


def se(d):
    return float(d.std() / np.sqrt(len(d)))


def run_item(item, names, xv_full, xh, xv_fold=None, extra=None):
    """xv_full: list of raw val columns (used for the nested CV unless xv_fold is given, and for theta_full).
    xv_fold: optional callable k -> list of raw val columns built from fold-k training data (label-derived items).
    xh: list of raw holdout columns (built with the full-val / B9 holdout source)."""
    c = control()
    Gfull, use_nan = design(xv_full, "v")
    pred = np.empty(len(yv))
    for k in range(5):
        tr = rowfold != k
        Gk = Gfull if xv_fold is None else design(xv_fold(k), "v", use_nan)[0]
        th = fit_lin(Gk, tr)
        pred[~tr] = Gk[~tr] @ th
    d = ndv(pred) - c["cv"]
    dm, ds = float(d.mean()), se(d)
    z = dm / ds
    p1 = float(1 - norm.cdf(z))
    pm, pse = pmix.wmean_se(d, PMW)
    th_full = fit_lin(Gfull, np.ones(len(yv), bool))
    Gh, _ = design(xh, "h", use_nan)
    dh = ndh(Gh @ th_full) - BASE_H_PF
    res = dict(id=item, columns=names, n_cols_lin=int(Gfull.shape[1]), val_delta=dm, val_se=ds, z=float(z), p_one_sided=p1,
               pmix=float(pm), pmix_se=float(pse), pmix_veto=bool(pm < -pse),
               hold_delta=float(dh.mean()), hold_se=se(dh), hold_z=float(dh.mean() / se(dh)),
               late_delta=float(dh[H_LATE].mean()), late_se=se(dh[H_LATE]),
               early_delta=float(dh[H_EARLY].mean()), early_se=se(dh[H_EARLY]),
               theta_full=[float(t) for t in th_full], isnan_terms=use_nan,
               slices={sn: [float(d[m].mean()), se(d[m])] for sn, m in SL.items()},
               in_noise_band=bool(-0.00047 <= dm <= 0.00022))
    aucs = {}
    for nm, x in zip(names, xv_full):
        a, n = feed_auc(np.asarray(x, np.float64), yv, ov, Mv)
        aucs[nm] = float(a)
    res["auc_within_feed"] = aucs
    res["coverage_nonnan_val"] = {nm: float(np.isfinite(np.asarray(x, np.float64)).mean()) for nm, x in zip(names, xv_full)}
    res["coverage_nonnan_hold"] = {nm: float(np.isfinite(np.asarray(x, np.float64)).mean()) for nm, x in zip(names, xh)}
    if extra:
        res["info"] = extra
    log(f"== {item} {names}: nested val {dm:+.5f} (SE {ds:.5f}, z {z:+.2f}, p1 {p1:.4f}) | holdout {res['hold_delta']:+.5f} "
        f"(SE {res['hold_se']:.5f}, z {res['hold_z']:+.2f}) | late {res['late_delta']:+.5f} ({res['late_se']:.5f}) | early "
        f"{res['early_delta']:+.5f} ({res['early_se']:.5f}) | pmix {pm:+.5f} ({pse:.5f}) veto {res['pmix_veto']} | slices "
        + " ".join(f"{k} {v[0]:+.5f}" for k, v in res["slices"].items()) + f" | noise band {res['in_noise_band']}")
    log(f"   theta_full {np.round(th_full, 4).tolist()} | AUC {aucs} | cov val {res['coverage_nonnan_val']} hold {res['coverage_nonnan_hold']}")
    with open(OUT / f"{item}.json", "w") as fh:
        json.dump(res, fh, indent=1)
    return res


def lin_info(name, xs, label):
    """INFORMATION-only nested LIN delta vs control (no holdout)."""
    c = control()
    G, _ = design(xs, "v")
    pred = np.empty(len(yv))
    for k in range(5):
        tr = rowfold != k
        pred[~tr] = G[~tr] @ fit_lin(G, tr)
    d = ndv(pred) - c["cv"]
    log(f"   INFO {label}: nested val {d.mean():+.5f} (SE {se(d):.5f}, z {d.mean() / se(d):+.2f})")
    return dict(label=label, val_delta=float(d.mean()), val_se=se(d))
