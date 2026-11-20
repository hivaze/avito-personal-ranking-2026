"""CCB-1113 tester-2 harness (copy of the s04_resid_probe LIN protocol + B2/B6/B7 holdout / pmix read).

PREDECLARED rule: <repo>/nn_x/insights/ccb_1113/PREDECLARED_ccb.txt (B1-B8, D).
Read-only on the project; numba caches only in this directory.
"""
import os
for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS"]:
    os.environ[k] = "12"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
import sys
sys.dont_write_bytecode = True
import json, time
from pathlib import Path
import numpy as np, polars as pl, numba as nb
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.model_selection import GroupKFold

sys.path.insert(0, "<repo>/src")
from metrics import ndcg_per_feed  # noqa: E402
import pmix  # noqa: E402

ROOT = Path("<repo>")
FLAT = ROOT / "data/flat"
HERE = Path(__file__).resolve().parent
FEATD = HERE / "feats"
T0 = time.time()


@nb.njit(cache=False)
def feed_z_nanmin(x, off, M):
    out = np.zeros(len(x))
    for i in range(len(M)):
        o = off[i]; m = M[i]; mn = 1e30; c = 0
        for k in range(m):
            v = x[o + k]
            if v == v:
                c += 1
                if v < mn:
                    mn = v
        if c == 0:
            continue
        s = 0.0; s2 = 0.0
        for k in range(m):
            v = x[o + k]
            if v != v:
                v = mn
            s += v; s2 += v * v
        mu = s / m; sd = np.sqrt(max(s2 / m - mu * mu, 0.0))
        if sd < 1e-9:
            continue
        for k in range(m):
            v = x[o + k]
            if v != v:
                v = mn
            out[o + k] = (v - mu) / sd
    return out


@nb.njit(cache=False)
def pctrank(x, off, M):
    """within-feed pct rank in [-0.5, 0.5], ties averaged, NaN -> below min (copy of lib_av.pctrank)"""
    out = np.zeros(len(x))
    for f_ in range(len(M)):
        o = off[f_]; m = M[f_]
        if m <= 1:
            continue
        v = np.empty(m)
        for k in range(m):
            t = x[o + k]
            v[k] = -1e30 if t != t else t
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


# ---------------------------------------------------------------- val (B1, B3)
fv = pl.read_parquet(FLAT / "val/feeds.parquet")
Mv = fv["M"].to_numpy().astype(np.int64); ov = fv["cand_off"].to_numpy().astype(np.int64)
user_v = fv["user_id"].to_numpy()
yv = np.load(FLAT / "val/cand_target.npy").astype(np.float64)
nF = len(Mv); fid = np.repeat(np.arange(nF), Mv); npos_f = np.bincount(fid, weights=yv, minlength=nF)
day_v = (fv["timestamp"].to_numpy().astype(np.int64) + 10800) // 86400
zv = lambda a: feed_z_nanmin(np.asarray(a, np.float64), ov, Mv)  # noqa: E731
b = zv(0.5 * zv(np.load(ROOT / "preds/ens_G3_ft_val.npy")) + 0.5 * zv(np.load(ROOT / "preds/ens_G3_to_val.npy")))
folds = list(GroupKFold(n_splits=5).split(np.arange(nF), groups=user_v))
rowfold = np.empty(len(yv), np.int64)
for k, (trf, vaf) in enumerate(folds):
    rowfold[np.isin(fid, vaf)] = k
_luo = np.load(ROOT / "data/feats/val/it_hist_all_luo.npy") if (ROOT / "data/feats/val/it_hist_all_luo.npy").exists() else None
SL = {"M<50": Mv < 50, "M>=50": Mv >= 50, "tue": day_v <= day_v.min() + 1, "thu": day_v >= day_v.min() + 2}
if _luo is not None:
    cold = _luo == 0
    coldpos_f = np.bincount(fid, weights=yv * cold, minlength=nF)
    SL["coldpos"] = (coldpos_f == npos_f) & (npos_f > 0); SL["warmpos"] = (coldpos_f == 0) & (npos_f > 0)

# ---------------------------------------------------------------- holdout (B2)
HD = ROOT / "preds/holdout"
Mh, oh = np.load(HD / "holdout_meta.npy"); Mh = Mh.astype(np.int64); oh = oh.astype(np.int64)
yh = np.load(HD / "holdout_y.npy").astype(np.float64)
_f = pl.read_parquet(FLAT / "train/feeds.parquet", columns=["M", "timestamp", "user_id", "cand_off", "hist_off", "H"])
_M = _f["M"].to_numpy().astype(np.int64)
_p = np.nan_to_num(np.load(ROOT / "eda/labels_position/out/train_prandom.npz")["p_random"].astype(np.float32), nan=0.0)
_p[_M < 50] = 0.0
HOLD = np.sort(np.random.default_rng(12345).choice(np.where((_p > 0.9) & (_M >= 50))[0], 30000, replace=False))
assert np.array_equal(_M[HOLD], Mh), "holdout feed ids do not reproduce holdout_meta"
HOLD_DAY_ABS = (_f["timestamp"].to_numpy()[HOLD].astype(np.int64) + 10800) // 86400
_day = HOLD_DAY_ABS - HOLD_DAY_ABS.min()
H_LATE, H_EARLY = _day >= 11, _day <= 2
assert H_LATE.sum() == 8279 and H_EARLY.sum() == 7823, (H_LATE.sum(), H_EARLY.sum())
HOLD_USER = _f["user_id"].to_numpy()[HOLD]
_coff = _f["cand_off"].to_numpy().astype(np.int64)
HOLD_ROWS = np.concatenate([np.arange(o, o + m) for o, m in zip(_coff[HOLD], Mh)])
del _f, _p, _M
zh = lambda a: feed_z_nanmin(np.asarray(a, np.float64), oh, Mh)  # noqa: E731
MEMB = [f"v23_gru_pre2s_s50{i}" for i in range(1, 7)] + [f"r6_cd_m02_r10{i}" for i in range(1, 8)] + \
       [f"r6_cd_m02_p70k_r10{i}" for i in range(1, 8)]
nFh = len(Mh); fidh = np.repeat(np.arange(nFh), Mh)


def base_h():
    acc = np.zeros(len(yh))
    for k in MEMB:
        acc += zh(np.load(HD / f"nn_{k}_holdout.npy"))
    return zh(acc / len(MEMB))


bh = base_h()
nd = lambda s: ndcg_per_feed(s, yv, ov, Mv)  # noqa: E731
ndh = lambda s: ndcg_per_feed(s, yh, oh, Mh)  # noqa: E731


def log_to(fh):
    def log(*a):
        s = f"[{time.time()-T0:6.0f}s] " + " ".join(str(x) for x in a)
        print(s, flush=True)
        if fh is not None:
            fh.write(s + "\n"); fh.flush()
    return log


def sanity(log):
    bv = nd(b).mean(); bhv = ndh(bh).mean()
    log(f"SANITY base val {bv:.5f} (14,423 feeds, {len(yv)} rows)  holdout {bhv:.5f} (30,000 feeds, {len(yh)} rows)")
    assert abs(bv - 0.40032) <= 2e-5 and abs(bhv - 0.32379) <= 2e-5
    yy = np.load(FLAT / "train/cand_target.npy", mmap_mode="r")[HOLD_ROWS]
    assert np.array_equal(np.asarray(yy, np.float64), yh), "holdout rows != train cand_target"
    log("SANITY holdout rows reproduce holdout_y from data/flat/train (HOLD_ROWS)")


# ---------------------------------------------------------------- LIN (B4, B5)
def nll(theta, G, rows):
    s = G @ theta; ff = fid[rows]
    e = np.exp(np.clip(s, -50, 20.0)); Z = np.bincount(ff, weights=e, minlength=nF)
    ok = Z > 0; lZ = np.log(np.where(ok, Z, 1.0)); yy = yv[rows]
    L = -(yy * s).sum() + (npos_f * lZ).sum()
    p = e / np.where(ok, Z, 1.0)[ff]; w = npos_f[ff] * p - yy
    return L, G.T @ w


def fit_lin(G, tr):
    th0 = np.zeros(G.shape[1]); th0[0] = 0.94
    rows = np.flatnonzero(tr)
    Gt = np.ascontiguousarray(G[tr])
    r = minimize(lambda t: nll(t, Gt, rows), th0, jac=True, method="L-BFGS-B", options=dict(maxiter=200))
    return r.x


def cv_lin(G):
    pred = np.empty(len(yv))
    for k in range(5):
        tr = rowfold != k; pred[~tr] = G[~tr] @ fit_lin(G, tr)
    return pred


def design(cols, zf, M, off):
    """cols: list of arrays -> list of [z, pctrank, isnan?] blocks; returns stacked matrix and the isnan pattern."""
    out, pat = [], []
    for x in cols:
        x = np.asarray(x, np.float64); isn = np.isnan(x)
        out += [zf(x), pctrank(x, off, M)]
        pat.append(bool(isn.any()))
        if isn.any():
            out.append(isn.astype(np.float64))
    return out, pat


def design_h(cols, pat):
    out = []
    for x, p in zip(cols, pat):
        x = np.asarray(x, np.float64)
        out += [zh(x), pctrank(x, oh, Mh)]
        if p:
            out.append(np.isnan(x).astype(np.float64))
    return out


def stat(d):
    se = d.std() / np.sqrt(len(d)); return float(d.mean()), float(se), float(d.mean() / se) if se > 0 else 0.0


def evaluate(item, names, Xv, Xh, log, extra=None):
    """Xv: list of val columns (len(yv)), Xh: list of holdout columns (len(yh)) in the same order."""
    t = time.time()
    for n, x in zip(names, Xv):
        x = np.asarray(x, np.float64)
        log(f"  col {n:22s} val: nan {np.isnan(x).mean():.4f} zero {np.mean(x == 0):.4f} mean {np.nanmean(x):+.4g} sd {np.nanstd(x):.4g}")
    for n, x in zip(names, Xh):
        x = np.asarray(x, np.float64)
        log(f"  col {n:22s} hold: nan {np.isnan(x).mean():.4f} zero {np.mean(x == 0):.4f} mean {np.nanmean(x):+.4g} sd {np.nanstd(x):.4g}")
    blocks, pat = design(Xv, zv, Mv, ov)
    G = np.stack([b] + blocks, 1)
    log(f"{item}: LIN design {G.shape[1]} columns (beta + {len(names)} x [z, pctrank] + {sum(pat)} isnan)")
    c_pred = cv_lin(b[:, None]); c_nd = nd(c_pred)
    p_pred = cv_lin(G); p_nd = nd(p_pred)
    d = p_nd - c_nd
    dv, sev, zv_ = stat(d); pval = float(1 - norm.cdf(zv_))
    res = dict(id=item, val_delta=dv, val_se=sev, val_z=zv_, p_one_sided=pval, ctrl_val=float(c_nd.mean()), lin_val=float(p_nd.mean()))
    line = f"{item} NESTED val delta {dv:+.5f} (SE {sev:.5f}) z {zv_:+.2f} p {pval:.4g} | control {c_nd.mean():.5f} LIN {p_nd.mean():.5f}"
    for sn, m in SL.items():
        res[f"sl_{sn}"] = [float(d[m].mean()), float(d[m].std() / np.sqrt(m.sum()))]; line += f" | {sn} {d[m].mean():+.5f}"
    log(line)
    W = pmix.weights()["private"]
    pm, pse = pmix.wmean_se(d, W)
    res["pmix_delta"] = float(pm); res["pmix_se"] = float(pse)
    log(f"{item} PMIX(private) {pm:+.5f} (SE {pse:.5f}) veto={'YES' if pm < -pse else 'no'}")
    # full-val fit -> holdout (B6)
    th = fit_lin(G, np.ones(len(yv), bool))
    res["theta_full"] = [float(v) for v in th]
    log(f"{item} theta_full beta {th[0]:.4f} rest {np.round(th[1:], 4).tolist()}")
    Gh = np.stack([bh] + design_h(Xh, pat), 1)
    sh = Gh @ th
    dh = ndh(sh) - ndh(bh)
    hd, hse, hz = stat(dh); ld, lse, lz = stat(dh[H_LATE]); ed, ese, ez = stat(dh[H_EARLY])
    res.update(hold_delta=hd, hold_se=hse, hold_z=hz, late_delta=ld, late_se=lse, early_delta=ed, early_se=ese)
    log(f"{item} HOLDOUT delta {hd:+.5f} (SE {hse:.5f}) z {hz:+.2f} | LATE {ld:+.5f} ({lse:.5f}) | EARLY {ed:+.5f} ({ese:.5f})")
    crit = dict(c1=dv >= 0.0005, c3=hd >= 0, c4=ld >= 0, c5=not (pm < -pse))
    res["criteria_ex_holm"] = crit
    log(f"{item} criteria (Holm applied by the synthesiser): (1) val>=+0.0005 {crit['c1']} (3) hold>=0 {crit['c3']} "
        f"(4) late>=0 {crit['c4']} (5) pmix veto-free {crit['c5']}; noise band s04 -0.00047..+0.00022; {time.time()-t:.0f}s")
    if extra:
        res.update(extra)
    return res
