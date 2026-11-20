"""Shared helpers for the val<->public calibration / residual anatomy (nn_x/audit/valpub/calib). CPU only, 12 threads.
Nothing outside this directory is written. Public scores are used as DATA for the analysis the user asked for."""
import os
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "POLARS_MAX_THREADS", "NUMBA_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[k] = "12"
import sys
from pathlib import Path
import numpy as np
import polars as pl
sys.path.insert(0, "<repo>/src"); sys.path.insert(0, "<repo>/eda/errors/segments")
from metrics import ndcg_per_feed  # noqa: E402

ROOT = Path("<repo>")
FLAT = ROOT / "data/flat"
P = ROOT / "preds"
HERE = ROOT / "nn_x/audit/valpub/calib"
OUT = HERE / "out"
OUT.mkdir(exist_ok=True, parents=True)
BOUND = 1790497920
N_PUB, N_PRIV = 8562, 20033

# ---------------------------------------------------------------- scored files (public = exact LB readings)
# modern pools: per-feed val arrays exist; holdout = train-only member pool on the 30K holdout (None if not defined)
MODERN = {
    #  name     public      val      holdout   re-cool share  w_ft   val-selected?
    "Cd":     dict(pub=0.39208,   val=0.39901, hold=0.32319, share=0.00, wft=0.5, sel=0),
    "Pall":   dict(pub=0.392436,  val=0.39896, hold=0.32335, share=0.00, wft=0.5, sel=0),
    "PF7x":   dict(pub=0.392723,  val=0.39974, hold=0.32360, share=0.18, wft=0.5, sel=0),
    "G3":     dict(pub=0.392820,  val=0.40032, hold=0.32379, share=0.70, wft=0.5, sel=0),
    "M2":     dict(pub=0.392434,  val=0.40094, hold=0.32361, share=1.00, wft=0.5, sel=1),
    "S2":     dict(pub=0.392473,  val=0.40088, hold=None,    share=0.70, wft=0.5, sel=1),
    "TOonly": dict(pub=0.392346,  val=0.40035, hold=0.32379, share=0.70, wft=0.0, sel=0),
}
OLD = {  # older generations, scalar val only (per-feed arrays reconstructed where possible)
    "v3":  dict(pub=0.3739,  val=0.3816),
    "v4":  dict(pub=0.3772,  val=0.3839),
    "v5":  dict(pub=0.3802,  val=0.3869),
    "v10": dict(pub=0.38786, val=0.39399),
}
OVERLAY = dict(pub=0.392819, val=0.40486)  # control: leak neutralised on test; excluded from every fit
ORDER = ["Cd", "Pall", "PF7x", "G3", "M2", "S2", "TOonly"]
CONTRASTS = ["Cd", "Pall", "PF7x", "M2", "S2", "TOonly"]  # vs G3


def feed_meta(split):
    f = pl.read_parquet(FLAT / split / "feeds.parquet")
    return f, f["M"].to_numpy().astype(np.int64), f["cand_off"].to_numpy().astype(np.int64)


def zfeed(s, M):
    s = np.asarray(s, np.float64)
    n = len(M); fid = np.repeat(np.arange(n), M)
    mu = np.bincount(fid, s, n) / M
    sd = np.sqrt(np.maximum(np.bincount(fid, s * s, n) / M - mu ** 2, 0)) + 1e-9
    return (s - mu[fid]) / sd[fid]


def msk_day(ts):
    return (np.asarray(ts, np.int64) + 10800) // 86400


def msk_hour(ts):
    return ((np.asarray(ts, np.int64) + 10800) % 86400) // 3600


def weekday(ts):
    return (msk_day(ts) + 3) % 7  # 0=Mon


def final_score(name, split):
    """final per-candidate score of a modern pool on split (val/test)."""
    M = feed_meta(split)[1]
    if name == "S2":
        return np.load(P / f"nnstk_g3gate3r_{split}.npy").astype(np.float64)
    if name == "TOonly":
        return np.load(P / f"ens_G3_toonly_{split}.npy").astype(np.float64)
    if name == "G3" and (P / f"ens_G3_final_{split}.npy").exists():
        return np.load(P / f"ens_G3_final_{split}.npy").astype(np.float64)
    ft = np.load(P / f"ens_{name}_ft_{split}.npy"); to = np.load(P / f"ens_{name}_to_{split}.npy")
    return 0.5 * zfeed(ft, M) + 0.5 * zfeed(to, M)


def old_val_score(name):
    """reconstruct the val score of an old-generation submission where the recipe is on record (review_shift.md)."""
    M = feed_meta("val")[1]
    if name == "v10":  # 0.5 z(nn_final4_val_oof) + 0.5 z(nn_final4_trainonly_val) + 0.1 z(pre_final_val)
        s = 0.5 * zfeed(np.load(P / "nn_final4_val_oof.npy"), M) + 0.5 * zfeed(np.load(P / "nn_final4_trainonly_val.npy"), M)
        pre = P / "pre_final_val.npy"
        if pre.exists():
            s = s + 0.1 * zfeed(np.load(pre), M)
        return s
    return None


def val_perfeed(name):
    """per-feed val NDCG of a scored file (cached)."""
    c = OUT / f"pf_val_{name}.npy"
    if c.exists():
        return np.load(c)
    f, M, off = feed_meta("val"); y = np.load(FLAT / "val/cand_target.npy")
    s = final_score(name, "val") if name in MODERN else old_val_score(name)
    if s is None:
        return None
    pf = ndcg_per_feed(np.asarray(s, np.float32), y, off, M)
    np.save(c, pf)
    return pf


def val_half_perfeed(name, half):
    """per-feed val NDCG of one half (ft/to) of a pool, where the half arrays exist."""
    c = OUT / f"pf_val_{name}_{half}.npy"
    if c.exists():
        return np.load(c)
    p = P / f"ens_{name}_{half}_val.npy"
    if not p.exists():
        return None
    f, M, off = feed_meta("val"); y = np.load(FLAT / "val/cand_target.npy")
    pf = ndcg_per_feed(zfeed(np.load(p), M).astype(np.float32), y, off, M)
    np.save(c, pf)
    return pf


def test_disagreement(a, b):
    """label-free per-feed disagreement of two pools on test: 1 - within-feed Spearman of their final scores (cached)."""
    c = OUT / f"dis_test_{a}_{b}.npy"
    if c.exists():
        return np.load(c)
    f, M, off = feed_meta("test")
    sa, sb = final_score(a, "test"), final_score(b, "test")
    n = len(M); fid = np.repeat(np.arange(n), M)
    ra = np.empty(len(sa)); rb = np.empty(len(sb))
    oa = np.lexsort((sa, fid)); ra[oa] = np.arange(len(sa)) - np.repeat(off, M)
    ob = np.lexsort((sb, fid)); rb[ob] = np.arange(len(sb)) - np.repeat(off, M)
    d2 = np.bincount(fid, (ra - rb) ** 2, n)
    rho = 1 - 6 * d2 / np.maximum(M * (M.astype(np.float64) ** 2 - 1), 1)
    out = (1 - rho).astype(np.float64)
    np.save(c, out)
    return out


def val_disagreement(a, b):
    c = OUT / f"dis_val_{a}_{b}.npy"
    if c.exists():
        return np.load(c)
    f, M, off = feed_meta("val")
    sa, sb = final_score(a, "val"), final_score(b, "val")
    n = len(M); fid = np.repeat(np.arange(n), M)
    ra = np.empty(len(sa)); rb = np.empty(len(sb))
    oa = np.lexsort((sa, fid)); ra[oa] = np.arange(len(sa)) - np.repeat(off, M)
    ob = np.lexsort((sb, fid)); rb[ob] = np.arange(len(sb)) - np.repeat(off, M)
    d2 = np.bincount(fid, (ra - rb) ** 2, n)
    rho = 1 - 6 * d2 / np.maximum(M * (M.astype(np.float64) ** 2 - 1), 1)
    out = (1 - rho).astype(np.float64)
    np.save(c, out)
    return out


# ---------------------------------------------------------------- descriptors (label-free) for val / test
def desc(split):
    d = pl.read_parquet(ROOT / f"nn_x/pubdiag/shift/desc_{split}.parquet")
    f = feed_meta(split)[0]
    ts = f["timestamp"].to_numpy().astype(np.int64)
    D = {k: d[k].to_numpy().astype(np.float64) for k in d.columns}
    D["ts"] = ts; D["hour"] = msk_hour(ts).astype(np.float64); D["wd"] = weekday(ts).astype(np.float64)
    D["weekend"] = (D["wd"] >= 5).astype(np.float64)
    D["day"] = msk_day(ts).astype(np.float64)
    if split == "val":
        D["tue"] = (D["day"] < 20720).astype(np.float64)
    D["M"] = f["M"].to_numpy().astype(np.float64); D["H"] = f["H"].to_numpy().astype(np.float64)
    D["uid0"] = (f["user_id"].to_numpy() == 0).astype(np.float64)
    return D


M_EDGES = np.array([10, 20, 30, 50, 75, 100, 150, 200, 250])          # src/calib.py bins
SEG_M = [10, 25, 50, 100, 200]                                           # segdefs bins
SEG_COLD = [0.15, 0.3, 0.45]
HOUR_EDGES = [6, 10, 14, 18]                                             # h00-05 06-09 10-13 14-17 18-23


def bins(D, kind):
    """integer cell id per feed for a reweighting scheme."""
    if kind == "M":
        return np.searchsorted(M_EDGES, D["M"], side="right")
    if kind == "Mseg":
        return np.searchsorted(np.array(SEG_M, float), D["M"], side="left")
    if kind == "cold":
        return np.searchsorted(np.array(SEG_COLD), D["cold_share"], side="left")
    if kind == "luocold":
        return np.searchsorted(np.array([0.2, 0.35, 0.5, 0.65]), D["luo_cold_share"], side="left")
    if kind == "hour5":
        return np.digitize(D["hour"], HOUR_EDGES)
    if kind == "hour24":
        return D["hour"].astype(int)
    if kind == "Mxcold":
        return bins(D, "Mseg") * 10 + bins(D, "cold")
    if kind == "Mxluocold":
        return bins(D, "Mseg") * 10 + bins(D, "luocold")
    if kind == "Mxcoldxhour":
        return bins(D, "Mxcold") * 10 + bins(D, "hour5")
    if kind == "Mxhour":
        return bins(D, "Mseg") * 10 + bins(D, "hour5")
    if kind == "coldxhour":
        return bins(D, "cold") * 10 + bins(D, "hour5")
    if kind == "H":
        return np.searchsorted(np.array([0, 19, 63, 191, 255], float), D["H"], side="left")
    if kind == "user":
        u = np.where(D["uid0"] > 0, 4, np.searchsorted(np.array([0, 1, 4], float), D["n_train_feeds"], side="left"))
        return u
    if kind == "Mxuser":
        return bins(D, "Mseg") * 10 + bins(D, "user")
    if kind == "cih":
        return np.searchsorted(np.array([0.5, 0.8, 0.95]), D["cand_cat_in_hist_share"], side="left")
    if kind == "ncat":
        return np.searchsorted(np.array([4, 8, 12, 18], float), D["n_cat"], side="left")
    if kind == "topcat":
        return D["top_cat"].astype(int)
    if kind == "MxcoldxH":
        return bins(D, "Mxcold") * 10 + bins(D, "H")
    raise KeyError(kind)


def poststrat_weights(Dv, Dt, mask_t, kind):
    """val weights (mean 1) that match the val cell distribution to the test[mask_t] cell distribution; cells absent
    from val get mass 0 (the lost test mass is reported)."""
    bv, bt = bins(Dv, kind), bins(Dt, kind)[mask_t]
    cells = np.unique(np.r_[bv, bt])
    pv = np.array([(bv == c).mean() for c in cells]); pt = np.array([(bt == c).mean() for c in cells])
    w = np.zeros(len(bv)); lost = 0.0
    for c, a, b in zip(cells, pv, pt):
        if a > 0:
            w[bv == c] = b / a
        else:
            lost += b
    return w / w.mean(), lost


def neff(w):
    return w.sum() ** 2 / (w ** 2).sum()


def wmean_se(x, w=None):
    x = np.asarray(x, np.float64)
    if w is None:
        return x.mean(), x.std() / np.sqrt(len(x))
    w = np.asarray(w, np.float64)
    m = np.average(x, weights=w)
    return m, np.sqrt((w ** 2 * (x - m) ** 2).sum()) / w.sum()


def wcov(Dm, w=None):
    """weighted per-feed covariance of the columns of Dm (n x K)."""
    w = np.ones(len(Dm)) if w is None else np.asarray(w, np.float64)
    mu = (w[:, None] * Dm).sum(0) / w.sum(); R = Dm - mu
    return (w[:, None] * R).T @ R / w.sum()


def pub_delta(name):
    return MODERN[name]["pub"] - MODERN["G3"]["pub"]


def fmt(m, se):
    return f"{m:+.5f}±{se:.5f}" if se is not None else f"{m:+.5f}"
