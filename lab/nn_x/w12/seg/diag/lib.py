"""Shared helpers for nn_x/w12/seg/diag (CPU only). Read-only on everything outside this directory."""
import os
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS"):
    os.environ.setdefault(k, "12")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys
sys.path.insert(0, "<repo>/src")
from pathlib import Path
import numpy as np, polars as pl
from common import feed_meta, FLAT
from metrics import ndcg_per_feed
from stack_val import feed_rank_z

ROOT = Path("<repo>")
P = ROOT / "preds"
OUT = ROOT / "nn_x/w12/seg/diag/out"
OUT.mkdir(exist_ok=True)
FC = ROOT / "nn/feat_cache/val"

fv, Mv, ov = feed_meta("val")
yv = np.load(FLAT / "val/cand_target.npy")
Hv = fv["H"].to_numpy().astype(np.int64)
uid = fv["user_id"].to_numpy().astype(np.int64)
NF = len(Mv)
fid = np.repeat(np.arange(NF), Mv)
# clusters: user_id, except user 0 (pooled anonymous bucket) where each feed is its own cluster
clu = np.where(uid == 0, -1 - np.arange(NF), uid)
_, clu = np.unique(clu, return_inverse=True)
NCL = clu.max() + 1

Z = lambda x: feed_rank_z(ov, Mv, np.asarray(x, np.float32))[:, 1]
ev = lambda s: ndcg_per_feed(np.asarray(s, np.float32), yv, ov, Mv)
fin = lambda ft, to: 0.5 * Z(ft) + 0.5 * Z(to)

BUCK = [("H=0", 0, 0), ("1-4", 1, 4), ("5-9", 5, 9), ("10-19", 10, 19), ("20-49", 20, 49), ("50-99", 50, 99),
        ("100-255", 100, 255), ("H=256", 256, 10 ** 9)]
bmask = {n: (Hv >= a) & (Hv <= b) for n, a, b in BUCK}
REST = Hv < 256
NEAR = (Hv >= 100) & (Hv <= 255)
CAP = Hv >= 256
SEGS = dict(ALL=np.ones(NF, bool), **bmask, REST=REST, NEAR=NEAR, **{"H<=9": Hv <= 9, "H1-9": (Hv >= 1) & (Hv <= 9), "uid0": uid == 0})


def cse(x, m=None):
    """mean and cluster-robust SE (by user) of x over the feeds in mask m; NaN entries are dropped."""
    x = np.asarray(x, np.float64)
    ok = ~np.isnan(x) if m is None else (m & ~np.isnan(x))
    n = int(ok.sum())
    if n == 0:
        return np.nan, np.nan, 0
    mu = x[ok].mean()
    r = np.bincount(clu[ok], weights=x[ok] - mu, minlength=NCL)
    g = int((np.bincount(clu[ok], minlength=NCL) > 0).sum())
    se = np.sqrt((r ** 2).sum() * (g / max(g - 1, 1))) / n
    return mu, se, n


def cdiff(x, a, b):
    """difference of means of x between feed sets a and b (possibly sharing users): cluster-robust SE of the contrast."""
    x = np.asarray(x, np.float64)
    oka, okb = a & ~np.isnan(x), b & ~np.isnan(x)
    na, nb = oka.sum(), okb.sum()
    ma, mb = x[oka].mean(), x[okb].mean()
    infl = np.zeros(NF)
    infl[oka] += (x[oka] - ma) / na
    infl[okb] -= (x[okb] - mb) / nb
    r = np.bincount(clu, weights=infl, minlength=NCL)
    return ma - mb, np.sqrt((r ** 2).sum())


def feed_auc(s):
    """per-feed AUC = mean over positives of the share of negatives ranked below (ties 0.5); NaN if no pos or no neg."""
    s = np.asarray(s, np.float64)
    order = np.lexsort((s, fid))
    ss, yy = s[order], yv[order].astype(np.float64)
    # mid-ranks within feed
    pos_in = np.arange(len(ss)) - np.repeat(ov, Mv)
    # handle ties: average rank among equal scores within a feed
    new = np.ones(len(ss), bool)
    new[1:] = (ss[1:] != ss[:-1]) | (fid[1:] != fid[:-1])
    gid = np.cumsum(new) - 1
    gsum = np.bincount(gid, weights=pos_in)
    gcnt = np.bincount(gid)
    mid = (gsum / gcnt)[gid]  # 0-based mid-rank ascending
    npos = np.bincount(fid, weights=yy, minlength=NF)
    nneg = Mv - npos
    rs = np.bincount(fid, weights=mid * yy, minlength=NF)
    with np.errstate(invalid="ignore", divide="ignore"):
        auc = (rs - npos * (npos - 1) / 2) / (npos * nneg)
    auc[(npos == 0) | (nneg == 0)] = np.nan
    return auc


_disc = 1.0 / np.log2(np.arange(2, 2002))
_cdisc = np.concatenate([[0.0], np.cumsum(_disc)])
NPOS = np.bincount(fid, weights=yv, minlength=NF).astype(np.int64)


def rand_ndcg():
    """exact expectation of NDCG under a uniformly random order given (M, n_pos)."""
    e = np.ones(NF)
    ok = NPOS > 0
    e[ok] = NPOS[ok] * (_cdisc[Mv[ok]] / Mv[ok]) / _cdisc[NPOS[ok]]
    return e


def tab(rows, hdr):
    w = [max(len(str(r[i])) for r in rows + [hdr]) for i in range(len(hdr))]
    line = lambda r: "| " + " | ".join(str(r[i]).ljust(w[i]) for i in range(len(hdr))) + " |"
    return "\n".join([line(hdr), "|" + "|".join("-" * (x + 2) for x in w) + "|"] + [line(r) for r in rows])


def load_final():
    return fin(np.load(P / "ens_FD16P20kd_ft_val.npy"), np.load(P / "ens_FD16P20kd_to_val.npy"))
