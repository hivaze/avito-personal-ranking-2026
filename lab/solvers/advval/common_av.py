"""Shared helpers for the adversarial-validation study (CPU only, read-only on project data)."""
import os
for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS"]:
    os.environ[k] = "16"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys
import numpy as np
import polars as pl

sys.path.insert(0, "<repo>/src")
from metrics import ndcg_per_feed  # noqa: E402

ROOT = "<repo>"
FLAT = f"{ROOT}/data/flat"
FEATS = f"{ROOT}/data/feats"
FEATS_NEW = f"{ROOT}/data/feats_new"
CACHE = f"{ROOT}/nn/feat_cache"
PREDS = f"{ROOT}/preds"
OUT = f"{ROOT}/solvers/out/advval"
SCR = "/tmp/scratchpad/solvers/advval"
BOUND = 1790497920  # public = test ts < BOUND
MSK = 3 * 3600
os.makedirs(OUT, exist_ok=True)
os.makedirs(SCR, exist_ok=True)


def meta(split):
    f = pl.read_parquet(f"{FLAT}/{split}/feeds.parquet")
    return f, f["M"].to_numpy().astype(np.int64), f["cand_off"].to_numpy().astype(np.int64)


def feat_file(split, name):
    """Read-only path resolution: nn/feat_cache first (what the NN reads), then data/feats, then data/feats_new."""
    for d in (CACHE, FEATS, FEATS_NEW):
        p = f"{d}/{split}/{name}.npy"
        if os.path.exists(p):
            return p
    return None


def load_feat(split, name, rows=None):
    p = feat_file(split, name)
    x = np.load(p, mmap_mode="r")
    return np.asarray(x if rows is None else x[rows], dtype=np.float32)


def rows_of(off, M, fidx):
    """candidate row indices of the given feeds (in feed order)"""
    o, m = off[fidx], M[fidx]
    return np.concatenate([np.arange(a, a + b) for a, b in zip(o, m)]) if len(fidx) else np.zeros(0, np.int64)


def train_random_like():
    """random-like train feeds = the nn/train.py holdout pool definition: p_random > 0.9 and M >= 50"""
    f, M, off = meta("train")
    pr = np.nan_to_num(np.load(f"{ROOT}/eda/labels_position/out/train_prandom.npz")["p_random"], nan=0.0)
    return np.where((pr > 0.9) & (M >= 50))[0]


def feed_z(off, M, x):
    x = np.asarray(x, np.float64)
    fid = np.repeat(np.arange(len(M)), M)
    s1 = np.bincount(fid, x, len(M)); s2 = np.bincount(fid, x * x, len(M))
    mu = s1 / M; sd = np.sqrt(np.maximum(s2 / M - mu * mu, 0)) + 1e-9
    return (x - mu[fid]) / sd[fid]


def load_pred(name):
    return np.load(f"{PREDS}/{name}.npy").astype(np.float64)


def wmean_se(x, w):
    """weighted mean and SE (ratio estimator, independent feeds)"""
    w = np.asarray(w, float); x = np.asarray(x, float)
    w = w / w.sum()
    m = (w * x).sum()
    return m, np.sqrt((w ** 2 * (x - m) ** 2).sum())


def psi(a, b, bins=10):
    """population stability index of b relative to a (quantile bins of a; ties handled by unique edges)"""
    a = np.asarray(a, float); b = np.asarray(b, float)
    na_a, na_b = np.isnan(a).mean(), np.isnan(b).mean()
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) == 0 or len(b) == 0:
        return np.nan
    ua = np.unique(a)
    if len(ua) <= 20:  # discrete feature: categorical PSI over the observed values
        vals = np.unique(np.r_[ua, np.unique(b)])
        pa = np.r_[np.array([(a == v).mean() for v in vals]) * (1 - na_a), na_a]
        pb = np.r_[np.array([(b == v).mean() for v in vals]) * (1 - na_b), na_b]
        pa = np.maximum(pa, 1e-4); pb = np.maximum(pb, 1e-4)
        return float(((pb - pa) * np.log(pb / pa)).sum())
    edges = np.unique(np.quantile(a, np.linspace(0, 1, bins + 1)[1:-1]))
    pa = np.bincount(np.searchsorted(edges, a, side="right"), minlength=len(edges) + 1) / len(a) * (1 - na_a)
    pb = np.bincount(np.searchsorted(edges, b, side="right"), minlength=len(edges) + 1) / len(b) * (1 - na_b)
    pa = np.r_[pa, na_a]; pb = np.r_[pb, na_b]
    eps = 1e-4
    pa = np.maximum(pa, eps); pb = np.maximum(pb, eps)
    return float(((pb - pa) * np.log(pb / pa)).sum())
