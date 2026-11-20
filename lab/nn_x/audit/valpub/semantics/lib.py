"""Shared helpers for nn_x/audit/valpub/semantics (val semantics vs test). CPU only, 12 threads, new files only."""
import os
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "POLARS_MAX_THREADS", "NUMBA_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[k] = "12"
import sys
from pathlib import Path
import numpy as np
import polars as pl
sys.path.insert(0, "<repo>/src")
from metrics import ndcg_per_feed  # noqa: E402

ROOT = Path("<repo>")
FLAT = ROOT / "data/flat"
P = ROOT / "preds"
FC = ROOT / "nn_x/r6/feat_cache"
SHIFT = ROOT / "nn_x/pubdiag/shift"
SEG = ROOT / "eda/errors/segments"
HERE = ROOT / "nn_x/audit/valpub/semantics"
OUT = HERE / "out"
OUT.mkdir(parents=True, exist_ok=True)
BOUND = 1790497920
MSK = 10800

# public-scored modern pools (exact public score on the 8,562 public feeds) + v10 (old generation, arrays exist)
PUBLIC = {"Cd": 0.39208, "Pall": 0.392436, "PF7x": 0.392723, "G3": 0.392820, "M2": 0.392434, "S2": 0.392473,
          "TOonly": 0.392346, "v10": 0.38786}
LEVEL_ONLY = {"v3": (0.3816, 0.3739), "v4": (0.3839, 0.3772), "v5": (0.3869, 0.3802)}  # (val, public), no arrays
MODERN = ["Cd", "Pall", "PF7x", "G3", "M2", "S2", "TOonly"]
# pools selected on val after the fact (winner's-curse candidates) vs pools built as references/hedges
VAL_SELECTED = {"G3": 1, "M2": 1, "S2": 1, "TOonly": 0, "PF7x": 0, "Cd": 0, "Pall": 0}
RECOOL_SHARE = {"Cd": 0.0, "Pall": 0.0, "PF7x": 0.18, "G3": 0.70, "M2": 1.0, "S2": 0.70, "TOonly": 0.70}


def feed_meta(split):
    f = pl.read_parquet(FLAT / split / "feeds.parquet")
    return f, f["M"].to_numpy().astype(np.int64), f["cand_off"].to_numpy().astype(np.int64)


def zfeed(s, M):
    s = np.asarray(s, np.float64)
    n = len(M); fid = np.repeat(np.arange(n), M)
    mu = np.bincount(fid, s, n) / M
    sd = np.sqrt(np.maximum(np.bincount(fid, s * s, n) / M - mu ** 2, 0)) + 1e-9
    return (s - mu[fid]) / sd[fid]


def pool_scores(name, split):
    _, M, _ = feed_meta(split)
    if name in ("Cd", "Pall", "PF7x", "M2"):
        return 0.5 * zfeed(np.load(P / f"ens_{name}_ft_{split}.npy"), M) + 0.5 * zfeed(np.load(P / f"ens_{name}_to_{split}.npy"), M)
    if name == "G3":
        return np.load(P / f"ens_G3_final_{split}.npy").astype(np.float64)
    if name == "S2":
        return np.load(P / f"nnstk_g3gate3r_{split}.npy").astype(np.float64)
    if name == "TOonly":
        return np.load(P / f"ens_G3_toonly_{split}.npy").astype(np.float64)
    if name == "G3ft":
        return np.load(P / f"ens_G3_ft_{split}.npy").astype(np.float64)
    if name == "G3to":
        return np.load(P / f"ens_G3_to_{split}.npy").astype(np.float64)
    if name == "v10":
        oof = "val_oof" if split == "val" else "test"
        to = "trainonly_val" if split == "val" else "trainonly_test"
        return (0.5 * zfeed(np.load(P / f"nn_final4_{oof}.npy"), M) + 0.5 * zfeed(np.load(P / f"nn_final4_{to}.npy"), M)
                + 0.1 * zfeed(np.load(P / f"pre_final_{split}.npy"), M))
    raise KeyError(name)


_PF = {}


def per_feed(name):
    """per-feed val NDCG of a pool (cached on disk)."""
    if name in _PF:
        return _PF[name]
    fp = OUT / f"pf_{name}.npy"
    if fp.exists():
        _PF[name] = np.load(fp); return _PF[name]
    f, M, off = feed_meta("val"); y = np.load(FLAT / "val/cand_target.npy")
    pf = ndcg_per_feed(pool_scores(name, "val").astype(np.float32), y, off, M)
    np.save(fp, pf); _PF[name] = pf
    return pf


def age_hours(f4):
    """listing age in hours from f4 (feats_v2/life_/post_time.py knots; f4 >= 63/64 -> capped, returned as knot value)."""
    kn = np.load(ROOT / "data/feats/life_f4_age_knots.npy")
    K = len(kn)
    u = np.asarray(f4, np.float64) * K
    k = np.clip(np.floor(u).astype(int), 0, K - 2); w = np.clip(u - k, 0, 1)
    return kn[k] * (1 - w) + kn[k + 1] * w


def descriptors(split):
    """label-free per-feed descriptors (row = fidx). Reuses nn_x/pubdiag/shift/desc_*.parquet and adds calendar + age."""
    fp = OUT / f"desc_{split}.parquet"
    if fp.exists():
        return pl.read_parquet(fp)
    f, M, off = feed_meta(split)
    d = pl.read_parquet(SHIFT / f"desc_{split}.parquet")
    n = len(M); fid = np.repeat(np.arange(n), M)
    ts = f["timestamp"].to_numpy().astype(np.int64)
    dayidx = (ts + MSK) // 86400
    dow = (dayidx + 3) % 7  # 1970-01-01 was a Thursday -> (dayidx+3)%7: 0=Mon
    f4 = np.load(FLAT / f"{split}/cand_f4.npy").astype(np.float64)
    age = age_hours(f4)
    capped = f4 >= 63 / 64 - 1e-6
    extra = {
        "dayidx": dayidx, "dow": dow, "weekend": (dow >= 5).astype(np.int64),
        "public": (ts < BOUND).astype(np.int64) if split == "test" else np.full(n, -1),
        "age_logmean_h": np.bincount(fid, np.log1p(age), n) / M,
        "age_lt1d_share": np.bincount(fid, age < 24, n) / M,
        "age_lt7d_share": np.bincount(fid, age < 24 * 7, n) / M,
        "age_capped_share": np.bincount(fid, capped, n) / M,
        "age_median_h": np.array([np.median(age[o:o + m]) for o, m in zip(off, M)]),
        "H0": (f["H"].to_numpy() == 0).astype(np.int64),
        "H256": (f["H"].to_numpy() == 256).astype(np.int64),
        "train_user": (d["n_train_feeds"].to_numpy() > 0).astype(np.int64),
        "hourb": np.searchsorted(np.array([5, 9, 13, 17]), d["hour"].to_numpy(), side="right"),  # 0..4 = segdefs hour bins
    }
    if split == "val":
        seg = pl.read_parquet(SEG / "feed_table.parquet")
        extra["npos"] = seg["npos"].to_numpy(); extra["day"] = seg["day"].to_numpy()
        extra["leak"] = seg["leak"].to_numpy().astype(np.int64)
    out = d.with_columns([pl.Series(k, v) for k, v in extra.items()])
    out.write_parquet(fp)
    return out


def wmean_se(x, w=None):
    x = np.asarray(x, np.float64)
    if w is None:
        return float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x)))
    w = np.asarray(w, np.float64)
    m = np.average(x, weights=w)
    neff = w.sum() ** 2 / (w ** 2).sum()
    return float(m), float(np.sqrt(np.average((x - m) ** 2, weights=w) / neff))


def cell_weights(kv, kt, mask_t):
    """val weights so that the distribution of cell key kv matches that of kt[mask_t] (mean 1). Unsupported test cells
    are reported as mass dropped."""
    w = np.zeros(len(kv))
    uv = set(np.unique(kv).tolist())
    ktm = kt[mask_t]
    dropped = 0.0
    for k, c in zip(*np.unique(ktm, return_counts=True)):
        if k in uv:
            w[kv == k] = (c / len(ktm)) / ((kv == k).mean())
        else:
            dropped += c / len(ktm)
    return w / w.mean(), dropped


def js_div(p, q):
    p = np.asarray(p, np.float64) + 1e-12; q = np.asarray(q, np.float64) + 1e-12
    p /= p.sum(); q /= q.sum(); m = 0.5 * (p + q)
    return float(0.5 * (p * np.log2(p / m)).sum() + 0.5 * (q * np.log2(q / m)).sum())


def smd(a, b):
    """standardised mean difference (b - a) / pooled sd"""
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    s = np.sqrt(0.5 * (a.var(ddof=1) + b.var(ddof=1))) + 1e-12
    return float((b.mean() - a.mean()) / s)


def random_ndcg(M, npos):
    """expected full-list NDCG of a uniformly random ranking given M and npos (binary gains)."""
    M = np.asarray(M, np.int64); npos = np.asarray(npos, np.int64)
    out = np.ones(len(M))
    disc = 1.0 / np.log2(np.arange(1, M.max() + 1) + 1.0)
    cum = np.cumsum(disc)
    for i, (m, k) in enumerate(zip(M, npos)):
        if k <= 0:
            continue
        e_dcg = k * cum[m - 1] / m
        out[i] = e_dcg / cum[min(k, m) - 1]
    return out


def fmt(x, se=None, d=5):
    return f"{x:+.{d}f}" + (f"±{se:.{d}f}" if se is not None else "")
