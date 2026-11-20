"""Shared helpers for the val->public shift diagnosis (nn_x/pubdiag/shift).
CPU only. No existing file is modified."""
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
OUT = ROOT / "nn_x/pubdiag/shift"
BOUND = 1790497920  # public = test feeds with timestamp < BOUND

FAM = {
    "F5cd": ["v15_cd_all_e10", "v15_cd_all_s81", "v15_cd_all_s82", "v15_cd_allprew_s83", "v15_cd_allprew_s84",
             "v15_cd_allprew_s85"],
    "F6": ["v13_lr1_mix1"] + [f"v14_s{i}" for i in range(92, 97)],
    "F7": [f"v17_pre2s_s{i}" for i in range(101, 108)],
    "F9": [f"v22_pre2s_lr1_s{i}" for i in range(401, 407)],
    "F10": [f"v23_gru_pre2s_s{i}" for i in range(501, 507)],
    "F12": [f"v26_d192_pre2s_s{i}" for i in range(701, 708)],
    "F7x": [f"r6_cd_m02_r{i}" for i in range(101, 108)],
    "F7x70": [f"r6_cd_m02_p70k_r{i}" for i in range(101, 108)],
    "F7b": [f"r6_cd_c0_r{i}" for i in range(101, 108)],
}
POOLS = {
    "Cd": FAM["F5cd"] + FAM["F6"] + FAM["F7"],
    "PF7x": FAM["F5cd"] + FAM["F6"] + FAM["F7x"] + FAM["F9"] + FAM["F10"] + FAM["F12"],
    "G3": FAM["F10"] + FAM["F7x"] + FAM["F7x70"],
    "M2": FAM["F7x"] + FAM["F7x70"],
    "F7x": FAM["F7x"], "F7b": FAM["F7b"], "F7": FAM["F7"], "F7x70": FAM["F7x70"],
}


def feed_meta(split):
    f = pl.read_parquet(FLAT / split / "feeds.parquet")
    return f, f["M"].to_numpy().astype(np.int64), f["cand_off"].to_numpy().astype(np.int64)


def zfeed(s, M):
    """per-feed z-score of a flat score array (population std), float64"""
    s = np.asarray(s, np.float64)
    n = len(M); fid = np.repeat(np.arange(n), M)
    mu = np.bincount(fid, s, n) / M
    sd = np.sqrt(np.maximum(np.bincount(fid, s * s, n) / M - mu ** 2, 0)) + 1e-9
    return (s - mu[fid]) / sd[fid]


def zmean(arrs, M):
    acc = None
    for a in arrs:
        z = zfeed(a, M)
        acc = z if acc is None else acc + z
    return acc / len(arrs)


def holdout_index():
    """train feed indices of the 30K holdout (identical selection to nn/train.py --holdout / nn/holdout_eval.py)."""
    ftr = pl.read_parquet(FLAT / "train/feeds.parquet")
    Mtr = ftr["M"].to_numpy().astype(np.int64)
    pr = np.load(ROOT / "eda/labels_position/out/train_prandom.npz")
    assert (pr["fidx"] == np.arange(len(ftr))).all()
    p = np.nan_to_num(pr["p_random"].astype(np.float32), nan=0.0)
    p[Mtr < 50] = 0.0
    cand = np.where((p > 0.9) & (Mtr >= 50))[0]
    hold = np.sort(np.random.default_rng(12345).choice(cand, 30000, replace=False))
    Mh, offh = np.load(P / "holdout/holdout_meta.npy")
    assert (Mtr[hold] == Mh).all(), "holdout reconstruction mismatch"
    return hold, ftr[hold], Mh.astype(np.int64), offh.astype(np.int64)


def holdout_pool(members, M):
    return zmean([np.load(P / f"holdout/nn_{m}_holdout.npy") for m in members], M)


def split_pool(members, split, M):
    """final pool score = 0.5*z(mean_m z(ft_m)) + 0.5*z(mean_m z(to_m)) (member-equal spec of build_pool.py)"""
    ftn, ton = ("n2cpv3_val_oof", "trainonly_val") if split == "val" else ("n2cpv3_test", "trainonly_test")
    ft = zmean([np.load(P / f"nn_{m}_{ftn}.npy") for m in members], M)
    to = zmean([np.load(P / f"nn_{m}_{ton}.npy") for m in members], M)
    return 0.5 * zfeed(ft, M) + 0.5 * zfeed(to, M), ft, to


def msk_day(ts):
    return (np.asarray(ts, np.int64) + 10800) // 86400


def msk_hour(ts):
    return ((np.asarray(ts, np.int64) + 10800) % 86400) // 3600


def weekday(ts):
    """0=Mon..6=Sun (1970-01-01 was a Thursday)"""
    return (msk_day(ts) + 3) % 7


def mse(x, w=None):
    x = np.asarray(x, np.float64)
    if w is None:
        return x.mean(), x.std() / np.sqrt(len(x))
    w = np.asarray(w, np.float64)
    m = np.average(x, weights=w); neff = w.sum() ** 2 / (w ** 2).sum()
    return m, np.sqrt(np.average((x - m) ** 2, weights=w) / neff)


def fmt(m, se):
    return f"{m:+.5f} ± {se:.5f} (z {m / se:+.1f})" if se > 0 else f"{m:+.5f}"
