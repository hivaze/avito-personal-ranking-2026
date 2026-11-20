"""Shared helpers for the clu_ family (embedding clusters + dense collaborative target encoding)."""
import json, sys, time
from pathlib import Path
import numba as nb
import numpy as np
import polars as pl

ROOT = Path("<repo>/data")
FLAT = ROOT / "flat"
FEATS = ROOT / "feats"
WORK = ROOT / "clu_work"
IKS = (256, 1024, 4096)     # item cluster resolutions
UKS = (64, 256, 1024)       # user cluster resolutions
D = 15                      # train calendar days (MSK) 0..14 ; day 0 = Sun Sep 6
T0 = time.time()


def log(*a):
    print(f"[{time.time()-T0:7.0f}s]", *a, flush=True)


def day_index(ts):
    """MSK calendar day relative to Sun Sep 6 2026 (train day 0). val = 16..19, test = 20..23."""
    return ((ts.astype(np.int64) + 3 * 3600) // 86400 - 20702).astype(np.int64)


def feeds(split):
    f = pl.read_parquet(FLAT / split / "feeds.parquet")
    return dict(M=f["M"].to_numpy().astype(np.int64), H=f["H"].to_numpy().astype(np.int64),
                co=f["cand_off"].to_numpy().astype(np.int64), ho=f["hist_off"].to_numpy().astype(np.int64),
                ts=f["timestamp"].to_numpy().astype(np.int64), uid=f["user_id"].to_numpy().astype(np.int64))


def cand_cl(split, K):
    p = WORK / f"cand_cl_{K}_{split}.npy"
    if not p.exists():
        e = np.load(FLAT / split / "cand_emb_idx.npy")
        t = np.load(WORK / f"item_cl_{K}_{split}.npy")
        np.save(p, np.where(e >= 0, t[np.maximum(e, 0)], -1).astype(np.int16))
    return np.load(p, mmap_mode="r")


def hist_cl(split, K):
    p = WORK / f"hist_cl_{K}_{split}.npy"
    if not p.exists():
        e = np.load(FLAT / split / "hist_emb_idx.npy")
        t = np.load(WORK / f"item_cl_{K}_{split}.npy")
        np.save(p, np.where(e >= 0, t[np.maximum(e, 0)], -1).astype(np.int16))
    return np.load(p, mmap_mode="r")


def cand_ccos(split, K):
    e = np.load(FLAT / split / "cand_emb_idx.npy")
    t = np.load(WORK / f"item_ccos_{K}_{split}.npy")
    return np.where(e >= 0, t[np.maximum(e, 0)].astype(np.float32), np.nan).astype(np.float32)


def user_cl(split, K):
    return np.load(WORK / f"user_cl_{K}_{split}.npy").astype(np.int64)


# final KEEP list (see reports/feat_clu_.md). With env CLU_KEEP_ONLY=1, build.py writes only these files.
KEEP = ["clu_hcnt1024", "clu_hmin1024", "clu_hrw4096", "clu_het14096", "clu_pmi1024_mean", "clu_pmi11024_mean"]


def save_feat(split, name, arr):
    assert name.startswith("clu_")
    import os
    if os.environ.get("CLU_KEEP_ONLY") and name not in KEEP:
        return
    np.save(FEATS / split / f"{name}.npy", np.asarray(arr, dtype=np.float32))


# --------------------------------------------------------------------------- position model for expected clicks
POS_EDGES = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 50, 100, 200, 10_000])
RFE_EDGES = np.array([0, 1, 2, 3, 4, 5, 10, 20, 10_000])
M_EDGES = np.array([0, 10, 20, 40, 80, 150, 10_000])
NPB, NRB, NMB = len(POS_EDGES) - 1, len(RFE_EDGES) - 1, len(M_EDGES) - 1


@nb.njit(parallel=True, cache=True)
def rank_in_feed(co, M, pos, out):
    for f in nb.prange(len(M)):
        o = co[f]; m = M[f]
        order = np.argsort(pos[o:o + m])
        for r in range(m):
            out[o + order[r]] = r
