# Second pass of the error analysis (2026-11-11, evening), on the pool that was best that day (0.39793 on val).
# Shared loaders for the s01 ... s17 scripts: feed metadata, per-feed NDCG, within-feed z-scores and ranks.
# Imported by the other scripts of this pass; not run on its own. CPU only.
# It reads intermediate arrays from a scratch folder (SCR) that was not kept, so the scripts of this pass
# document the method and do not run as copied. Their outputs are in lab/solvers/out/errors/.
# Kept as it ran: paths and names refer to the lab tree (see lab/ and docs/glossary.md).
# Original: lab/solvers/errors/common_e.py
"""Shared loaders for the v11 error analysis (CPU only). Read-only on the project."""
import os
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS"):
    os.environ.setdefault(k, "16")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys
import numpy as np
import polars as pl

ROOT = "<repo>"
SCR = "/tmp/scratchpad/solvers/errors"
sys.path.insert(0, f"{ROOT}/src")
from metrics import ndcg_per_feed  # noqa: E402

PRIV_BOUND = 1790497920


def meta(split="val"):
    f = pl.read_parquet(f"{ROOT}/data/flat/{split}/feeds.parquet")
    M = f["M"].to_numpy().astype(np.int64)
    off = f["cand_off"].to_numpy().astype(np.int64)
    fid = np.repeat(np.arange(len(M)), M)
    return f, M, off, fid


def fz(s, fid, M):
    s = np.asarray(s, dtype=np.float64)
    mu = np.bincount(fid, s, len(M)) / M
    sd = np.sqrt(np.maximum(np.bincount(fid, s * s, len(M)) / M - mu ** 2, 0)) + 1e-6
    return (s - mu[fid]) / sd[fid]


def pred(name):
    return np.load(f"{ROOT}/preds/{name}.npy", mmap_mode="r").astype(np.float64)


def v11(split="val"):
    f, M, off, fid = meta(split)
    if split == "val":
        a, b = pred("nn_final5_val_oof"), pred("nn_final5_trainonly_val")
    else:
        a, b = pred("nn_final5_test"), pred("nn_final5_trainonly_test")
    return 0.5 * fz(a, fid, M) + 0.5 * fz(b, fid, M)


def ranks(score, y, fid, off, M):
    """0-based within-feed rank, pessimistic ties (negatives before positives at equal score)."""
    order = np.lexsort((y, -score, fid))
    r = np.empty(len(score), np.int64)
    r[order] = np.arange(len(order)) - np.repeat(off, M)
    return r


def pos_loss(score, y, fid, off, M):
    """Per-row share of (1 - NDCG) attributed to each positive (0 on negatives). Sum over a feed = 1 - NDCG."""
    r = ranks(score, y, fid, off, M)
    n = len(M)
    npos = np.bincount(fid, y, n)
    disc = lambda k: 1.0 / np.log2(k + 2.0)  # noqa: E731
    ideal = np.zeros(n)
    # IDCG per feed
    pmax = int(npos.max())
    cs = np.concatenate([[0], np.cumsum(disc(np.arange(pmax)))])
    idcg = cs[npos.astype(int)]
    # ordinal of each positive among the feed's positives (by rank)
    pi = np.where(y > 0)[0]
    o = np.lexsort((r[pi], fid[pi]))
    pis = pi[o]
    fp = fid[pis]
    first = np.r_[0, np.flatnonzero(np.diff(fp)) + 1]
    ordn = np.arange(len(pis)) - np.repeat(first, np.diff(np.r_[first, len(pis)]))
    loss = np.zeros(len(score))
    loss[pis] = (disc(ordn) - disc(r[pis])) / idcg[fp]
    return loss, r
