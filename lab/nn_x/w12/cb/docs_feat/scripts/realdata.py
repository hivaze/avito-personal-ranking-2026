"""Read-only access to the real data for the docs_feat probes (no file outside nn_x/w12/cb/docs_feat is written).
pool_feeds(n, seed): random n feeds of the cb26 exploration pool (p_random > 0.5, M >= 50, minus the 30K holdout), rows contiguous per feed.
pair_embedding(split, rows): float32 [n, 128] = user_emb[feed] * item_emb[cand] / 65536 (its row sum is the cosine <u, e>); missing item -> zeros.
"""
import json, sys
from pathlib import Path
import numpy as np, polars as pl
sys.path.insert(0, "<repo>/nn_x/cb26"); sys.path.insert(0, "<repo>/src")
from cbdata import hold_index, ROOT  # noqa: E402  (read-only import)

FLAT = ROOT / "data/flat"


def _table(split):
    dd = ROOT / f"data/competition/{split}/item_embeddings"
    mt = json.loads((dd / "data.meta").read_text())
    return np.memmap(dd / "data.memmap", dtype=np.int8, mode="r", shape=tuple(mt["shape"]))


def pool_feeds(n_feeds, seed=0):
    hold, ftr = hold_index()
    M = ftr["M"].to_numpy().astype(np.int64); off = ftr["cand_off"].to_numpy().astype(np.int64)
    pr = np.load(ROOT / "eda/labels_position/out/train_prandom.npz")
    p = np.nan_to_num(pr["p_random"].astype(np.float32), nan=0.0); p[M < 50] = 0.0
    cand = np.setdiff1d(np.where((p > 0.5) & (M >= 50))[0], hold)
    assert np.isin(cand, hold).sum() == 0
    sel = np.sort(np.random.default_rng(seed).choice(cand, min(n_feeds, len(cand)), replace=False))
    rows = np.repeat(off[sel] - np.r_[0, np.cumsum(M[sel])[:-1]], M[sel]) + np.arange(M[sel].sum())
    gid = np.repeat(sel, M[sel])
    y = np.asarray(np.load(FLAT / "train/cand_target.npy", mmap_mode="r")[rows], np.float32)
    return sel, rows, gid, y


def pair_embedding(split, rows=None, max_rows=None, with_meta=False, kind="prod"):
    ei = np.load(FLAT / split / "cand_emb_idx.npy", mmap_mode="r")
    fi = np.load(FLAT / split / "cand_fidx.npy", mmap_mode="r")
    if rows is None:
        n = len(ei) if max_rows is None else None
        if n is None:  # cut at a feed boundary
            f = pl.read_parquet(FLAT / split / "feeds.parquet"); cs = np.cumsum(f["M"].to_numpy().astype(np.int64)); n = int(cs[np.searchsorted(cs, max_rows) - 1])
        rows = np.arange(n)
    e = np.asarray(ei[rows]); g = np.asarray(fi[rows])
    T = _table(split); U = np.load(FLAT / split / "user_emb.npy", mmap_mode="r")
    out = np.empty((len(rows), 128), np.float32)
    B = 500_000
    for s in range(0, len(rows), B):
        ee = e[s:s + B]; it = np.asarray(T[np.maximum(ee, 0)], np.float32); it[ee < 0] = 0
        u = np.asarray(U[g[s:s + B]], np.float32)
        if kind == "prod":
            out[s:s + B] = it * u / 65536.0
        elif kind == "item":
            out[s:s + B] = it / 256.0
        elif kind == "diff":
            out[s:s + B] = (it - u) / 256.0
    if with_meta:
        y = np.asarray(np.load(FLAT / split / "cand_target.npy", mmap_mode="r")[rows], np.float32) if split != "test" else None
        return out, g, y
    return out
