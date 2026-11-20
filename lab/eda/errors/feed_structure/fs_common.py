"""Shared loaders for the feed-structure / multi-positive lens (val split)."""
import os, sys, json
os.environ.setdefault("NUMBA_NUM_THREADS", "16")
for v in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
    os.environ.setdefault(v, "16")
from pathlib import Path
import numpy as np, polars as pl
sys.path.insert(0, "<repo>/src")
from metrics import ndcg_per_feed

ROOT = Path("<repo>")
FLAT = ROOT / "data/flat"
PREDS = ROOT / "preds"
OUT = ROOT / "eda/errors/feed_structure"
CACHE = OUT / "cache"


def leak_mask_val():
    """True for the val feeds whose (non-zero) user has a later feed in val or test (to be EXCLUDED)."""
    v = pl.read_parquet(FLAT / "val/feeds.parquet").select("fidx", "user_id", "timestamp")
    t = pl.read_parquet(FLAT / "test/feeds.parquet").select("user_id", "timestamp")
    allf = pl.concat([v.select("user_id", "timestamp"), t])
    mx = allf.group_by("user_id").agg(pl.col("timestamp").max().alias("tmax"))
    j = v.join(mx, on="user_id", how="left").sort("fidx")
    m = ((j["tmax"] > j["timestamp"]) & (j["user_id"] != 0)).to_numpy()
    return m


def load_val():
    f = pl.read_parquet(FLAT / "val/feeds.parquet")
    M = f["M"].to_numpy().astype(np.int64)
    off = f["cand_off"].to_numpy().astype(np.int64)
    d = {"feeds": f, "M": M, "off": off}
    for n in ["target", "category_id", "item_id", "emb_idx", "f1", "f2", "f3", "f4", "f5"]:
        d[n] = np.load(FLAT / f"val/cand_{n}.npy")
    d["fid"] = np.repeat(np.arange(len(M)), M)
    d["leak"] = leak_mask_val()
    return d


def load_emb_norm(emb_idx):
    meta = json.load(open(ROOT / "data/competition/val/item_embeddings/data.meta"))
    E = np.memmap(ROOT / "data/competition/val/item_embeddings/data.memmap", dtype=np.int8, mode="r", shape=tuple(meta["shape"]))
    miss = emb_idx < 0
    X = np.asarray(E[np.where(miss, 0, emb_idx)], dtype=np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
    X[miss] = 0.0
    return X, miss


def zfeed(x, off, M):
    fid = np.repeat(np.arange(len(M)), M)
    x = x.astype(np.float64)
    s = np.bincount(fid, x, len(M)); s2 = np.bincount(fid, x * x, len(M))
    mu = s / M; sd = np.sqrt(np.maximum(s2 / M - mu * mu, 0)) + 1e-9
    return ((x - mu[fid]) / sd[fid])


def paired(a, b, mask=None):
    d = a - b
    if mask is not None:
        d = d[mask]
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d))), int(len(d))
