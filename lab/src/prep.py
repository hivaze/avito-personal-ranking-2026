"""Flatten nested parquet feeds into memory-mappable numpy arrays.

Output per split in data/flat/<split>/:
  feeds.parquet         feed-level: fidx, feed_id, timestamp, user_id, M, H, cand_off, hist_off
  user_emb.npy          int8 [n_feeds,128]
  cand_*.npy            candidate-level flat arrays (feed order preserved): fidx,item_id,category_id,f1..f5,emb_idx,[pos],[target]
  hist_*.npy            history-level flat arrays: fidx,item_id,category_id,event_type,event_source,emb_idx,hpos (0=newest)
"""
import sys, time
from pathlib import Path
import numpy as np
import polars as pl

ROOT = Path("<repo>/data")
CAND = ["item_id", "category_id", "f1", "f2", "f3", "f4", "f5", "item_id__embedding_idx"]
HIST = ["history__item_id", "history__category_id", "history__event_type", "history__event_source",
        "history__item_id__embedding_idx"]


def flat_list(s: pl.Series) -> np.ndarray:
    return s.explode().to_numpy() if s.dtype == pl.List else s.to_numpy()


def run(split: str):
    t = time.time()
    out = ROOT / "flat" / split
    out.mkdir(parents=True, exist_ok=True)
    files = sorted((ROOT / "competition" / split / "impressions").glob("*.parquet"))
    df = pl.read_parquet(files)
    df = df.sort(["timestamp", "feed_id"])  # chronological order
    n = len(df)
    extra = [c for c in ("pos", "target") if c in df.columns]
    M = df["item_id"].list.len().to_numpy().astype(np.int64)
    H = df["history__item_id"].list.len().to_numpy().astype(np.int64)
    feeds = pl.DataFrame({
        "fidx": np.arange(n, dtype=np.int32),
        "feed_id": df["feed_id"], "timestamp": df["timestamp"], "user_id": df["user_id"],
        "M": M.astype(np.int32), "H": H.astype(np.int32),
        "cand_off": np.r_[0, np.cumsum(M)[:-1]].astype(np.int64),
        "hist_off": np.r_[0, np.cumsum(H)[:-1]].astype(np.int64),
    })
    feeds.write_parquet(out / "feeds.parquet")
    ue = np.stack(df["user_embedding"].to_numpy()) if False else df["user_embedding"].list.to_array(128).to_numpy()
    np.save(out / "user_emb.npy", ue.astype(np.int8))
    np.save(out / "cand_fidx.npy", np.repeat(np.arange(n, dtype=np.int32), M))
    for c in CAND + extra:
        name = "emb_idx" if c == "item_id__embedding_idx" else c
        arr = df[c].explode().to_numpy()
        assert len(arr) == M.sum(), c
        np.save(out / f"cand_{name}.npy", arr)
    np.save(out / "hist_fidx.npy", np.repeat(np.arange(n, dtype=np.int32), H))
    np.save(out / "hist_hpos.npy", (np.arange(H.sum()) - np.repeat(np.r_[0, np.cumsum(H)[:-1]], H)).astype(np.int16))
    for c in HIST:
        name = c.replace("history__", "").replace("item_id__embedding_idx", "emb_idx")
        s = df[c]
        arr = s.explode().drop_nulls().to_numpy() if True else None
        # explode of empty list yields null; drop them
        assert len(arr) == H.sum(), (c, len(arr), H.sum())
        np.save(out / f"hist_{name}.npy", arr)
    print(split, n, M.sum(), H.sum(), f"{time.time()-t:.1f}s", flush=True)


if __name__ == "__main__":
    for s in sys.argv[1:]:
        run(s)
