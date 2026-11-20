"""Shared loaders for the event-source study (nn_x/codes/sources).

load_hist(split, fidx_sel=None) -> polars DataFrame, one row per history event:
  fidx, hpos (0 = newest), item, cat, et (event type), es (event source), noemb (bool), user_id, ts (feed timestamp), H
For train pass a sorted array of feed indices (sample); val/test default to all feeds.
"""
import os
import numpy as np
import polars as pl

ROOT = "<repo>"
FLAT = f"{ROOT}/data/flat"
OUT = f"{ROOT}/nn_x/codes/sources/out"


def feeds(split):
    f = pl.read_parquet(f"{FLAT}/{split}/feeds.parquet")
    if split == "train":
        pr = np.load(f"{ROOT}/eda/labels_position/out/train_prandom.npz")
        assert (pr["fidx"] == f["fidx"].to_numpy()).all()
        f = f.with_columns(pl.Series("p_random", np.nan_to_num(pr["p_random"], nan=0.0)))
    else:
        f = f.with_columns(pl.lit(1.0).alias("p_random"))
    return f


def _rows(off, ln):
    """Concatenated index ranges [off_i, off_i+ln_i)."""
    ln = ln.astype(np.int64)
    tot = int(ln.sum())
    starts = np.repeat(off.astype(np.int64) - np.concatenate([[0], np.cumsum(ln)[:-1]]), ln)
    return starts + np.arange(tot, dtype=np.int64)


def load_hist(split, fidx_sel=None):
    f = feeds(split)
    if fidx_sel is None:
        fidx_sel = np.arange(f.height)
    fidx_sel = np.sort(np.asarray(fidx_sel))
    fs = f[fidx_sel]
    H = fs["H"].to_numpy()
    rows = _rows(fs["hist_off"].to_numpy(), H)
    g = lambda n: np.load(f"{FLAT}/{split}/hist_{n}.npy", mmap_mode="r")[rows]
    df = pl.DataFrame({
        "fidx": np.repeat(fs["fidx"].to_numpy(), H),
        "hpos": g("hpos").astype(np.int16),
        "item": g("item_id"),
        "cat": g("category_id").astype(np.int8),
        "et": g("event_type").astype(np.int8),
        "es": g("event_source").astype(np.int8),
        "noemb": g("emb_idx") < 0,
        "user_id": np.repeat(fs["user_id"].to_numpy(), H),
        "ts": np.repeat(fs["timestamp"].to_numpy(), H),
        "H": np.repeat(H, H).astype(np.int16),
    })
    return df, fs


def load_cand(split, fidx_sel=None, cols=("item_id", "target", "category_id")):
    f = feeds(split)
    if fidx_sel is None:
        fidx_sel = np.arange(f.height)
    fidx_sel = np.sort(np.asarray(fidx_sel))
    fs = f[fidx_sel]
    M = fs["M"].to_numpy()
    rows = _rows(fs["cand_off"].to_numpy(), M)
    d = {"fidx": np.repeat(fs["fidx"].to_numpy(), M)}
    for c in cols:
        d[c] = np.load(f"{FLAT}/{split}/cand_{c}.npy", mmap_mode="r")[rows]
    return pl.DataFrame(d), fs


def train_sample(n=200_000, seed=0):
    nf = pl.read_parquet(f"{FLAT}/train/feeds.parquet", columns=["fidx"]).height
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(nf, n, replace=False))


def pshow(df, n=60):
    with pl.Config(tbl_rows=n, tbl_cols=40, tbl_width_chars=400, float_precision=4, fmt_str_lengths=200):
        print(df)
