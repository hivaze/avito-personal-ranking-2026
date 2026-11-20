"""Shared helpers for the category-code study (nn_x/codes/categories). CPU only, read-only on project data."""
import os, sys, json, time
from pathlib import Path
import numpy as np
import polars as pl

ROOT = Path("<repo>")
FLAT = ROOT / "data/flat"
FEATS = ROOT / "data/feats"
COMP = ROOT / "data/competition"
OUT = ROOT / "nn_x/codes/categories/out"
OUT.mkdir(parents=True, exist_ok=True)
NC = 54  # codes 0..53
MSK = 3 * 3600
DAY0 = (1788642048 + MSK) // 86400  # Sep 6 2026 (MSK) = day 0, a Sunday
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def meta(split):
    f = pl.read_parquet(FLAT / split / "feeds.parquet")
    d = dict(ts=f["timestamp"].to_numpy().astype(np.int64), user=f["user_id"].to_numpy().astype(np.int64),
             M=f["M"].to_numpy().astype(np.int64), H=f["H"].to_numpy().astype(np.int64),
             off=f["cand_off"].to_numpy().astype(np.int64), hoff=f["hist_off"].to_numpy().astype(np.int64))
    t = d["ts"] + MSK
    d["hour"] = (t // 3600) % 24
    d["day"] = t // 86400 - DAY0
    d["dow"] = (d["day"] + 6) % 7  # Monday = 0 ... Sunday = 6
    d["n"] = len(d["M"])
    return d


def cand(split, name):
    return np.load(FLAT / split / f"cand_{name}.npy", mmap_mode="r")


def hist(split, name):
    return np.load(FLAT / split / f"hist_{name}.npy", mmap_mode="r")


def p_random():
    z = np.load(ROOT / "eda/labels_position/out/train_prandom.npz")
    return np.nan_to_num(z["p_random"], nan=0.0)


def emb_table(split):
    m = json.load(open(COMP / split / "item_embeddings/data.meta"))
    return np.memmap(COMP / split / "item_embeddings/data.memmap", dtype=np.int8, mode="r", shape=tuple(m["shape"]))


def rows_of_feeds(off, M, feeds):
    """candidate row indices of the given feeds (concatenated)."""
    st = off[feeds]; ln = M[feeds]
    tot = int(ln.sum())
    rep = np.repeat(st - np.concatenate([[0], np.cumsum(ln)[:-1]]), ln)
    return rep + np.arange(tot)


def age_hours(f4):
    knots = np.load(FEATS / "life_f4_age_knots.npy")
    x = np.asarray(f4, dtype=np.float64)
    u = x * 64; k = np.clip(np.floor(u).astype(int), 1, 62); w = u - k
    return np.where(x < 63 / 64 - 1e-6, knots[k] * (1 - w) + knots[k + 1] * w, np.nan)


def fmt_table(header, rows, fmt=None):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)
