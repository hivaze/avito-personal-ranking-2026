"""Listing-age proxy from first appearance (feed_structure H3: freshness to separate near-duplicate listings).

first_seen(j) = min timestamp of any feed (train/val/test, any user) where item j appears as a candidate or in the history.
An appearance in a history of a feed at time T only proves existence before T, so T is an upper bound on creation time.
For a candidate (feed g, item j): age = ts_g - first_seen(j) >= 0. Only appearances at or before ts_g can set the minimum
(the item is in feed g itself), so the feature is causal. Capped at 14 days so val and test have the same range.
Also: ag_first_hist = 1 if the first appearance is in a history (item existed before any impression we see), and
ag_rank = within-feed rank of age (fresh = high), plus ag_iid_rk = within-feed rank of item_id (if ids are sequential).
"""
import os
os.environ["POLARS_MAX_THREADS"] = "16"
import numpy as np, polars as pl, time
from pathlib import Path

F = Path("<repo>/data/flat")
OUT = Path("<repo>/eda/errors/synthesis/derived")
t0 = time.time()
need = np.unique(np.concatenate([np.load(F / s / "cand_item_id.npy") for s in ("val", "test")]))
first = np.full(len(need), np.iinfo(np.int64).max, np.int64)
first_kind = np.zeros(len(need), np.int8)  # 1 = candidate, 2 = history
for split in ("train", "val", "test"):
    ts = pl.read_parquet(F / split / "feeds.parquet", columns=["timestamp"])["timestamp"].to_numpy().astype(np.int64)
    for kind, pre in ((1, "cand"), (2, "hist")):
        it = np.load(F / split / f"{pre}_item_id.npy", mmap_mode="r")
        fi = np.load(F / split / f"{pre}_fidx.npy", mmap_mode="r")
        CH = 100_000_000
        for s in range(0, len(it), CH):
            a = np.asarray(it[s:s + CH]); ix = np.searchsorted(need, a); ix[ix >= len(need)] = 0
            ok = need[ix] == a
            ix = ix[ok]; t = ts[np.asarray(fi[s:s + CH])[ok]]
            mn = np.full(len(need), np.iinfo(np.int64).max, np.int64)
            np.minimum.at(mn, ix, t) if len(ix) < 5_000_000 else None
            if len(ix) >= 5_000_000:
                d = pl.DataFrame({"i": ix, "t": t}).group_by("i").agg(pl.col("t").min())
                mn[d["i"].to_numpy()] = d["t"].to_numpy()
            upd = mn < first
            first_kind[upd] = kind
            first = np.minimum(first, mn)
        print(f"[{time.time() - t0:.0f}s] {split} {pre}", flush=True)
CAP = 14 * 86400
for split in ("val", "test"):
    f = pl.read_parquet(F / split / "feeds.parquet")
    M = f["M"].to_numpy(); off = f["cand_off"].to_numpy(); ts = f["timestamp"].to_numpy().astype(np.int64)
    it = np.load(F / split / "cand_item_id.npy"); fi = np.load(F / split / "cand_fidx.npy")
    j = np.searchsorted(need, it)
    age = np.minimum(ts[fi] - first[j], CAP).astype(np.float32) / 86400
    assert (age >= 0).all()
    o = OUT / split; o.mkdir(parents=True, exist_ok=True)
    np.save(o / "ag_age.npy", age)
    np.save(o / "ag_logage.npy", np.log1p(age * 24).astype(np.float32))
    np.save(o / "ag_first_hist.npy", (first_kind[j] == 2).astype(np.float32))
    rk = np.zeros(len(it), np.float32); rk2 = np.zeros(len(it), np.float32)
    for g in range(len(M)):
        s, m = off[g], M[g]
        rk[s:s + m] = (np.argsort(np.argsort(-age[s:s + m], kind="stable"), kind="stable") + 0.5) / m
        rk2[s:s + m] = (np.argsort(np.argsort(it[s:s + m], kind="stable"), kind="stable") + 0.5) / m
    np.save(o / "ag_rank.npy", rk); np.save(o / "ag_iid_rk.npy", rk2)
    print(split, "age days q10/50/90", np.percentile(age, [10, 50, 90]).round(2), "first in hist", (first_kind[j] == 2).mean().round(3))
