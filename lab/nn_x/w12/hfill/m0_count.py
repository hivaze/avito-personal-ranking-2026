"""hfill Stage A, step 0 (CPU): how many val / test history events have emb_idx < 0, and how many of them an EARLIER split's
item table recovers (val: train table; test: train + val tables). Also the train rate (reference: what members saw in training).

Leak note: the train table belongs to 09-06..09-19, the val table to 09-22..09-24, both strictly before test (09-26..29); the
val split only uses the train table. Item -> row maps come from the flat (item_id, emb_idx) pairs of cand + hist of the
source split (1:1 asserted, as in lab/nn_x/w12/gap1117/lifelong/s3_build_arrays.py item_to_row).
Writes out/m0_counts.json and out/fill_map_<split>.npz (item ids, source split code, source row) for the replay.
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import polars as pl

FLAT = Path("<repo>/data/flat")
OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


def item_to_row(split):
    it = np.r_[np.load(FLAT / split / "hist_item_id.npy", mmap_mode="r"), np.load(FLAT / split / "cand_item_id.npy", mmap_mode="r")].astype(np.int64)
    em = np.r_[np.load(FLAT / split / "hist_emb_idx.npy", mmap_mode="r"), np.load(FLAT / split / "cand_emb_idx.npy", mmap_mode="r")].astype(np.int64)
    m = em >= 0
    p = np.unique(it[m] * (1 << 32) + em[m])
    items, rows = p >> 32, p & ((1 << 32) - 1)
    assert len(np.unique(items)) == len(items), f"{split}: an item has several table rows"
    no_emb = np.unique(it[~m])
    return items, rows, no_emb


def lookup(items_sorted, rows, q):
    pos = np.searchsorted(items_sorted, q)
    pos = np.minimum(pos, len(items_sorted) - 1)
    ok = items_sorted[pos] == q
    return ok, np.where(ok, rows[pos], -1)


res = {}
maps = {s: item_to_row(s) for s in ("train", "val", "test")}
for s in maps:
    log(s, "table items", len(maps[s][0]), "items seen without a vector", len(maps[s][2]),
        "of which also with a vector in the same split", int(np.isin(maps[s][2], maps[s][0]).sum()))

for split, srcs in (("val", ("train",)), ("test", ("val", "train"))):  # test: val first (newer), then train
    f = pl.read_parquet(FLAT / split / "feeds.parquet")
    H = f["H"].to_numpy().astype(np.int64)
    hit = np.load(FLAT / split / "hist_item_id.npy").astype(np.int64)
    hem = np.load(FLAT / split / "hist_emb_idx.npy")
    hfi = np.load(FLAT / split / "hist_fidx.npy")
    cem = np.load(FLAT / split / "cand_emb_idx.npy")
    n_ev = len(hem)
    miss = hem < 0
    log(split, f"hist events {n_ev}, masked {miss.sum()} ({miss.mean():.4f}); item_id 0 among masked {(hit[miss] == 0).sum()}; "
        f"cand masked {(cem < 0).mean():.4f}")
    src_code = np.full(n_ev, -1, np.int8)
    src_row = np.full(n_ev, -1, np.int64)
    for k, s in enumerate(srcs):
        todo = miss & (src_code < 0) & (hit > 0)
        ok, r = lookup(maps[s][0], maps[s][1], hit[todo])
        idx = np.where(todo)[0][ok]
        src_code[idx] = {"train": 0, "val": 1}[s]
        src_row[idx] = r[ok]
        log(f"  {split}: recovered from {s}: {ok.sum()}")
    rec = src_code >= 0
    nf = len(H)
    feed_masked = np.bincount(hfi[miss], minlength=nf) > 0
    feed_rec = np.bincount(hfi[rec], minlength=nf) > 0
    # recent window: events in the first 32 / 64 positions (history newest first)
    hpos = np.load(FLAT / split / "hist_hpos.npy")
    r = dict(n_events=int(n_ev), masked=int(miss.sum()), masked_share=float(miss.mean()),
             recovered=int(rec.sum()), recovered_share_of_masked=float(rec.sum() / max(miss.sum(), 1)),
             recovered_share_of_events=float(rec.mean()),
             recovered_by_src={s: int((src_code == {"train": 0, "val": 1}[s]).sum()) for s in srcs},
             n_feeds=int(nf), feeds_with_masked=int(feed_masked.sum()), feeds_with_recovered=int(feed_rec.sum()),
             feeds_with_recovered_share=float(feed_rec.mean()),
             masked_in_top32=int((miss & (hpos < 32)).sum()), recovered_in_top32=int((rec & (hpos < 32)).sum()),
             masked_in_top64=int((miss & (hpos < 64)).sum()), recovered_in_top64=int((rec & (hpos < 64)).sum()),
             distinct_recovered_items=int(len(np.unique(hit[rec]))),
             cand_masked_share=float((cem < 0).mean()))
    per_feed_rec = np.bincount(hfi[rec], minlength=nf)
    r["recovered_per_feed_q50_q90_q99_of_feeds_with_any"] = [float(np.quantile(per_feed_rec[feed_rec], q)) for q in (0.5, 0.9, 0.99)] if feed_rec.any() else []
    res[split] = r
    log(split, json.dumps(r))
    np.savez(OUT / f"fill_map_{split}.npz", ev=np.where(rec)[0].astype(np.int64), item=hit[rec].astype(np.int32),
             src=src_code[rec], row=src_row[rec], feeds_rec=np.where(feed_rec)[0].astype(np.int64))
    del hit, hem, hfi, cem, hpos

# train reference rate (what members saw in training)
hem = np.load(FLAT / "train" / "hist_emb_idx.npy", mmap_mode="r")
n = len(hem); m = 0
for i in range(0, n, 100_000_000):
    m += int((np.asarray(hem[i:i + 100_000_000]) < 0).sum())
res["train"] = dict(n_events=int(n), masked=m, masked_share=m / n)
log("train", res["train"])
(OUT / "m0_counts.json").write_text(json.dumps(res, indent=1))
log("done")
