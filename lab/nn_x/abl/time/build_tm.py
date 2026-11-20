"""TIME arm inputs (ledger nn_x/abl/ledger/ledger.md section 9; CURRENT_STATUS Log 19:08 "TIME BLIND SPOT"). Modelled on
nn_x/abl/pop2/build_xp_ctr.py (exclusive create, hard links, parity statistics).

Three per-candidate float32 columns, broadcast from the FEED's timestamp to every candidate of the feed (cand_fidx), for
train / val / test; Moscow local time = UTC + 3 h (no DST):
  tm_weekend = 1.0 iff the feed's Moscow weekday is Sat / Sun, else 0.0
  tm_hsin    = sin(2 pi * f),  f = local seconds-of-day / 86400 (hour-of-day fraction of the feed)
  tm_hcos    = cos(2 pi * f)
No labels, no history, no item statistics -> leak-free by construction; the timestamp is a feed attribute every split has.
attach_xt (nn_x/abl/data.py) takes the default branch (clip +-10): values in {0, 1} / [-1, 1], nothing clipped, no rank /
z channel (the names do not end in _z / _rk and do not start with seq_cv / it_hist).

Writes NEW files only (exclusive create, never overwrites):
  data/feats/<split>/tm_{weekend,hsin,hcos}.npy
  nn/feat_cache/<split>/tm_{weekend,hsin,hcos}.npy = HARD LINKS of the data/feats files (as xp_ctr_c / xp_imp_c50 are
                                                     exposed to the nn_x/abl trainer; existing entries untouched)
and prints / stores the parity statistics (nn_x/abl/time/out/tm_parity.json): per-split means, feed counts per weekday,
weekend share of feeds and of candidate rows, hour mix by 3-hour block, PSI train-vs-val / train-vs-test (weekend atom and
8 hour blocks), and the weekend share of the 30K in-period holdout (train.py's rng-12345 subset of the p_random > 0.9,
M >= 50 train feeds). PSI train-vs-val of tm_weekend is large BY CONSTRUCTION (val has 0 weekend feeds): it is documented,
not a parity failure, and it is the reason the holdout weekend slice is the pre-declared primary read.

  cd <repo> && OMP_NUM_THREADS=12 .venv/bin/python nn_x/abl/time/build_tm.py [--stats-only]
"""
import datetime as dt
import json
import os
import sys

import numpy as np
import polars as pl

ROOT = "<repo>"
FEATS = f"{ROOT}/data/feats"
CACHE = f"{ROOT}/nn/feat_cache"
FLAT = f"{ROOT}/data/flat"
OUT = f"{ROOT}/nn_x/abl/time/out"
NAMES = ("tm_weekend", "tm_hsin", "tm_hcos")
SPLITS = ("train", "val", "test")
MSK = 3 * 3600
WD = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


def feed_time(split):
    f = pl.read_parquet(f"{FLAT}/{split}/feeds.parquet", columns=["timestamp", "M"])
    ts = f["timestamp"].to_numpy().astype(np.int64)
    loc = ts + MSK
    day = loc // 86400
    wd = (day + 3) % 7  # epoch day 0 (1970-01-01) was a Thursday = 3 in the Mon=0 convention
    # independent check of the weekday arithmetic with datetime on every distinct day of the split
    for d in np.unique(day):
        i = int(np.where(day == d)[0][0])
        ref = (dt.datetime(1970, 1, 1, tzinfo=dt.timezone.utc) + dt.timedelta(seconds=int(ts[i]))
               + dt.timedelta(hours=3)).weekday()
        assert ref == wd[i], (split, d, ref, wd[i])
    frac = (loc % 86400) / 86400.0
    return dict(ts=ts, wd=wd, weekend=(wd >= 5), frac=frac, M=f["M"].to_numpy().astype(np.int64))


def build(split):
    t = feed_time(split)
    fidx = np.load(f"{FLAT}/{split}/cand_fidx.npy")
    assert fidx.min() == 0 and fidx.max() == len(t["ts"]) - 1 and len(fidx) == t["M"].sum()
    cols = {"tm_weekend": t["weekend"].astype(np.float32),
            "tm_hsin": np.sin(2 * np.pi * t["frac"]).astype(np.float32),
            "tm_hcos": np.cos(2 * np.pi * t["frac"]).astype(np.float32)}
    X = {n: c[fidx] for n, c in cols.items()}
    for n, x in X.items():
        assert x.dtype == np.float32 and np.isfinite(x).all() and x.shape == fidx.shape
    assert set(np.unique(X["tm_weekend"]).tolist()) <= {0.0, 1.0}
    assert np.abs(X["tm_hsin"] ** 2 + X["tm_hcos"] ** 2 - 1).max() < 1e-5
    return X, t, fidx


def save_new(path, x):
    with open(path, "xb") as f:  # exclusive create: refuses to overwrite
        np.save(f, x)


def psi(pa, pb):
    pa, pb = np.clip(np.asarray(pa, float), 1e-6, None), np.clip(np.asarray(pb, float), 1e-6, None)
    return float(np.sum((pa - pb) * np.log(pa / pb)))


def main():
    stats_only = "--stats-only" in sys.argv
    os.makedirs(OUT, exist_ok=True)
    X, T, F = {}, {}, {}
    for sp in SPLITS:
        X[sp], T[sp], F[sp] = build(sp)
        if not stats_only:
            for n in NAMES:
                dst, lnk = f"{FEATS}/{sp}/{n}.npy", f"{CACHE}/{sp}/{n}.npy"
                assert not os.path.exists(dst) and not os.path.exists(lnk), f"{dst} / {lnk} exists: refusing to overwrite"
            for n in NAMES:
                dst, lnk = f"{FEATS}/{sp}/{n}.npy", f"{CACHE}/{sp}/{n}.npy"
                save_new(dst, X[sp][n])
                os.link(dst, lnk)
                st_d, st_l = os.stat(dst), os.stat(lnk)
                assert st_d.st_ino == st_l.st_ino and st_d.st_nlink == 2
                back = np.load(lnk)
                assert back.dtype == np.float32 and np.array_equal(back, X[sp][n])
                print(f"{sp}: wrote {dst} ({len(back):,} rows), hard link {lnk} (inode {st_d.st_ino}, nlink 2)", flush=True)
    res = {"names": list(NAMES), "def": "tm_weekend = 1[Moscow weekday in Sat/Sun]; tm_hsin/hcos = sin/cos(2 pi * local "
                                         "seconds-of-day / 86400); feed value broadcast to every candidate (cand_fidx); float32"}
    blocks = np.arange(0, 25, 3)
    for sp in SPLITS:
        t, x = T[sp], X[sp]
        cnt = np.bincount(t["wd"], minlength=7)
        hb = np.histogram(t["frac"] * 24, blocks)[0]
        res[sp] = dict(feeds=int(len(t["ts"])), rows=int(len(F[sp])),
                       feeds_per_weekday={WD[i]: int(cnt[i]) for i in range(7)},
                       weekend_share_feeds=float(t["weekend"].mean()), weekend_share_rows=float(x["tm_weekend"].mean()),
                       mean={n: float(x[n].mean()) for n in NAMES}, sd={n: float(x[n].std()) for n in NAMES},
                       hour_block_share_feeds=(hb / hb.sum()).round(4).tolist(),
                       hour_block_share_rows=(np.histogram(np.arctan2(x["tm_hsin"], x["tm_hcos"]) % (2 * np.pi) / (2 * np.pi) * 24,
                                                           blocks)[0] / len(x["tm_hsin"])).round(4).tolist(),
                       local_hour_mean=float((t["frac"] * 24).mean()),
                       days=sorted({f"{dt.datetime(1970, 1, 1, tzinfo=dt.timezone.utc) + dt.timedelta(days=int(d)):%a %b %d}"
                                    for d in np.unique((t["ts"] + MSK) // 86400)}))  # Moscow local days
    res["hour_block_labels"] = [f"{a:02d}-{b:02d}" for a, b in zip(blocks[:-1], blocks[1:])]
    for other in ("val", "test"):
        pa = [1 - res["train"]["weekend_share_rows"], res["train"]["weekend_share_rows"]]
        pb = [1 - res[other]["weekend_share_rows"], res[other]["weekend_share_rows"]]
        res[f"psi_train_{other}"] = dict(weekend_rows=psi(pa, pb),
                                        hour_blocks_rows=psi(res["train"]["hour_block_share_rows"], res[other]["hour_block_share_rows"]),
                                        hour_blocks_feeds=psi(res["train"]["hour_block_share_feeds"], res[other]["hour_block_share_feeds"]))
    res["psi_val_test"] = dict(hour_blocks_feeds=psi(res["val"]["hour_block_share_feeds"], res["test"]["hour_block_share_feeds"]))
    # the in-period 30K holdout (train.py: rng 12345 over p_random > 0.9 & M >= 50 train feeds)
    pr = np.load(f"{ROOT}/eda/labels_position/out/train_prandom.npz")
    prand = np.nan_to_num(pr["p_random"].astype(np.float32), nan=0.0)
    Mtr = T["train"]["M"]
    prand[Mtr < 50] = 0.0
    cand = np.where((prand > 0.9) & (Mtr >= 50))[0]
    hold = np.sort(np.random.default_rng(12345).choice(cand, 30000, replace=False))
    meta = f"{ROOT}/nn_x/abl/preds/holdout/holdout_meta.npy"
    if os.path.exists(meta):
        Mh = np.load(meta)[0]
        assert np.array_equal(Mtr[hold], Mh), "holdout selection does not match nn_x/abl/preds/holdout/holdout_meta.npy"
        res["holdout_meta_check"] = "OK (M of the rng-12345 subset == holdout_meta.npy)"
    hw = T["train"]["weekend"][hold]
    hcnt = np.bincount(T["train"]["wd"][hold], minlength=7)
    res["holdout"] = dict(feeds=int(len(hold)), weekend_feeds=int(hw.sum()), weekday_feeds=int((~hw).sum()),
                          weekend_share=float(hw.mean()), feeds_per_weekday={WD[i]: int(hcnt[i]) for i in range(7)},
                          hour_block_share_feeds=(np.histogram(T["train"]["frac"][hold] * 24, blocks)[0] / len(hold)).round(4).tolist())
    print(json.dumps(res, indent=1))
    json.dump(res, open(f"{OUT}/tm_parity.json", "w"), indent=1)
    assert res["val"]["weekend_share_feeds"] == 0.0, "expected: val has no weekend feeds (Tue / Thu only)"
    assert res["psi_train_val"]["hour_blocks_feeds"] < 0.1 and res["psi_train_test"]["hour_blocks_feeds"] < 0.1, "hour mix differs"


if __name__ == "__main__":
    main()
