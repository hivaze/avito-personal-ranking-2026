"""Full independent build of the counts for ALL val + test rows from the raw parquet (polars self-join on item), compared
with the builder's arrays on every row; then the label-free position / coverage tables of CHECK 4.
Writes out/my_counts.npy [N, 7] int32 in the flat row order (val then test): a, a24, b, d, e, a_from_val, a_from_test.
"""
import json
import time

import numpy as np
import polars as pl

from rlib import BOUND, MSK, OUT, ROOT, load_raw

t0 = time.time()
raw = load_raw()
parts = []
for k, sp in enumerate(("val", "test")):
    d = raw[sp].with_columns(pl.lit(k, dtype=pl.Int8).alias("sp"))
    parts.append(d)
feeds = pl.concat(parts).with_row_index("g")
feeds = feeds.with_columns(pl.when(pl.col("user_id") == 0).then(-1 - pl.col("g").cast(pl.Int64)).otherwise(pl.col("user_id").cast(pl.Int64)).alias("uk"),
                           ((pl.col("timestamp").cast(pl.Int64) + MSK) // 86400).alias("day"), pl.col("timestamp").cast(pl.Int64).alias("ts"))
rows = feeds.select("g", "feed_id", "uk", "ts", "day", "sp", "item_id").explode("item_id")
print(f"rows {len(rows)} | {time.time() - t0:.0f}s", flush=True)
src = rows.select(pl.col("item_id"), pl.col("g").alias("g2"), pl.col("uk").alias("uk2"), pl.col("ts").alias("ts2"), pl.col("day").alias("day2"), pl.col("sp").alias("sp2"))
j = rows.join(src, on="item_id", how="inner").filter((pl.col("g") != pl.col("g2")) & (pl.col("uk") != pl.col("uk2")))
print(f"pairs after own-feed / same-user exclusion {len(j)} | {time.time() - t0:.0f}s", flush=True)
agg = j.group_by("g", "item_id").agg(
    (pl.col("ts2") < pl.col("ts")).sum().alias("a"),
    ((pl.col("ts2") < pl.col("ts")) & (pl.col("ts2") >= pl.col("ts") - 86400)).sum().alias("a24"),
    (pl.col("day2") == pl.col("day")).sum().alias("b"),
    pl.len().alias("d"),
    (pl.col("ts2") > pl.col("ts")).sum().alias("e"),
    ((pl.col("ts2") < pl.col("ts")) & (pl.col("sp2") == 0)).sum().alias("a_val"),
    ((pl.col("ts2") < pl.col("ts")) & (pl.col("sp2") == 1)).sum().alias("a_test"))
# flat row order (the builder's and the prediction files' order)
fl = []
for sp in ("val", "test"):
    f = pl.read_parquet(f"{ROOT}/data/flat/{sp}/feeds.parquet")
    fl.append(pl.DataFrame({"feed_id": np.repeat(f["feed_id"].to_numpy(), f["M"].to_numpy()), "item_id": np.load(f"{ROOT}/data/flat/{sp}/cand_item_id.npy")}))
flat = pl.concat(fl).with_row_index("row")
flat = flat.join(feeds.select("feed_id", "g", "ts", "day", "sp", "user_id"), on="feed_id", how="left")
flat = flat.join(agg, on=["g", "item_id"], how="left").fill_null(0).sort("row")
assert len(flat) == len(rows)
C = flat.select("a", "a24", "b", "d", "e", "a_val", "a_test").to_numpy().astype(np.int32)
np.save(f"{OUT}/my_counts.npy", C)
meta = dict(ts=flat["ts"].to_numpy(), day=flat["day"].to_numpy(), sp=flat["sp"].to_numpy(), g=flat["g"].to_numpy().astype(np.int64), uid0=(flat["user_id"].to_numpy() == 0))
np.savez(f"{OUT}/my_rowmeta.npz", **meta)
B = np.load(f"{ROOT}/nn_x/w12/fresh/pregate/out/cnt_main.npy")
res = {"n_rows": int(len(C))}
for k, (w, bc) in enumerate((("a", 0), ("a24", 1), ("b", 2), ("d", 4), ("e", 5))):
    eq = float((C[:, k] == B[:, bc]).mean()); res["equal_all_rows_" + w] = eq
    print(f"ALL ROWS window {w:3s}: mine == builder's on {100 * eq:.5f}% of {len(C)} rows (max abs diff {int(np.abs(C[:, k] - B[:, bc]).max())})")
# ---------------- CHECK 4 label-free part: coverage by position in the period
ts = meta["ts"]; day = meta["day"]; sp = meta["sp"]
cold = np.concatenate([np.load(f"{ROOT}/data/feats/{s}/tte_it_imp.npy") == 0 for s in ("val", "test")])
tod = (ts + MSK) % 86400
print("\n=== coverage (count >= 1, percent) by Moscow day x third of the calendar day (0-8h, 8-16h, 16-24h MSK): rows | a all / cold | a24 all / cold | a from val feeds only | a from test feeds only")
res["by_day_third"] = {}
for d in np.unique(day):
    for th in range(3):
        m = (day == d) & (tod // 28800 == th)
        if m.sum() < 3000:
            continue
        r = [int(m.sum()), float((C[m, 0] >= 1).mean()), float((C[m & cold, 0] >= 1).mean()), float((C[m, 1] >= 1).mean()), float((C[m & cold, 1] >= 1).mean()),
             float((C[m, 5] >= 1).mean()), float((C[m, 6] >= 1).mean())]
        res["by_day_third"][f"{int(d)}|{th}"] = r
        print(f"day {d} third {th}: rows {r[0]:8d} | a {100 * r[1]:5.1f} / {100 * r[2]:5.1f} | a24 {100 * r[3]:5.1f} / {100 * r[4]:5.1f} | from val {100 * r[5]:5.1f} | from test {100 * r[6]:5.1f}")
first_test = ts[sp == 1].min()
print(f"\n=== first test day: hours since the first test feed (ts {first_test}); coverage of a / a24 / a from val only / a from test only (percent), cold rows in brackets")
res["first_test_hours"] = {}
for h0, h1 in ((0, 1), (1, 3), (3, 6), (6, 12), (12, 24), (24, 48), (48, 72), (72, 200)):
    m = (sp == 1) & (ts >= first_test + 3600 * h0) & (ts < first_test + 3600 * h1)
    if m.sum() == 0:
        continue
    mc = m & cold
    r = [int(m.sum()), float((C[m, 0] >= 1).mean()), float((C[m, 1] >= 1).mean()), float((C[m, 5] >= 1).mean()), float((C[m, 6] >= 1).mean()),
         float((C[mc, 0] >= 1).mean()), float((C[mc, 1] >= 1).mean())]
    res["first_test_hours"][f"{h0}-{h1}h"] = r
    print(f"  {h0:3d}-{h1:3d} h: rows {r[0]:8d} | a {100 * r[1]:5.1f} ({100 * r[5]:5.1f}) | a24 {100 * r[2]:5.1f} ({100 * r[6]:5.1f}) | a from val {100 * r[3]:5.1f} | a from test {100 * r[4]:5.1f}")
pub = (sp == 1) & (ts < BOUND); priv = (sp == 1) & (ts >= BOUND); val = sp == 0
for nm, m in (("val", val), ("public-like", pub), ("private-like", priv)):
    print(f"{nm:13s}: a>=1 {100 * (C[m, 0] >= 1).mean():5.1f}% (cold {100 * (C[m & cold, 0] >= 1).mean():5.1f}%) a24>=1 {100 * (C[m, 1] >= 1).mean():5.1f}% | "
          f"of the a-covered rows, share covered ONLY through val feeds {100 * ((C[m, 0] >= 1) & (C[m, 6] == 0)).sum() / max((C[m, 0] >= 1).sum(), 1):5.1f}%")
    res["cov_" + nm] = [float((C[m, 0] >= 1).mean()), float((C[m & cold, 0] >= 1).mean()), float((C[m, 1] >= 1).mean())]
json.dump(res, open(f"{OUT}/r2_build_full.json", "w"), indent=1)
print(f"DONE {time.time() - t0:.0f}s")
