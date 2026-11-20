"""CHECK 1: rebuild the as-of count of 500 random val feeds from the RAW parquet tables with independent code.

Reads only feed_id, timestamp, user_id, item_id (never the target). No import from the builder's directory.
  cd <repo> && CUDA_VISIBLE_DEVICES="" POLARS_MAX_THREADS=12 .venv/bin/python nn_x/w12/fresh/review/r1_rebuild.py
"""
import glob
import json
import time
from collections import defaultdict

import numpy as np
import polars as pl

ROOT = "<repo>"
OUT = f"{ROOT}/nn_x/w12/fresh/review/out"
BLD = f"{ROOT}/nn_x/w12/fresh/pregate/out"
t0 = time.time()
COLS = ["feed_id", "timestamp", "user_id", "item_id"]
raw = {}
for sp in ("val", "test"):
    fs = sorted(glob.glob(f"{ROOT}/data/competition/{sp}/impressions/part-*.parquet"))
    raw[sp] = pl.concat([pl.read_parquet(f, columns=COLS) for f in fs])
    print(sp, raw[sp].shape, raw[sp].schema, flush=True)
nv, nt = len(raw["val"]), len(raw["test"])
# ---- source index: item -> list of (ts, user_key, day, feed_id) over ALL val + test feeds
idx = defaultdict(list)
n_rows = 0; n_item0 = 0; n_dup = 0
feed_user = {}
for sp in ("val", "test"):
    d = raw[sp]
    for fid, ts, uid, items in zip(d["feed_id"].to_list(), d["timestamp"].to_list(), d["user_id"].to_list(), d["item_id"].to_list()):
        ukey = ("anon", fid) if uid == 0 else ("u", uid)      # uid 0: every feed is its own user
        day = (ts + 10800) // 86400
        n_rows += len(items); n_dup += len(items) - len(set(items))
        for it in items:
            if it == 0:
                n_item0 += 1
            idx[it].append((ts, ukey, day, fid, uid))
print(f"index built: {nv + nt} feeds, {n_rows} rows, {len(idx)} items; candidate rows with item_id 0: {n_item0}; "
      f"within-feed duplicate (feed, item): {n_dup} | {time.time() - t0:.0f}s", flush=True)
# ---- label-free facts
uv = set(raw["val"]["user_id"].to_list()) - {0}; ut = set(raw["test"]["user_id"].to_list()) - {0}
print(f"non-zero users: val {len(uv)}, test {len(ut)}, in both {len(uv & ut)}; uid-0 feeds val {(raw['val']['user_id'] == 0).sum()} "
      f"test {(raw['test']['user_id'] == 0).sum()}")
tsv = raw["val"]["timestamp"].to_numpy(); tst = raw["test"]["timestamp"].to_numpy()
print(f"timestamp grid: gcd of val ts differences {np.gcd.reduce(np.diff(np.sort(tsv)))}, test {np.gcd.reduce(np.diff(np.sort(tst)))}; "
      f"val days {sorted(set(((tsv + 10800) // 86400).tolist()))} test days {sorted(set(((tst + 10800) // 86400).tolist()))}")
# ---- 500 random val feeds
rng = np.random.default_rng(20261115)
pick = np.sort(rng.choice(nv, 500, replace=False))
dv = raw["val"][pick]
rows = []
for fid, ts, uid, items in zip(dv["feed_id"].to_list(), dv["timestamp"].to_list(), dv["user_id"].to_list(), dv["item_id"].to_list()):
    ukey = ("anon", fid) if uid == 0 else ("u", uid)
    day = (ts + 10800) // 86400
    for it in items:
        a = a24 = b = dd = e = 0          # the declared rule: other users only
        a_same = 0                        # variant: same-user feeds NOT excluded (own feed still excluded)
        a_nz = 0                          # variant: uid-0 feeds never sources
        a_own = 0                         # variant: own feed included (ts < t can never hold for the own feed -> uses ts <= t)
        for (ts2, uk2, day2, fid2, uid2) in idx[it]:
            if fid2 == fid:
                a_own += 1
                continue
            if ts2 < ts:
                a_same += 1
            if uk2 == ukey:
                continue
            dd += 1
            if ts2 < ts:
                a += 1
                if uid2 != 0:
                    a_nz += 1
                if ts2 >= ts - 86400:
                    a24 += 1
            elif ts2 > ts:
                e += 1
            if day2 == day:
                b += 1
        rows.append((fid, it, uid, ts, a, a24, b, dd, e, a_same, a_nz, a_own))
mine = pl.DataFrame(rows, schema=["feed_id", "item_id", "user_id", "ts", "a", "a24", "b", "d", "e", "a_same", "a_nz", "a_own"], orient="row")
print(f"my counts for {len(pick)} feeds: {len(mine)} rows | {time.time() - t0:.0f}s", flush=True)
# ---- the builder's arrays, keyed by (feed_id, item_id) through the flat tables (the builder's row order)
f = pl.read_parquet(f"{ROOT}/data/flat/val/feeds.parquet")
print("flat val feeds columns:", f.columns)
M = f["M"].to_numpy(); fid_flat = np.repeat(f["feed_id"].to_numpy(), M)
it_flat = np.load(f"{ROOT}/data/flat/val/cand_item_id.npy")
N = len(it_flat)
cm = np.load(f"{BLD}/cnt_main.npy", mmap_mode="r")[:N]; cz = np.load(f"{BLD}/cnt_nzsrc.npy", mmap_mode="r")[:N]
bld = pl.DataFrame({"feed_id": fid_flat, "item_id": it_flat, "B_a": np.asarray(cm[:, 0]), "B_a24": np.asarray(cm[:, 1]), "B_b": np.asarray(cm[:, 2]),
                    "B_c": np.asarray(cm[:, 3]), "B_d": np.asarray(cm[:, 4]), "B_e": np.asarray(cm[:, 5]), "B_a_nz": np.asarray(cz[:, 0])})
j = mine.join(bld, on=["feed_id", "item_id"], how="left")
assert len(j) == len(mine) and j["B_a"].null_count() == 0, "row mapping failed"
res = {"n_feeds": int(len(pick)), "n_rows": int(len(mine)), "item0_rows": int(n_item0), "dup_rows": int(n_dup)}
for w in ("a", "a24", "b", "d", "e"):
    eq = (j[w] == j["B_" + w]).mean()
    res["equal_" + w] = float(eq)
    print(f"window {w:3s}: my count == builder's on {100 * eq:.4f}% of {len(j)} rows; my >=1 share {100 * (j[w] >= 1).mean():.2f}%  mean {j[w].mean():.4f}")
eqz = (j["a_nz"] == j["B_a_nz"]).mean(); res["equal_a_nz"] = float(eqz)
print(f"NZSRC a : my count == builder's on {100 * eqz:.4f}% of rows")
# leak part
dif_same = j.filter(pl.col("a_same") != pl.col("a"))
res["rows_where_same_user_feeds_would_add"] = int(len(dif_same))
res["builder_equals_excluding_version_there"] = float((dif_same["B_a"] == dif_same["a"]).mean()) if len(dif_same) else None
res["builder_equals_including_version_there"] = float((dif_same["B_a"] == dif_same["a_same"]).mean()) if len(dif_same) else None
print(f"same-user exclusion: rows where the user's own EARLIER feeds would add to the count: {len(dif_same)} "
      f"(feeds {dif_same['feed_id'].n_unique()}); builder == excluding version on {res['builder_equals_excluding_version_there']}, "
      f"== including version on {res['builder_equals_including_version_there']}")
res["own_feed_occurrences_min_max"] = [int(j["a_own"].min()), int(j["a_own"].max())]
print(f"own feed: occurrences of the item in its own feed min {j['a_own'].min()} max {j['a_own'].max()} (1 = no duplicate; never counted)")
# the whole-period count never includes the user's own later feeds either: check on d with a same-user-any-time variant
print(f"uid-0 feeds among the 500: {(dv['user_id'] == 0).sum()}; rows of uid-0 feeds {int((j['user_id'] == 0).sum())}")
json.dump(res, open(f"{OUT}/r1_rebuild.json", "w"), indent=1)
ok = all(res["equal_" + w] == 1.0 for w in ("a", "a24", "b", "d", "e")) and res["equal_a_nz"] == 1.0
print("CHECK 1 " + ("PASS" if ok else "FAIL") + f" | {time.time() - t0:.0f}s")
