"""CHECK 1 leak part (targeted): every val AND test feed of a non-zero user who has >= 2 feeds in val + test.
For their candidate rows: count with and without the same user's other feeds (raw tables, my code) vs the builder's array."""
import glob, json, time
from collections import defaultdict
import numpy as np, polars as pl
ROOT = "<repo>"; BLD = f"{ROOT}/nn_x/w12/fresh/pregate/out"; OUT = f"{ROOT}/nn_x/w12/fresh/review/out"
COLS = ["feed_id", "timestamp", "user_id", "item_id"]
raw = {sp: pl.concat([pl.read_parquet(f, columns=COLS) for f in sorted(glob.glob(f"{ROOT}/data/competition/{sp}/impressions/part-*.parquet"))]) for sp in ("val", "test")}
allf = pl.concat([raw["val"].with_columns(pl.lit(0).alias("sp")), raw["test"].with_columns(pl.lit(1).alias("sp"))])
nfeeds_u = allf.filter(pl.col("user_id") != 0).group_by("user_id").agg(pl.len().alias("n"), pl.col("sp").n_unique().alias("nsp"))
multi = nfeeds_u.filter(pl.col("n") >= 2)
print(f"non-zero users with >= 2 feeds in val + test: {len(multi)} (feeds {multi['n'].sum()}); in both splits {int((multi['nsp'] == 2).sum())}; max feeds per user {multi['n'].max()}")
idx = defaultdict(list)
for fid, ts, uid, items in zip(allf["feed_id"].to_list(), allf["timestamp"].to_list(), allf["user_id"].to_list(), allf["item_id"].to_list()):
    for it in items:
        idx[it].append((ts, uid, fid))
q = allf.filter(pl.col("user_id").is_in(multi["user_id"].to_list()))
rows = []
for fid, ts, uid, sp, items in zip(q["feed_id"].to_list(), q["timestamp"].to_list(), q["user_id"].to_list(), q["sp"].to_list(), q["item_id"].to_list()):
    for it in items:
        a = d = a_inc = d_inc = 0
        for ts2, uid2, fid2 in idx[it]:
            if fid2 == fid:
                continue
            d_inc += 1; a_inc += ts2 < ts
            if uid2 == uid:
                continue
            d += 1; a += ts2 < ts
        rows.append((fid, it, sp, a, d, a_inc, d_inc))
mine = pl.DataFrame(rows, schema=["feed_id", "item_id", "sp", "a", "d", "a_inc", "d_inc"], orient="row")
fl = []
for sp in ("val", "test"):
    f = pl.read_parquet(f"{ROOT}/data/flat/{sp}/feeds.parquet")
    fl.append(pl.DataFrame({"feed_id": np.repeat(f["feed_id"].to_numpy(), f["M"].to_numpy()), "item_id": np.load(f"{ROOT}/data/flat/{sp}/cand_item_id.npy")}))
cm = np.load(f"{BLD}/cnt_main.npy")
bld = pl.concat(fl).with_columns(pl.Series("B_a", cm[:, 0]), pl.Series("B_d", cm[:, 4]))
j = mine.join(bld, on=["feed_id", "item_id"], how="left")
assert j["B_a"].null_count() == 0
res = dict(users=int(len(multi)), feeds=int(len(q)), rows=int(len(j)),
           eq_a=float((j["a"] == j["B_a"]).mean()), eq_d=float((j["d"] == j["B_d"]).mean()),
           rows_same_user_changes_a=int((j["a"] != j["a_inc"]).sum()), rows_same_user_changes_d=int((j["d"] != j["d_inc"]).sum()))
x = j.filter(pl.col("d") != pl.col("d_inc"))
res["on_changed_rows_builder_eq_excluding_d"] = float((x["B_d"] == x["d"]).mean()); res["on_changed_rows_builder_eq_including_d"] = float((x["B_d"] == x["d_inc"]).mean())
x = j.filter(pl.col("a") != pl.col("a_inc"))
res["on_changed_rows_builder_eq_excluding_a"] = float((x["B_a"] == x["a"]).mean()) if len(x) else None
res["on_changed_rows_builder_eq_including_a"] = float((x["B_a"] == x["a_inc"]).mean()) if len(x) else None
print(json.dumps(res, indent=1)); json.dump(res, open(f"{OUT}/r1b_sameuser.json", "w"), indent=1)
