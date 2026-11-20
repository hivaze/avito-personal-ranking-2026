"""Update cadence of the organiser features (real-time vs batch):
(a) item side: consecutive appearances of the same item (any users) in a 400K-feed train sample: share of EXACTLY equal f5 / f2 / f1
    by time gap and by same-vs-different MSK day.
(b) user side: consecutive feeds of the same user (all train feeds of 150K sampled users? -> use the 400K sample's users with >= 2 feeds),
    same category in both feeds: share of exactly equal f3, by gap and same/different MSK day; and whether f3 moves with new
    same-category history events between the two feeds."""
import sys; sys.path.insert(0, "<repo>/nn_x/codes/features")
from fcommon import *

fidx = train_sample("uniform", 400_000, 1)
d = load("train", fidx)
ts = d["timestamp"].astype(np.int64); item = d["item"].astype(np.int64)
day = (ts + 3 * 3600) // 86400
o = np.lexsort((ts, item)); it = item[o]
same = it[1:] == it[:-1]
gap = (ts[o][1:] - ts[o][:-1])[same]; sd = (day[o][1:] == day[o][:-1])[same]
print(f"(a) consecutive same-item appearances: {same.sum()}")
bins = [(0, 1), (1, 600), (600, 3600), (3600, 6 * 3600), (6 * 3600, 86400), (86400, 3 * 86400), (3 * 86400, 30 * 86400)]
for k in ["f5", "f4", "f2", "f1"]:
    v = d[k][o]; eq = (v[1:] == v[:-1])[same]
    row = []
    for lo, hi in bins:
        m = (gap >= lo) & (gap < hi)
        row.append(f"[{lo//60}m,{hi//60}m): {eq[m].mean():.3f}")
    print(f"   {k} exact-equal by gap: " + " | ".join(row))
    for lo, hi in [(3600, 6 * 3600), (6 * 3600, 86400)]:
        m = (gap >= lo) & (gap < hi)
        print(f"      gap {lo//3600}-{hi//3600} h: same MSK day {eq[m & sd].mean():.3f} (n={np.sum(m&sd)}), different day {eq[m & ~sd].mean():.3f} (n={np.sum(m&~sd)})")

# (b) same user consecutive feeds: need complete user timelines -> take all train feeds of 30K random non-zero users
f = feeds("train")
u = f["user_id"].to_numpy()
rng = np.random.default_rng(7)
uu = rng.choice(np.unique(u[u > 0]), 30_000, replace=False)
fi = np.flatnonzero(np.isin(u, uu))
d2 = load("train", fi)
print(f"\n(b) users 30000, their train feeds {len(fi)}")
key = d2["feed"].astype(np.int64) * 64 + d2["cat"]
_, first = np.unique(key, return_index=True)
fc = pl.DataFrame({"feed": d2["feed"][first], "cat": d2["cat"][first].astype(np.int32), "f3": d2["f3"][first], "user": d2["user_id"][first], "ts": d2["timestamp"][first].astype(np.int64)})
# history same-category counts per (feed, cat)
fH = f["H"].to_numpy()[fi]; hoff = f["hist_off"].to_numpy()[fi]
hrows = np.repeat(hoff - np.cumsum(np.r_[0, fH[:-1]]), fH) + np.arange(fH.sum())
hc = np.asarray(np.load(FLAT / "train/hist_category_id.npy", mmap_mode="r")[hrows]).astype(np.int64)
hf = np.repeat(np.arange(len(fi)), fH)
hcnt = pl.DataFrame({"feed": hf, "cat": hc.astype(np.int32)}).group_by(["feed", "cat"]).agg(pl.len().alias("hn"))
fc = fc.join(hcnt, on=["feed", "cat"], how="left").with_columns(pl.col("hn").fill_null(0))
feedord = pl.DataFrame({"feed": np.arange(len(fi)), "user": f["user_id"].to_numpy()[fi], "ts": f["timestamp"].to_numpy()[fi].astype(np.int64)}).sort(["user", "ts", "feed"])
feedord = feedord.with_columns(pl.col("feed").shift(1).over("user").alias("prev"), (pl.col("ts") - pl.col("ts").shift(1).over("user")).alias("gap"))
pairs = feedord.drop_nulls("prev").select(["feed", "prev", "gap", "ts"])
a = fc.join(pairs, on="feed").join(fc.select([pl.col("feed").alias("prev"), "cat", pl.col("f3").alias("f3p"), pl.col("hn").alias("hnp"), pl.col("ts").alias("tsp")]), on=["prev", "cat"])
a = a.with_columns(((pl.col("ts") + 10800) // 86400 == (pl.col("tsp") + 10800) // 86400).alias("sameday"), (pl.col("f3") == pl.col("f3p")).alias("eq"))
print(f"   (feed, category) pairs present in both consecutive feeds: {len(a)}")
for lo, hi in bins:
    m = a.filter((pl.col("gap") >= lo) & (pl.col("gap") < hi))
    if len(m) == 0: continue
    print(f"   gap [{lo//60}m,{hi//60}m): f3 exactly equal {m['eq'].mean():.3f} (n={len(m)}); same-day {m.filter(pl.col('sameday'))['eq'].mean() if m['sameday'].sum() else float('nan'):.3f}, cross-day {m.filter(~pl.col('sameday'))['eq'].mean() if (~m['sameday']).sum() else float('nan'):.3f}")
# does f3 move with same-category history growth?
a = a.with_columns((pl.col("hn") - pl.col("hnp")).alias("dh"), (pl.col("f3") - pl.col("f3p")).alias("df3"))
sh = a.filter(pl.col("gap") < 86400)
for lab, cond in [("same-cat history count grew", pl.col("dh") > 0), ("unchanged", pl.col("dh") == 0), ("shrank", pl.col("dh") < 0)]:
    m = sh.filter(cond)
    if len(m) == 0: continue
    print(f"   gap < 1 d, {lab}: n {len(m)}, mean df3 {m['df3'].mean():+.4f}, share df3 > 0 {(m['df3'] > 0).mean():.3f}, share equal {m['eq'].mean():.3f}")

# share-normalised or time-decayed? among pairs with UNCHANGED same-category count: df3 vs gap, and vs growth of the feed's total history
tot = pl.DataFrame({"feed": np.arange(len(fi)), "H": fH})
a = a.join(tot, on="feed").join(tot.select([pl.col("feed").alias("prev"), pl.col("H").alias("Hp")]), on="prev")
un = a.filter(pl.col("dh") == 0)
print("   unchanged same-category count: mean df3 by gap: " + " | ".join(
    f"[{lo//60}m,{hi//60}m): {un.filter((pl.col('gap')>=lo)&(pl.col('gap')<hi))['df3'].mean():+.4f} (n={len(un.filter((pl.col('gap')>=lo)&(pl.col('gap')<hi)))})" for lo, hi in bins[1:]))
un2 = un.filter(pl.col("gap") < 6 * 3600)
for lab, cond in [("total history grew <= 2", (pl.col("H") - pl.col("Hp")) <= 2), ("grew 3-10", ((pl.col("H") - pl.col("Hp")) > 2) & ((pl.col("H") - pl.col("Hp")) <= 10)), ("grew > 10", (pl.col("H") - pl.col("Hp")) > 10)]:
    m = un2.filter(cond)
    if len(m): print(f"   unchanged same-cat count, gap < 6 h, {lab}: n {len(m)}, mean df3 {m['df3'].mean():+.4f}, share df3<0 {(m['df3']<0).mean():.3f}")
