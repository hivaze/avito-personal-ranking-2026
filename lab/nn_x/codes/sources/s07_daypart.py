"""s07: desktop-vs-mobile style usage rhythm per source. Train sample 600K feeds (Sep 6-19; weekends Sep 6, 12, 13, 19).
Newest source-bearing events (hpos < 8, types 1/4/7) are dated by the feed time. Per source: share on weekend vs weekday,
weekday work hours (10-18 MSK) vs weekday evening (19-24 MSK) vs night (0-6), relative to the all-source profile."""
import sys, numpy as np, polars as pl
sys.path.insert(0, "<repo>/nn_x/codes/sources")
from scommon import load_hist, train_sample, pshow, OUT
df, fs = load_hist("train", train_sample(600_000, seed=7))
d = df.filter((pl.col("hpos") < 8) & pl.col("et").is_in([1, 4, 7])).with_columns(((pl.col("ts") + 3 * 3600) % 86400 // 3600).alias("h"), (((pl.col("ts") + 3 * 3600) // 86400 + 3) % 7).alias("dow"))
# dow: 0=Mon..6=Sun (1970-01-01 was a Thursday -> +3; Sep 6 2026 MSK = Sunday = 6)
print(d.select(pl.col("ts").min(), pl.col("dow").filter(pl.col("ts") == pl.col("ts").min()).first()))
d = d.with_columns(pl.col("dow").is_in([5, 6]).alias("wkend"),
                   pl.when(pl.col("h") < 6).then(pl.lit("night0_6")).when(pl.col("h") < 10).then(pl.lit("morn6_10")).when(pl.col("h") < 19).then(pl.lit("work10_19")).otherwise(pl.lit("eve19_24")).alias("part"))
base = d.group_by(["wkend", "part"]).len().with_columns((pl.col("len") / pl.col("len").sum()).alias("pb")).select("wkend", "part", "pb")
g = d.group_by(["es", "wkend", "part"]).len().with_columns((pl.col("len") / pl.col("len").sum().over("es")).alias("p")).join(base, on=["wkend", "part"]).with_columns((pl.col("p") / pl.col("pb")).alias("lift"))
g = g.with_columns((pl.when(pl.col("wkend")).then(pl.lit("WE_")).otherwise(pl.lit("WD_")) + pl.col("part")).alias("cell")).filter(pl.col("es") <= 16)
t = g.pivot(values="lift", index="es", on="cell", sort_columns=True).sort("es")
n = d.group_by("es").len().rename({"len": "n_events"})
t = t.join(n, on="es").with_columns((pl.col("WE_work10_19") + 0).alias("_"))
pshow(t.drop("_"))
t.drop("_").write_csv(f"{OUT}/s07_daypart_lift.csv")
