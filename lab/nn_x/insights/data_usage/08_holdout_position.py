"""Holdout (30K random-like train feeds, M>=50; shown order ~random): model loss vs the shown rank of the positive.
Under a position-based click model, a click at a high-attention slot needs less relevance, so in RANDOM feeds
top-slot positives should be HARDER (higher L) for a position-blind model; contamination by production feeds would make
them EASIER. Also: are end-of-feed (last item) positives different? CPU, read-only."""
import numpy as np, polars as pl
from pathlib import Path
D = Path("<repo>/nn_x/insights/data_usage/out")
h = pl.read_parquet(D / "holdout_feed_losses.parquet")
h = h.with_columns(((pl.col("L_f7x_101") + pl.col("L_f7x_102") + pl.col("L_f7x_103") + pl.col("L_f7x_104")) / 4).alias("L_f7x"),
                   ((pl.col("L_snap101") + pl.col("L_snap102") + pl.col("L_snap103") + pl.col("L_snap104")) / 4).alias("L_snap"),
                   ((pl.col("N_f7x_101") + pl.col("N_f7x_102") + pl.col("N_f7x_103") + pl.col("N_f7x_104")) / 4).alias("N_f7x"),
                   ((pl.col("rho_f7x_101") + pl.col("rho_f7x_102") + pl.col("rho_f7x_103") + pl.col("rho_f7x_104")) / 4).alias("rho_f7x"))
h = h.with_columns(pl.when(pl.col("r0") == 0).then(pl.lit("a0")).when(pl.col("r0") <= 2).then(pl.lit("b1-2")).when(pl.col("r0") <= 9).then(pl.lit("c3-9"))
                   .when(pl.col("r0") <= 49).then(pl.lit("d10-49")).otherwise(pl.lit("e>=50")).alias("r0b"),
                   (pl.col("e0") == 0).alias("last"),
                   pl.when(pl.col("M") < 100).then(pl.lit("[50,100)")).when(pl.col("M") < 200).then(pl.lit("[100,200)")).otherwise(pl.lit("[200,300]")).alias("Mb"))
pl.Config.set_tbl_rows(100); pl.Config.set_tbl_width_chars(200)
s1 = h.filter(pl.col("npos") == 1).group_by(["Mb", "r0b"]).agg(n=pl.len(), L_f7x=pl.col("L_f7x").mean(), L_snap=pl.col("L_snap").mean(),
       se=pl.col("L_f7x").std() / pl.len().sqrt(), N_f7x=pl.col("N_f7x").mean()).sort(["Mb", "r0b"])
print("single-positive holdout feeds: loss by shown rank of the positive"); print(s1)
s2 = h.filter(pl.col("npos") == 1).group_by(["Mb", "last"]).agg(n=pl.len(), L_f7x=pl.col("L_f7x").mean(), se=pl.col("L_f7x").std() / pl.len().sqrt(),
       N_f7x=pl.col("N_f7x").mean()).sort(["Mb", "last"])
print("single-positive holdout feeds: last-item positive vs not"); print(s2)
print("mean per-feed Spearman(model rank, shown rank), F7x, by EDA posterior bucket:")
print(h.with_columns(pl.col("prand").cut([0.95, 0.98, 0.995]).alias("pb")).group_by("pb").agg(n=pl.len(), rho=pl.col("rho_f7x").mean(),
      se=pl.col("rho_f7x").std() / pl.len().sqrt()).sort("pb"))
s1.write_csv(D / "08_holdout_rank.csv"); s2.write_csv(D / "08_holdout_last.csv")
