from wc import *
pl.Config.set_tbl_rows(50)
K = 10
df = load(); d = features(df)
d = d.with_columns(pl.when(pl.col("n_cand_dup95") >= 1).then(pl.len().over(["fidx", "cand_maxcos"])).otherwise(0).alias("k"))
d = d.with_columns(pl.when(pl.col("k") == 2).then(pl.col("rank").min().over(["fidx", "cand_maxcos"])).otherwise(9999).alias("pair_top"),
                   pl.when(pl.col("k") == 2).then(pl.col("rank").max().over(["fidx", "cand_maxcos"])).otherwise(9999).alias("pair_bot"))
d = d.with_columns(pl.when(pl.col("k") != 2).then(pl.lit("nopair")).when(pl.col("rank") == pl.col("pair_top")).then(pl.lit("upper")).otherwise(pl.lit("lower")).alias("role"),
                   pl.when(pl.col("k") != 2).then(pl.lit("-")).when(pl.col("pair_bot") <= K).then(pl.lit("both_in_head")).otherwise(pl.lit("partner_out")).alias("where"))
h = d.filter(pl.col("rank") <= K).sort(["fidx", "rank"])
h = h.with_columns(pl.col("label").sum().over("fidx").alias("hp")).filter(pl.col("hp") > 0)
off, sz = kit.feed_layout(h)
y = h["label"].to_numpy().astype(float); s = h["score"].to_numpy().astype(float)
a, _, _, _ = clogit(np.zeros((len(s), 0)), s, y, off, sz)
h = h.with_columns(pl.Series("z", a * s)).with_columns((pl.col("z") - pl.col("z").max().over("fidx")).exp().alias("e"))
h = h.with_columns((pl.col("e") / pl.col("e").sum().over("fidx") * pl.col("hp")).alias("E"))
t = h.group_by(["role", "where"]).agg(pl.len().alias("n"), pl.col("label").sum().alias("O"), pl.col("E").sum().alias("E"))
print(t.with_columns((pl.col("O") / pl.col("E")).round(3).alias("O/E"), ((pl.col("O") - pl.col("E")) / pl.col("E").sqrt()).round(2).alias("z")).sort(["role", "where"]))
