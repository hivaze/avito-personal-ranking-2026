from wc import *
pl.Config.set_tbl_rows(200); pl.Config.set_tbl_cols(12)
K = int(sys.argv[1])
df = load(); d = features(df)
h = d.filter(pl.col("rank") <= K).sort(["fidx", "rank"])
h = h.with_columns(pl.col("label").sum().over("fidx").alias("hp")).filter(pl.col("hp") > 0)
off, sz = kit.feed_layout(h)
y = h["label"].to_numpy().astype(float); s = h["score"].to_numpy().astype(float)
a, _, _, _ = clogit(np.zeros((len(s), 0)), s, y, off, sz)
h = h.with_columns(pl.Series("z", a * s)).with_columns((pl.col("z") - pl.col("z").max().over("fidx")).exp().alias("e"))
h = h.with_columns((pl.col("e") / pl.col("e").sum().over("fidx") * pl.col("hp")).alias("E"))
h = h.with_columns(
    (pl.col("hmax_idx") == pl.col("cat_newest_idx")).alias("near_is_catnewest"),
    pl.when(pl.col("cat_newest_idx") < 0).then(pl.lit("absent")).when(pl.col("cat_newest_idx") == 0).then(pl.lit("0")).when(pl.col("cat_newest_idx") < 5).then(pl.lit("1-4"))
      .when(pl.col("cat_newest_idx") < 32).then(pl.lit("5-31")).otherwise(pl.lit("32+")).alias("cnib"),
    pl.when(pl.col("h1") >= 0.95).then(pl.lit("h1>=.95")).when(pl.col("h1") >= 0.85).then(pl.lit("h1 .85-.95")).otherwise(pl.lit("h1<.85")).alias("h1b"),
    (pl.col("cat_n_in_feed") / pl.col("M") * 4).floor().clip(0, 3).alias("csq"),
    pl.when(pl.col("M") < 50).then(pl.lit("M<50")).when(pl.col("M") < 150).then(pl.lit("M<150")).otherwise(pl.lit("M150+")).alias("Mb"),
)
for by in ["near_is_catnewest", "cnib", ["same_cat_newest", "h1b"], "csq", ["Mb", "csq"], ["Mb", "cub"] if False else "newest_type", "cat"]:
    t = h.group_by(by).agg(pl.len().alias("n"), pl.col("label").sum().alias("O"), pl.col("E").sum().alias("E"))
    t = t.with_columns((pl.col("O") / pl.col("E")).round(3).alias("O/E"), ((pl.col("O") - pl.col("E")) / pl.col("E").sqrt()).round(2).alias("z"), pl.col("E").round(1)).filter(pl.col("n") >= 300).sort(by)
    print(t)
