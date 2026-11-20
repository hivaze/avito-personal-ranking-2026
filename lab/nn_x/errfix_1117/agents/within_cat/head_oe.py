"""O/E in the head (top-K) for discrete item cells; E = softmax of a*score within the head, scaled to the head's positives."""
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
    pl.when(pl.col("hmax") >= 0.99).then(pl.lit("a>=.99")).when(pl.col("hmax") >= 0.95).then(pl.lit("b.95-.99")).when(pl.col("hmax") >= 0.9).then(pl.lit("c.90-.95")).otherwise(pl.lit("d<.90")).alias("hmaxb"),
    pl.when(pl.col("hmax_idx") < 5).then(pl.lit("0-4")).when(pl.col("hmax_idx") < 32).then(pl.lit("5-31")).when(pl.col("hmax_idx") < 100).then(pl.lit("32-99")).otherwise(pl.lit("100+")).alias("hidxb"),
    pl.when(pl.col("n_cand_dup95") == 0).then(0).when(pl.col("n_cand_dup95") <= 2).then(1).otherwise(2).alias("dupb"),
    (pl.col("f4") * 5).floor().clip(0, 4).alias("f4q"), (pl.col("f5") * 5).floor().clip(0, 4).alias("f5q"),
    pl.when(pl.col("xp_imp_c50") == 0).then(pl.lit("cold")).when(pl.col("xp_imp_c50") < 2).then(pl.lit("low")).when(pl.col("xp_imp_c50") < 3.9).then(pl.lit("mid")).otherwise(pl.lit("cap")).alias("xpb"),
    pl.col("cat_user_rank").clip(0, 4).alias("cub"),
    pl.when(pl.col("cat_rank_in_feed") == 1).then(pl.lit("1")).when(pl.col("cat_rank_in_feed") <= 3).then(pl.lit("2-3")).otherwise(pl.lit("4+")).alias("crb"),
    pl.when(pl.col("c_contact") >= 0.95).then(pl.lit("cc>=.95")).when(pl.col("c_contact_na")).then(pl.lit("na")).otherwise(pl.lit("cc<.95")).alias("ccb"),
    pl.when(pl.col("c_fav") >= 0.95).then(pl.lit("cf>=.95")).when(pl.col("c_fav_na")).then(pl.lit("na")).otherwise(pl.lit("cf<.95")).alias("cfb"),
    pl.when(pl.col("c_vfeed") >= 0.95).then(pl.lit("cv>=.95")).when(pl.col("c_vfeed_na")).then(pl.lit("na")).otherwise(pl.lit("cv<.95")).alias("cvb"),
    pl.when(pl.col("n_hdup95") == 0).then(0).when(pl.col("n_hdup95") <= 3).then(1).otherwise(2).alias("hdupb"),
)
for by in ["hmaxb", "hidxb", ["hmaxb", "hidxb"], "hmax_type", "hmax_src", "dupb", "f4q", "f5q", "xpb", "cub", "crb", "ccb", "cfb", "cvb", "hdupb", "same_cat_newest", "f2_home"]:
    t = h.group_by(by).agg(pl.len().alias("n"), pl.col("label").sum().alias("O"), pl.col("E").sum().alias("E"))
    t = t.with_columns((pl.col("O") / pl.col("E")).round(3).alias("O/E"), ((pl.col("O") - pl.col("E")) / pl.col("E").sqrt()).round(2).alias("z"), pl.col("E").round(1)).filter(pl.col("n") >= 200).sort(by)
    print(t)
