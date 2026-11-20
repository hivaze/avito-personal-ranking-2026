"""Head analysis: when is rank 1 worse than rank 2 (or than the best of ranks 2-5)?"""
from util import *

df = load()
c = Ctx(df)
pl.Config.set_tbl_rows(100)
pl.Config.set_tbl_width_chars(250)
top = df.filter(pl.col("rank") <= 5).sort(["fidx", "rank"])
cols = ["label", "cat", "xp_imp_c50", "f4", "f1", "f2", "f5", "same_cat_newest", "cat_user_rank", "n_hdup95", "hmax",
        "hmax_idx", "hmax_type", "h1", "ucos", "score", "s_ft", "s_to", "gap_next", "cat_newest_idx", "n_cand_dup95",
        "cand_maxcos", "it_hist_all_luo", "seq_cv_invd20", "pre2_fe_z", "c_fav", "c_contact", "cat_n_in_feed", "f2_home"]
wide = None
for r in range(1, 6):
    t = top.filter(pl.col("rank") == r).select(["fidx"] + [pl.col(x).alias(f"{x}_{r}") for x in cols])
    wide = t if wide is None else wide.join(t, on="fidx", how="left")
feed = df.group_by("fidx", maintain_order=True).agg(pl.col("M").first(), pl.col("guest").first(),
                                                     pl.col("newest_type").first(), pl.col("newest_src").first(),
                                                     pl.col("n_hist_cats32").first(), pl.col("label").sum().alias("npos"),
                                                     pl.col("H").first())
wide = wide.join(feed, on="fidx")
wide = wide.with_columns((pl.col("label_2") - pl.col("label_1")).alias("d21"),
                         (pl.col("score_1") - pl.col("score_2")).alias("g12"))
print("overall y1 %.4f y2 %.4f" % (wide["label_1"].mean(), wide["label_2"].mean()))


def seg(expr, name):
    g = (wide.with_columns(expr.alias("bin")).group_by("bin")
         .agg(pl.len().alias("n"), pl.col("label_1").mean().round(4).alias("y1"), pl.col("label_2").mean().round(4).alias("y2"),
              pl.col("d21").mean().round(4).alias("d21"), (pl.col("d21").std() / pl.len().sqrt()).round(4).alias("se"))
         .with_columns((pl.col("d21") / pl.col("se")).round(2).alias("z")).sort("bin"))
    print("---", name)
    for r in g.iter_rows():
        print("   ", " | ".join(str(v) for v in r))


seg(pl.col("xp_imp_c50_1") == 0, "r1 cold")
seg(pl.col("f4_1") < 0.12, "r1 fresh10")
seg((pl.col("xp_imp_c50_1") == 0) & (pl.col("f4_1") < 0.25), "r1 cold&fresh")
seg(pl.col("same_cat_newest_1").cast(pl.Int8) * 2 + pl.col("same_cat_newest_2").cast(pl.Int8), "scn r1*2+r2")
seg(pl.col("cat_1") == pl.col("cat_2"), "same cat r1 r2")
seg(pl.col("cat_user_rank_1") == 0, "r1 cat absent")
seg(pl.col("n_hdup95_1").clip(0, 3), "r1 hd95")
seg(pl.col("hmax_idx_1") == 0, "r1 nearest = newest")
seg(pl.col("hmax_1") >= 0.98, "r1 hmax>=.98")
seg(pl.col("h1_1") >= 0.95, "r1 h1>=.95")
seg(pl.col("ucos_1") < 0.40, "r1 ucos<.40")
seg(pl.col("g12").qcut(5, allow_duplicates=True).cast(pl.Utf8), "gap12 q")
seg(pl.col("hmax_type_1"), "r1 hmax_type")
seg(pl.col("guest"), "guest")
seg(pl.col("n_cand_dup95_1").clip(0, 2), "r1 cand dup")
seg(pl.col("cand_maxcos_1") >= 0.98, "r1 cand_maxcos>=.98")
seg((pl.col("s_ft_1") - pl.col("s_ft_2")) * (pl.col("s_to_1") - pl.col("s_to_2")) < 0, "branches disagree on r1 vs r2")
seg(pl.col("f2_home_1"), "r1 home region")
seg(pl.col("f5_1") < 0.1, "r1 f5<.1")
seg(pl.col("it_hist_all_luo_1") == 0, "r1 never in others' histories")
seg(pl.col("newest_type"), "newest_type")
seg(pl.col("n_hist_cats32").clip(0, 8), "n_hist_cats32")
seg(pl.col("cat_n_in_feed_1").cut([1, 3, 10, 30]).cast(pl.Utf8), "r1 cat_n_in_feed")
seg(pl.col("M").cut([30, 60, 120, 200]).cast(pl.Utf8), "M")
wide.write_parquet("<repo>/lab/nn_x/errfix_1117/agents/deep_dive/scripts/head_wide.parquet")
