"""O/E with E from a cross-fitted calibration of the model's own outputs (score, gap_top, rank, M)."""
from util import *

D = "<repo>/lab/nn_x/errfix_1117/agents/deep_dive/scripts/"
df = load()
E = np.load(D + "E_calib.npy")
fl = pl.read_parquet(D + "feedlevel.parquet")
df = df.with_columns(pl.Series("E0", E)).with_columns(
    (pl.col("E0") * pl.col("label").sum().over("fidx") / pl.col("E0").sum().over("fidx")).alias("E")).join(fl.select("fidx", "gap12", "lab_r1"), on="fidx", how="left")
df = df.with_columns(pl.when(pl.col("rank") == 1).then(pl.lit("r1")).when(pl.col("rank") <= 3).then(pl.lit("r2-3"))
                     .when(pl.col("rank") <= 10).then(pl.lit("r4-10")).when(pl.col("rank") <= 30).then(pl.lit("r11-30"))
                     .otherwise(pl.lit("r31+")).alias("rk"))
print("total O/E", df["label"].sum() / df["E"].sum())

pl.Config.set_tbl_rows(200)
pl.Config.set_tbl_width_chars(250)
pl.Config.set_fmt_str_lengths(40)


def oe(expr, name, ranks=("r1", "r2-3", "r4-10", "r11-30", "r31+"), minn=0):
    d = df.with_columns(expr.alias("bin"))
    g = (d.group_by(["bin", "rk"]).agg(pl.len().alias("n"), pl.col("label").sum().alias("O"), pl.col("E").sum().alias("E"))
         .with_columns((pl.col("O") / pl.col("E")).round(2).alias("OE"),
                       ((pl.col("O") - pl.col("E")) / pl.col("E").sqrt()).round(1).alias("z")))
    w = g.pivot(on="rk", index="bin", values=["OE", "z"]).sort("bin")
    cols = ["bin"] + [f"OE_{r}" for r in ranks if f"OE_{r}" in w.columns] + [f"z_{r}" for r in ranks if f"z_{r}" in w.columns]
    tot = d.group_by("bin").agg(pl.len().alias("n"), pl.col("label").sum().alias("O"), (pl.col("label").sum() / pl.col("E").sum()).round(3).alias("OEall"))
    print(f"--- {name}")
    t = w.select(cols).join(tot, on="bin").sort("bin")
    print("   " + " | ".join(f"{c:>9s}" for c in t.columns))
    for r in t.iter_rows():
        print("   " + " | ".join(f"{str(v)[:9]:>9s}" for v in r))


import sys
oe(pl.col("xp_imp_c50") == 0, "cold xp=0")
oe(pl.col("f4").qcut(10, allow_duplicates=True).cast(pl.Utf8), "f4 decile")
oe(pl.col("n_hdup95").clip(0, 50).cut([0, 1, 3, 10, 30]).cast(pl.Utf8), "n_hdup95")
oe(pl.col("hmax").cut([0.7, 0.8, 0.9, 0.95, 0.98, 0.99]).cast(pl.Utf8), "hmax")
oe(pl.col("h1").cut([0.7, 0.8, 0.9, 0.95, 0.98, 0.99]).cast(pl.Utf8), "h1 (newest event cos)")
oe(pl.col("cat_user_rank").clip(0, 6), "cat_user_rank")
oe(pl.col("cat_n_in_feed").cut([1, 2, 5, 10, 30, 80]).cast(pl.Utf8), "cat_n_in_feed")
oe(pl.col("same_cat_newest"), "same_cat_newest")
oe(pl.col("cat_newest_idx").cut([-1, 0, 2, 5, 15, 50]).cast(pl.Utf8), "cat_newest_idx")
oe(pl.col("guest"), "guest")
oe(pl.col("f2_home"), "f2_home")
oe(pl.col("f2").cut([0, 0.2, 0.5, 0.8]).cast(pl.Utf8), "f2")
oe(pl.col("f5").qcut(5, allow_duplicates=True).cast(pl.Utf8), "f5 q")
oe(pl.col("f1").qcut(5, allow_duplicates=True).cast(pl.Utf8), "f1 q")
oe(pl.col("cand_maxcos").cut([0.8, 0.9, 0.95, 0.98]).cast(pl.Utf8), "cand_maxcos")
oe(pl.col("n_cand_dup95").clip(0, 10).cut([0, 1, 3]).cast(pl.Utf8), "n_cand_dup95")
oe(pl.col("hmax_type"), "hmax_type")
oe(pl.col("hmax_src"), "hmax_src")
oe(pl.col("newest_type"), "newest_type")
oe(pl.col("newest_src"), "newest_src")
oe((pl.col("s_ft") - pl.col("s_to")).cut([-0.5, -0.2, 0, 0.2, 0.5]).cast(pl.Utf8), "s_ft - s_to")
oe(pl.col("ucos").qcut(5, allow_duplicates=True).cast(pl.Utf8), "ucos q")
oe(pl.col("it_hist_all_luo").cut([0, 2, 10, 50, 200]).cast(pl.Utf8), "it_hist_all")
oe(pl.col("seq_cv_invd20").cut([0, 0.1, 0.5, 2]).cast(pl.Utf8), "seq_cv")
oe(pl.col("M").cut([30, 60, 120, 200]).cast(pl.Utf8), "M")
oe(pl.col("H").cut([0, 10, 50, 150, 255]).cast(pl.Utf8), "H")
oe(pl.col("hour").cut([5, 9, 13, 17, 21]).cast(pl.Utf8), "hour")
oe(pl.col("weekday"), "weekday")
oe(pl.col("c_contact").is_nan() | pl.col("c_contact").is_null(), "no contact events")
oe(pl.col("c_order").cut([0.8, 0.9, 0.95, 0.98]).cast(pl.Utf8), "c_order")
oe(pl.col("c_fav").cut([0.8, 0.9, 0.95, 0.98]).cast(pl.Utf8), "c_fav")
oe(pl.col("c_contact").cut([0.8, 0.9, 0.95, 0.98]).cast(pl.Utf8), "c_contact")
oe(pl.col("hmax_idx").cut([-1, 0, 2, 10, 32, 100]).cast(pl.Utf8), "hmax_idx")
oe(pl.col("cat_rank_in_feed").cut([1, 2, 3, 5, 10, 30]).cast(pl.Utf8), "cat_rank_in_feed")
oe(pl.col("n_feed_cats").cut([5, 10, 15, 20, 25]).cast(pl.Utf8), "n_feed_cats")
oe(pl.col("n_hist_cats32").cut([1, 2, 3, 5, 8]).cast(pl.Utf8), "n_hist_cats32")
