import hk, numpy as np, polars as pl
c = hk.Ctx(); df = c.df
r1 = df.filter(pl.col("rank") == 1)
r1 = r1.with_columns((pl.col("gap_next") > 0.6).alias("big"))
print(r1.group_by("big").agg(pl.len(), pl.col("label").mean(), pl.col("M").median(), pl.col("hmax").mean(), pl.col("hmax_idx").median(), pl.col("n_hdup95").mean(), pl.col("f4").mean(), pl.col("xp_imp_c50").mean(), pl.col("c_contact").mean(), pl.col("cat_newest_idx").median(), pl.col("h1").mean()))
b = r1.filter(pl.col("big"))
cols = ["hmax", "h1", "h5", "hmax_idx", "hmax_type", "n_hdup95", "f4", "f5", "xp_imp_c50", "c_contact", "c_fav", "c_vfeed", "c_vsearch", "cand_maxcos", "n_cand_dup95", "cat_n_in_feed", "pre2_fe_z", "same_cat_newest", "M", "H", "guest"]
print("big-lead rank1: pos vs neg means")
for col in cols:
    x = b[col].cast(pl.Float64).to_numpy(); y = b["label"].to_numpy()
    print(f"  {col:16s} pos={np.nanmean(x[y==1]):.3f} neg={np.nanmean(x[y==0]):.3f}  n_pos={int((y==1).sum())}")
