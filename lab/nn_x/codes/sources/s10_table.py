"""s10: assemble the per-source key-statistics table (val unless noted) from the s01-s06 outputs -> out/s10_source_keystats.csv"""
import polars as pl
O = "<repo>/nn_x/codes/sources/out"
p = pl.read_csv(f"{O}/s01_profile_val.csv").select("src", "n", "share", "share_et1", "share_et4", "share_et7", "hpos_mean", "hpos1_mean", "feeds_with", "noemb", "cat_top", "js_vs_t1",
                                                    "burst_sb", "burst_lift_sb", "rep_item", "multi_src_item", "user0_ev_share", "share_in_user0", "share_in_logged")
ptr = pl.read_csv(f"{O}/s01_profile_train.csv").select("src", pl.col("n").alias("n_train200k"), pl.col("share").alias("share_train"))
d = pl.read_csv(f"{O}/s02d_prevsim_val.csv").select(pl.col("es").alias("src"), "cos_prev", "cos_prev_med", "samecat_prev", "prev_same_src")
a = pl.read_csv(f"{O}/s02a_followup_val.csv").select(pl.col("src0").alias("src"), pl.col("contact").alias("fu_contact"), pl.col("msg").alias("fu_msg"), pl.col("t4").alias("fu_t4"), pl.col("t10").alias("fu_t10"), pl.col("review").alias("fu_review"))
e = pl.read_csv(f"{O}/s02e_preceding_val.csv").filter(pl.col("et") == 1).select(pl.col("es").alias("src"), pl.col("view_before").alias("v_view_before"), pl.col("contact_before").alias("v_contact_before"))
b = pl.read_csv(f"{O}/s04B_pool_val.csv").select(pl.col("es").alias("src"), "in_split_pool", "in_train_pool", "was_user_train_cand", "was_user_train_pos")
m = pl.read_csv(f"{O}/s06_msgcat_share.csv").select(pl.col("es").alias("src"), pl.col("share_views_in_msg_cats").alias("views_in_deliverycats"))
x = pl.read_csv(f"{O}/s05_platform_proxies.csv").select("src", "oe_type10", "oe_anon")
t = p.join(ptr, on="src", how="left")
for o in [d, a, e, b, m, x]:
    t = t.join(o, on="src", how="left")
t = t.filter(pl.col("src") <= 16)
t.write_csv(f"{O}/s10_source_keystats.csv")
with pl.Config(tbl_rows=40, tbl_cols=60, tbl_width_chars=600, float_precision=3):
    print(t)
