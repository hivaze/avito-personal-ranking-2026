from wc import *
pl.Config.set_tbl_rows(100); pl.Config.set_tbl_cols(20); pl.Config.set_fmt_str_lengths(30)
df = load(); d = add_expect(features(df))
cols = ["f1", "f4", "f5", "ucos", "hmax", "h1", "h5", "h10", "lhidx", "xp_imp_c50", "lpop", "lpop_et2", "cand_maxcos", "ldup", "pre2_fe_z", "pre2_new_z", "lcv", "sdiff", "n_hdup95"]
d = rel(d.filter(pl.col("cat_n_in_feed") >= 10), cols)
res = []
for c in cols:
    t = d.with_columns((pl.col(c + "_pr") * 5).floor().clip(0, 4).alias("q")).group_by("q").agg(pl.col("label").sum().alias("O"), pl.col("E").sum().alias("E")).sort("q")
    oe_ = (t["O"] / t["E"]).to_numpy(); z = ((t["O"] - t["E"]) / t["E"].sqrt()).to_numpy()
    res.append((c, " ".join("%.3f" % v for v in oe_), " ".join("%+.1f" % v for v in z)))
for r in res: print("%-12s O/E by within-cat quintile: %s | z %s" % r)
