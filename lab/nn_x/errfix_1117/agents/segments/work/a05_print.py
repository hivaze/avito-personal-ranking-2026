"""Print individual feeds: good (best positive rank 1-3) and bad (best positive 30+)."""
import fx, kit, numpy as np, polars as pl, sys
df = kit.load()
c = fx.Ctx(df)
F = pl.read_parquet("feeds.parquet")
rng = np.random.default_rng(int(sys.argv[2]) if len(sys.argv) > 2 else 1)
kind = sys.argv[1] if len(sys.argv) > 1 else "bad"
best = F["best"].to_numpy(); M = F["M"].to_numpy()
if kind == "bad":
    cand = np.nonzero((best > 30) & (M >= 60))[0]
elif kind == "good":
    cand = np.nonzero((best <= 3) & (M >= 60))[0]
elif kind == "smallbad":
    cand = np.nonzero((best > 5) & (M < 30))[0]
pick = rng.choice(cand, size=min(8, len(cand)), replace=False)
cols = ["rank", "score", "s_ft", "s_to", "cat", "f1", "f2", "f3", "f4", "f5", "ucos", "h1", "h5", "hmax", "hmax_idx",
        "hmax_type", "c_contact", "cat_user_rank", "cat_newest_idx", "cat_views", "cat_actions", "cat_rank_in_feed", "cat_n_in_feed",
        "xp_imp_c50", "it_hist_all_luo", "seq_cv_invd20", "pre2_fe_z", "cand_maxcos", "label"]
pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(400); pl.Config.set_tbl_cols(40)
pl.Config.set_float_precision(2); pl.Config.set_tbl_hide_dataframe_shape(True); pl.Config.set_tbl_hide_column_data_types(True)
for i in pick:
    o, m = c.off[i], c.sz[i]
    sub = df[o:o + m]
    r = sub.row(0, named=True)
    print(f"\n#### feed {r['fidx']} M={m} H={r['H']} guest={r['guest']} uvm={r['uvec_missing']} hour={r['hour']} wd={r['weekday']} "
          f"newest t/s/c={r['newest_type']}/{r['newest_src']}/{r['newest_cat']} nhc32={r['n_hist_cats32']} nfc={r['n_feed_cats']} "
          f"npos={int(c.npos[i])} best={best[i]} ndcg={c.base_nd[i]:.3f}")
    # category summary: counts of feed cats and positives
    cs = sub.group_by("cat").agg(pl.len().alias("n"), pl.col("label").sum().alias("pos"), pl.col("rank").min().alias("minrank"),
                                 pl.col("cat_user_rank").first().alias("ucr"), pl.col("cat_newest_idx").first().alias("cni"),
                                 pl.col("f3").first().alias("f3")).sort("minrank")
    print("cats:", " ".join(f"{a}:{n}/{p}/r{mr}/u{u}/i{ci}/f3={f:.2f}" for a, n, p, mr, u, ci, f in cs.iter_rows()))
    show = sub.filter((pl.col("rank") <= 8) | (pl.col("label") == 1)).sort("rank").select(cols)
    print(show)
