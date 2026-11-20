"""Screen segment-specific slopes of within-category deviations (x_dm) in a feed-level conditional logit."""
from wc import *
df = load(); d = features(df)
F = ["f1", "f4", "f5", "ucos", "hmax", "h1", "h5", "h10", "lhidx", "xp_imp_c50", "lpop", "lpop_et2", "ldup", "cand_maxcos", "pre2_fe_z", "pre2_new_z", "lcv", "sdiff", "n_hdup95", "f2"]
d = rel(d, F)
d = d.with_columns(
    pl.when(pl.col("newest_type") == -1).then(pl.lit("nt_empty")).when(pl.col("newest_type") == 1).then(pl.lit("nt_view"))
      .when(pl.col("newest_type").is_in([2, 3])).then(pl.lit("nt_contact")).when(pl.col("newest_type") == 4).then(pl.lit("nt_fav"))
      .when(pl.col("newest_type").is_in([5, 6, 8])).then(pl.lit("nt_order")).when(pl.col("newest_type") == 10).then(pl.lit("nt_guest")).otherwise(pl.lit("nt_other")).alias("seg_nt"),
    pl.when(pl.col("M") < 50).then(pl.lit("M<50")).when(pl.col("M") < 150).then(pl.lit("M<150")).otherwise(pl.lit("M150+")).alias("seg_M"),
    pl.when(pl.col("cat_user_rank") == 1).then(pl.lit("cu1")).when(pl.col("cat_user_rank") == 0).then(pl.lit("cu0")).when(pl.col("cat_user_rank") <= 3).then(pl.lit("cu23")).otherwise(pl.lit("cu4+")).alias("seg_cu"),
    pl.when(pl.col("cat_n_in_feed") < 5).then(pl.lit("cn<5")).when(pl.col("cat_n_in_feed") < 20).then(pl.lit("cn<20")).otherwise(pl.lit("cn20+")).alias("seg_cn"),
    pl.col("same_cat_newest").cast(pl.Utf8).alias("seg_scn"),
)
topcats = d.filter(pl.col("label") == 1)["cat"].value_counts().sort("count", descending=True).head(14)["cat"].to_list()
d = d.with_columns(pl.when(pl.col("cat").is_in(topcats)).then(pl.col("cat").cast(pl.Utf8)).otherwise(pl.lit("other")).alias("seg_cat"))
off, sz = kit.feed_layout(d)
y = d["label"].to_numpy().astype(np.float64); s = d["score"].to_numpy().astype(np.float64)
a0, _, _, L0 = clogit(np.zeros((len(s), 0)), s, y, off, sz)
rows = []
for segc in ["seg_nt", "seg_M", "seg_cu", "seg_cn", "seg_scn", "seg_cat"]:
    seg = d[segc].to_numpy()
    for lv in np.unique(seg):
        m = (seg == lv)
        npos = y[m].sum()
        if npos < 150:
            continue
        for c in F:
            x = d[c + "_dm"].to_numpy().astype(np.float64)
            x = np.nan_to_num(x) * m
            sd = x[m].std()
            if sd == 0:
                continue
            X = (x / sd)[:, None]
            a, b, se, L = clogit(X, s, y, off, sz, l2=0)
            rows.append((segc, str(lv), c, int(npos), b[0], b[0] / se[0]))
rows.sort(key=lambda r: -abs(r[5]))
print("tests", len(rows))
for r in rows[:40]: print("%-8s %-10s %-12s npos=%5d coef/sd %+.4f z=%+.2f" % r)
