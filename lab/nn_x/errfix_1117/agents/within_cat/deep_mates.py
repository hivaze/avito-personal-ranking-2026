"""Items at within-category rank c>=5: deviation of x from the mean of the c-1 same-category items above it.
Compare positives with negatives at the same within-category rank and similar overall rank (matched by strata)."""
from wc import *
df = load(); d = features(df)
F = ["score", "f1", "f4", "f5", "ucos", "hmax", "h1", "h5", "h10", "lhidx", "xp_imp_c50", "lpop", "lpop_et2", "ldup", "cand_maxcos", "pre2_fe_z", "pre2_new_z", "lcv", "sdiff", "f2", "n_hdup95", "c_contact", "c_fav", "c_vfeed"]
d = d.sort(["fidx", "cat", "cat_rank_in_feed"])
ex = []
for c in F:
    cs = pl.col(c).cum_sum().over(["fidx", "cat"]) - pl.col(c)
    ex.append((pl.col(c) - cs / (pl.col("cat_rank_in_feed") - 1).clip(1, None)).alias(c + "_vs_above"))
d = d.with_columns(ex).filter(pl.col("cat_rank_in_feed") >= 5)
d = d.with_columns(pl.col("cat_rank_in_feed").clip(None, 40).alias("cr"),
                   pl.when(pl.col("rank") <= 30).then(30).when(pl.col("rank") <= 100).then(100).otherwise(300).alias("rb"))
# stratified SMD: within each (cr, rb) stratum, mean(pos) - mean(neg), weighted by n_pos
out = []
for c in F:
    v = c + "_vs_above"
    t = d.group_by(["cr", "rb"]).agg(pl.col(v).filter(pl.col("label") == 1).mean().alias("mp"), pl.col(v).filter(pl.col("label") == 0).mean().alias("mn"),
                                     (pl.col("label") == 1).sum().alias("np"), pl.col(v).filter(pl.col("label") == 0).std().alias("sdn")).filter(pl.col("np") > 0)
    diff = ((t["mp"] - t["mn"]) * t["np"]).sum() / t["np"].sum()
    sdp = d.filter(pl.col("label") == 0)[v].std()
    se = sdp / np.sqrt(t["np"].sum())
    out.append((c, diff / sdp, diff / se, int(t["np"].sum())))
out.sort(key=lambda r: -abs(r[2]))
print("positives at within-cat rank>=5:", out[0][3])
for r in out: print("%-12s SMD %+.3f  z=%+.2f" % r[:3])
