"""Lens b: positives ranked deep. Within the same feed and the same rank region, how do positives differ from negatives?"""
from util import *

df = load()
NUM = ["f1", "f2", "f3", "f4", "f5", "xp_imp_c50", "it_hist_all_luo", "it_hist_et1_luo", "it_hist_et2_luo", "it_hist_et4_luo",
       "seq_cv_invd20", "pre2_fe_z", "pre2_new_z", "ucos", "hmax", "hmax_idx", "h1", "h5", "h10", "h32", "hmean5", "n_hdup95",
       "c_contact", "c_fav", "c_vfeed", "c_vsearch", "cat_views", "cat_actions", "cat_newest_idx", "cat_user_rank",
       "cat_hist_share", "cand_maxcos", "n_cand_dup95", "cat_n_in_feed", "cat_rank_in_feed", "s_ft", "s_to", "score", "gap_next"]
BOOL = ["same_cat_newest", "f2_home"]


def compare(sub, label):
    # per feed: mean over positives minus mean over negatives (same feed, same rank region); average over feeds
    sub = sub.with_columns([pl.col(x).cast(pl.Float64).fill_nan(None) for x in NUM + BOOL])
    g = sub.group_by(["fidx", "label"]).agg([pl.col(x).mean() for x in NUM + BOOL])
    p = g.filter(pl.col("label") == 1).drop("label")
    n = g.filter(pl.col("label") == 0).drop("label")
    j = p.join(n, on="fidx", suffix="_n")
    print(f"=== {label}: feeds {j.height}")
    out = []
    for x in NUM + BOOL:
        dd = (j[x] - j[x + "_n"]).drop_nulls().to_numpy()
        if len(dd) < 20:
            continue
        sd = sub[x].drop_nulls().std()
        z = dd.mean() / (dd.std(ddof=1) / np.sqrt(len(dd)) + 1e-12)
        out.append((abs(z), x, dd.mean(), dd.mean() / (sd + 1e-12), z, len(dd)))
    for a, x, m, ms, z, nn in sorted(out, reverse=True)[:25]:
        print(f"   {x:18s} diff={m:+.4f} (={ms:+.3f} sd) z={z:+.2f} n={nn}")


compare(df.filter((pl.col("M") < 120) & (pl.col("rank") >= 50)), "M<120, rank>=50")
compare(df.filter((pl.col("M") >= 120) & (pl.col("rank") >= 50)), "M>=120, rank>=50")
compare(df.filter((pl.col("rank") <= 10)), "rank<=10")
compare(df.filter((pl.col("rank") > 10) & (pl.col("rank") <= 30)), "rank 11-30")
