from wc import *
import pandas as pd
pd.set_option("display.width", 260); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 200)
df = load()
cols = ["rank", "label", "score", "s_ft", "s_to", "cat", "cat_rank_in_feed", "f1", "f4", "f5", "ucos", "hmax", "hmax_idx", "hmax_type", "h1", "h5", "h10",
        "cand_maxcos", "n_cand_dup95", "n_hdup95", "it_hist_all_luo", "xp_imp_c50", "seq_cv_invd20", "pre2_fe_z", "c_contact", "c_fav"]
mode = sys.argv[1]; n = int(sys.argv[2]); seed = int(sys.argv[3])
pos = df.filter(pl.col("label") == 1).group_by("fidx").agg(pl.col("rank").min().alias("best"), pl.col("cat_rank_in_feed").min().alias("cr"))
if mode == "bad":
    fs = pos.filter((pl.col("best") > 30) & (pl.col("cr") >= 8))
else:
    fs = pos.filter(pl.col("best") <= 3)
fs = fs.sample(n, seed=seed)["fidx"].to_list()
for f in fs:
    d = df.filter(pl.col("fidx") == f).sort("rank")
    r0 = d.row(0, named=True)
    print(f"=== feed {f} user {r0['user_id']} M={r0['M']} H={r0['H']} guest={r0['guest']} newest type/src/cat={r0['newest_type']}/{r0['newest_src']}/{r0['newest_cat']} nhc32={r0['n_hist_cats32']} hour={r0['hour']}")
    p = d.filter(pl.col("label") == 1)
    cats = p["cat"].unique().to_list()
    show = d.filter((pl.col("rank") <= 12) | (pl.col("label") == 1) | (pl.col("cat").is_in(cats) & (pl.col("rank") <= 40)))
    print(show.select(cols).to_pandas().round(3).to_string(index=False))
    # category summary of user history
    print(d.group_by("cat").agg(pl.len().alias("n_in_feed"), pl.col("cat_user_rank").first(), pl.col("cat_hist_share").first().round(3), pl.col("cat_newest_idx").first(), pl.col("label").sum()).sort("n_in_feed", descending=True).head(6).to_pandas().to_string(index=False))
