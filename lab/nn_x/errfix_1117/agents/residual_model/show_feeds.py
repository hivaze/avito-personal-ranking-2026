"""Print individual feeds: the top-5 rows by model score and every positive, with the main columns.

usage: python show_feeds.py <bad|good> <n_feeds> [seed]
"""
import sys

import numpy as np
import polars as pl

import prep
from prep import kit

kind, n = sys.argv[1], int(sys.argv[2])
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
df = kit.load()
off, sz = kit.feed_layout(df)
y = df["label"].to_numpy()
rk = df["rank"].to_numpy()
best = np.array([rk[o:o + m][y[o:o + m] == 1].min() for o, m in zip(off, sz)])
cand = np.nonzero(best >= 30)[0] if kind == "bad" else np.nonzero(best <= 3)[0]
rng = np.random.default_rng(seed)
pick = rng.choice(cand, n, replace=False)
cols = ["rank", "label", "score", "s_ft", "s_to", "cat", "f1", "f2", "f3", "f4", "f5", "xp_imp_c50",
        "it_hist_all_luo", "ucos", "hmax", "hmax_idx", "hmax_type", "h1", "h5", "c_vfeed", "c_vsearch",
        "c_contact", "cat_views", "cat_actions", "cat_newest_idx", "cat_user_rank", "cand_maxcos", "n_cand_dup95",
        "seq_cv_invd20", "pre2_fe_z"]
pl.Config.set_tbl_cols(40)
pl.Config.set_tbl_width_chars(400)
pl.Config.set_tbl_rows(40)
pl.Config.set_float_precision(2)
pl.Config.set_tbl_hide_dataframe_shape(True)
pl.Config.set_tbl_hide_column_data_types(True)
for f in pick:
    o, m = off[f], sz[f]
    sub = df[o:o + m]
    r = sub.row(0, named=True)
    print(f"\n### feed {r['fidx']} M={m} H={r['H']} guest={r['guest']} hour={r['hour']} newest type/src/cat="
          f"{r['newest_type']}/{r['newest_src']}/{r['newest_cat']} hist_cats32={r['n_hist_cats32']} "
          f"feed_cats={r['n_feed_cats']} best_pos_rank={best[f]}")
    show = sub.filter((pl.col("rank") <= 5) | (pl.col("label") == 1)).sort("rank").select(cols)
    pdf = show.to_pandas()
    pdf.columns = [c.replace("it_hist_all_luo", "ithist").replace("xp_imp_c50", "xp").replace("cat_newest_idx", "cnew")
                   .replace("cat_user_rank", "cur").replace("cat_actions", "cact").replace("cat_views", "cview")
                   .replace("cand_maxcos", "cmax").replace("n_cand_dup95", "ndup").replace("seq_cv_invd20", "cv")
                   .replace("pre2_fe_z", "pre2").replace("hmax_idx", "hidx").replace("hmax_type", "htyp")
                   .replace("c_contact", "ccon").replace("c_vsearch", "cvs").replace("c_vfeed", "cvf") for c in pdf.columns]
    print(pdf.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    # category composition of feed: model's top-10 categories vs positives' categories
    top10 = sub.filter(pl.col("rank") <= 10)["cat"].to_list()
    print("top10 cats:", top10)
