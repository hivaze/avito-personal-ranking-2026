"""Diagnostic upper bound: cross-fitted LightGBM re-ranker of the head (top K), head set fixed.
If a flexible model cannot gain out of fold, simple head rules will not either."""
import hk, numpy as np, polars as pl, lightgbm as lgb, sys

c = hk.Ctx(); df = c.df
K = int(sys.argv[1]) if len(sys.argv) > 1 else 10
row = np.arange(len(df))
r = c.col("rank")
inh = r <= K
feat_item = ["ucos", "hmax", "hmax_idx", "hmax_type", "hmax_src", "h1", "h5", "h10", "h32", "hmean5", "n_hdup95",
             "c_contact", "c_order", "c_fav", "c_lead", "c_vfeed", "c_vsearch", "cat_views", "cat_actions",
             "cat_newest_idx", "cat_user_rank", "cat_hist_share", "cand_maxcos", "n_cand_dup95", "cat_n_in_feed",
             "cat_rank_in_feed", "f1", "f2", "f3", "f4", "f5", "cat", "it_hist_all_luo", "it_hist_et1_luo",
             "it_hist_et2_luo", "it_hist_et3_luo", "it_hist_et4_luo", "xp_imp_c50", "seq_cv_invd20", "pre2_fe_z",
             "pre2_new_z"]
feat_feed = ["M", "H", "guest", "hour", "newest_type", "newest_src", "n_hist_cats32", "n_feed_cats"]
feat_model = ["score", "s_ft", "s_to", "rank", "gap_next", "gap_top"]
X = df.select([pl.col(x).cast(pl.Float64) for x in feat_item + feat_feed + feat_model]).to_numpy()
dftto = (c.col("s_ft") - c.col("s_to")).astype(float)[:, None]
X = np.hstack([X, dftto])
names = feat_item + feat_feed + feat_model + ["d_ft_to"]
mode = sys.argv[2] if len(sys.argv) > 2 else "binary"
fold = c.fold_row
fid = c.feed_of_row
pred = np.full(len(df), np.nan)
imp = np.zeros(len(names))
for f in range(5):
    tr = inh & (fold != f); te = inh & (fold == f)
    if mode == "binary":
        params = dict(objective="binary", learning_rate=0.03, num_leaves=7, min_data_in_leaf=200, feature_fraction=0.7,
                      bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, num_threads=12, verbose=-1, seed=1)
        ds = lgb.Dataset(X[tr], c.y[tr])
    else:
        params = dict(objective="lambdarank", learning_rate=0.03, num_leaves=7, min_data_in_leaf=200, feature_fraction=0.7,
                      bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, num_threads=12, verbose=-1, seed=1,
                      lambdarank_truncation_level=K)
        _, cnt = np.unique(fid[tr], return_counts=True)
        ds = lgb.Dataset(X[tr], c.y[tr], group=cnt)
    for nr in [int(sys.argv[3]) if len(sys.argv) > 3 else 200]:
        m = lgb.train(params, ds, num_boost_round=nr)
    pred[te] = m.predict(X[te], raw_score=True)
    imp += m.feature_importance("gain")
new = c.score.copy()
new[inh] = 1000.0 + pred[inh] + 1e-6 * c.score[inh]
mm, se, d = c.delta(new)
print(f"K={K} mode={mode} OOF delta={mm:+.5f} se={se:.5f} z={mm/se:+.2f}")
for f in range(5):
    print("  fold", f, f"{d[c.fold_feed==f].mean():+.5f}")
o = np.argsort(-imp)[:20]
print("top importance:", [(names[i], round(imp[i] / imp.sum(), 3)) for i in o])
