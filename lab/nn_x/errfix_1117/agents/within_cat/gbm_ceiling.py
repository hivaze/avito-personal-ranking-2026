"""Cross-fitted LightGBM lambdarank with the model score as init_score: is there ANY learnable residual?"""
from wc import *
import lightgbm as lgb
df = load(); d = features(df)
d = rel(d, ["f1", "f4", "f5", "ucos", "hmax", "h5", "h10", "h32", "lhidx", "xp_imp_c50", "lpop", "lpop_et2", "ldup", "cand_maxcos", "pre2_fe_z", "lcv", "sdiff"])
d = d.with_columns(
    (pl.col("score") - pl.col("score").max().over(["fidx", "cat"])).alias("gap_catlead"),
    pl.col("score").mean().over(["fidx", "cat"]).alias("cat_meanscore"),
    pl.col("hmax_type").cast(pl.Float64), pl.col("hmax_src").cast(pl.Float64),
    pl.col("same_cat_newest").cast(pl.Float64), pl.col("f2_home").cast(pl.Float64),
)
feat_sets = {
    "within": [c for c in d.columns if c.endswith(("_dm", "_dmax", "_pr"))] + ["lcatn", "lcrank", "ccrank", "gap_catlead", "cat_n_in_feed"],
    "all": [c for c in d.columns if c.endswith(("_dm", "_dmax", "_pr"))] + BASEF + ["gap_catlead", "cat_meanscore", "hmax_type", "hmax_src", "same_cat_newest", "f2_home", "M", "H", "guest", "uvec_missing", "newest_type", "newest_src", "n_hist_cats32", "n_feed_cats", "cat_user_rank", "cat", "rank", "gap_top", "hour"],
}
off, sz = kit.feed_layout(d)
y = d["label"].to_numpy()
s = d["score"].to_numpy().astype(np.float64)
fo = kit.folds(d)
nb = kit.ndcg_per_feed(s, y, off, sz)
which = sys.argv[1]
cols = feat_sets[which]
X = d.select(cols).with_columns(pl.all().cast(pl.Float64)).to_numpy()
for lr, nt, nl in [(0.01, 30, 3), (0.01, 100, 3), (0.005, 100, 5)]:
    oof = np.zeros(len(y))
    for k in range(5):
        tr = fo != k; te = ~tr
        offtr, sztr = kit.feed_layout(d.filter(pl.Series(tr)))
        ds = lgb.Dataset(X[tr], y[tr], group=sztr, init_score=s[tr], free_raw_data=False)
        m = lgb.train(dict(objective="lambdarank", learning_rate=lr, num_leaves=nl, min_data_in_leaf=500, feature_fraction=0.7,
                           bagging_fraction=0.8, bagging_freq=1, lambda_l2=10, verbose=-1, num_threads=12, lambdarank_truncation_level=300,
                           seed=k), ds, num_boost_round=nt)
        oof[te] = s[te] + m.predict(X[te], num_threads=12)
    nn = kit.ndcg_per_feed(oof, y, off, sz)
    dd = nn - nb
    print(which, lr, nt, nl, "oof delta %.5f se %.5f z %.2f" % (dd.mean(), dd.std(ddof=1) / np.sqrt(len(dd)), dd.mean() / (dd.std(ddof=1) / np.sqrt(len(dd)))), flush=True)
