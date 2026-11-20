"""Upper-bound check: can a cross-fitted LightGBM lambdarank residual (init_score = score) on category-context
features improve NDCG out of fold? If not, simple category rules cannot either."""
import sys
import numpy as np
import polars as pl
import lightgbm as lgb
import cu

C = cu.Ctx()
df = C.df
c = lambda n: df[n].to_numpy().astype(np.float64)
cat = df["cat"].to_numpy()
vert_id = np.array([cu.VERT_ID[cu.VERT[x]] for x in cat.tolist()])
nvert = np.array([cu.VERT_ID.get(cu.VERT.get(x, ""), -1) for x in df["newest_cat"].to_list()])
featset = sys.argv[1] if len(sys.argv) > 1 else "cat"
F = {
    "cat": cat, "vert": vert_id, "cat_views": c("cat_views"), "cat_actions": c("cat_actions"),
    "cat_newest_idx": c("cat_newest_idx"), "cat_user_rank": c("cat_user_rank"), "cat_hist_share": c("cat_hist_share"),
    "same_cat_newest": c("same_cat_newest"), "n_hist_cats32": c("n_hist_cats32"), "newest_type": c("newest_type"),
    "newest_src": c("newest_src"), "newest_vert": nvert, "cat_n_in_feed": c("cat_n_in_feed"),
    "cat_share": c("cat_n_in_feed") / c("M"), "cat_rank_in_feed": c("cat_rank_in_feed"), "M": c("M"),
    "rank": c("rank"), "n_feed_cats": c("n_feed_cats"), "guest": c("guest"), "H": c("H"),
    "s_ft": c("s_ft"), "s_to": c("s_to"), "gap_top": c("gap_top"),
}
if featset == "all":
    for n in ["ucos", "hmax", "hmax_idx", "hmax_type", "h1", "h5", "h32", "f1", "f2", "f2_home", "f3", "f4", "f5",
              "xp_imp_c50", "it_hist_all_luo", "seq_cv_invd20", "pre2_fe_z", "pre2_new_z", "cand_maxcos",
              "n_cand_dup95", "c_contact", "c_fav", "c_vfeed", "c_vsearch", "n_hdup95"]:
        F[n] = c(n)
names = list(F)
X = np.column_stack([F[n] for n in names]).astype(np.float32)
y = C.y
s = C.s
fold_row = df["fold"].to_numpy()
sz = C.sz
ffold = C.ffold
for params in [dict(num_leaves=7, learning_rate=0.03, n_rounds=150, min_data_in_leaf=500, lambda_l2=10.0),
               dict(num_leaves=15, learning_rate=0.03, n_rounds=300, min_data_in_leaf=300, lambda_l2=10.0)]:
    nr = params.pop("n_rounds")
    preds = {k: np.zeros(len(y)) for k in (0.25, 0.5, 1.0)}
    for k in range(5):
        tr = fold_row != k
        te = ~tr
        dtr = lgb.Dataset(X[tr], y[tr], group=sz[ffold != k], init_score=s[tr] * 2.0,
                          categorical_feature=[names.index("cat"), names.index("vert"), names.index("newest_vert")],
                          free_raw_data=False)
        p = dict(objective="lambdarank", metric="ndcg", eval_at=[300], num_threads=12, verbose=-1,
                 feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, seed=k, **params)
        b = lgb.train(p, dtr, num_boost_round=nr)
        raw = b.predict(X[te], raw_score=True)
        for sh in preds:
            preds[sh][te] = s[te] + sh * raw / 2.0
    for sh, pr in preds.items():
        d = C.d(pr + 1e-9 * s)
        m, se, z = cu.delta_stats(d)
        print(f"{featset} {params} shrink {sh}: OOF delta {m:+.5f} se {se:.5f} z {z:+.2f}")
    params["n_rounds"] = nr
