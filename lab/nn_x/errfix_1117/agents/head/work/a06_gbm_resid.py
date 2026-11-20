"""Residual GBM in the head: init_score from a per-fold logistic of label on (score, rank-bucket), then a small
boosted correction. Cross-fitted. Diagnostic upper bound only."""
import hk, numpy as np, polars as pl, lightgbm as lgb, sys
from sklearn.linear_model import LogisticRegression
c = hk.Ctx(); df = c.df
K = int(sys.argv[1]); NR = int(sys.argv[2]); LR = float(sys.argv[3])
r = c.col("rank"); inh = r <= K
feat_item = ["ucos","hmax","hmax_idx","hmax_type","hmax_src","h1","h5","h10","h32","hmean5","n_hdup95","c_contact","c_order","c_fav","c_lead","c_vfeed","c_vsearch","cat_views","cat_actions","cat_newest_idx","cat_user_rank","cat_hist_share","cand_maxcos","n_cand_dup95","cat_n_in_feed","cat_rank_in_feed","f1","f2","f3","f4","f5","cat","it_hist_all_luo","it_hist_et1_luo","it_hist_et2_luo","it_hist_et3_luo","it_hist_et4_luo","xp_imp_c50","seq_cv_invd20","pre2_fe_z","pre2_new_z"]
feat_feed = ["M","H","guest","hour","newest_type","newest_src","n_hist_cats32","n_feed_cats"]
feat_model = ["score","s_ft","s_to","rank","gap_next","gap_top"]
X = df.select([pl.col(x).cast(pl.Float64) for x in feat_item+feat_feed+feat_model]).to_numpy()
X = np.hstack([X, (c.col("s_ft")-c.col("s_to")).astype(float)[:,None]])
names = feat_item+feat_feed+feat_model+["d_ft_to"]
# base: within-feed softmax-ish: use score and log(M)
B = np.column_stack([c.score, np.log(c.col("M")), np.log(r)])
fold = c.fold_row; pred = np.full(len(df), np.nan); imp=np.zeros(len(names))
for f in range(5):
    tr = inh & (fold != f); te = inh & (fold == f)
    lr = LogisticRegression(C=1e4, max_iter=1000).fit(B[tr], c.y[tr])
    i0tr = lr.decision_function(B[tr]); i0te = lr.decision_function(B[te])
    params = dict(objective="binary", learning_rate=LR, num_leaves=5, min_data_in_leaf=400, feature_fraction=0.5,
                  bagging_fraction=0.7, bagging_freq=1, lambda_l2=50.0, num_threads=12, verbose=-1, seed=2)
    m = lgb.train(params, lgb.Dataset(X[tr], c.y[tr], init_score=i0tr), num_boost_round=NR)
    pred[te] = i0te + m.predict(X[te], raw_score=True)
    imp += m.feature_importance("gain")
new = c.score.copy(); new[inh] = 1000.0 + pred[inh] + 1e-6*c.score[inh]
mm, se, d = c.delta(new)
print(f"K={K} NR={NR} LR={LR} OOF delta={mm:+.5f} se={se:.5f} z={mm/se:+.2f} folds", [round(d[c.fold_feed==f].mean()*1e4,1) for f in range(5)])
o=np.argsort(-imp)[:15]; print("  imp:", [(names[i], round(imp[i]/imp.sum(),3)) for i in o])
