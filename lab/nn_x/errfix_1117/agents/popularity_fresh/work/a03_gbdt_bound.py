"""Upper-bound probe: LightGBM lambdarank on top of the model score (init_score), lens features only,
cross-fitted by user folds. If this cannot gain, simple rules in this lens cannot either."""
import sys

import lightgbm as lgb

from lib import *

D = Data()
df = D.df


def within(x, fn):
    return D.feedwise(x, fn)


xp = D.col("xp_imp_c50")
f4 = D.col("f4")
f5 = D.col("f5")
derived = {
    "f4_rel": f4 - within(f4, np.median),
    "xp_rel": xp - within(xp, np.mean),
    "cold_share": within((xp == 0).astype(float), np.mean),
    "f4_rank": D.within_rank(-f4) / D.col("M"),
    "s_gap": D.col("s_ft") - D.col("s_to"),
}
lens = ["xp_imp_c50", "it_hist_all_luo", "it_hist_et1_luo", "it_hist_et2_luo", "it_hist_et3_luo", "it_hist_et4_luo",
        "seq_cv_invd20", "f4", "f5", "pre2_fe_z", "pre2_new_z"]
ctx = ["rank", "score", "M", "H", "guest", "uvec_missing", "cat", "newest_type", "newest_src", "hour"]
sets = {
    "lens+rank": lens + ["rank", "score", "M"],
    "lens+ctx": lens + ctx,
    "lens+ctx+derived": lens + ctx + list(derived),
}
which = sys.argv[1] if len(sys.argv) > 1 else "lens+ctx"
feats = sets[which]
X = np.column_stack([derived[c] if c in derived else D.col(c) for c in feats]).astype(np.float32)
params = dict(objective="lambdarank", learning_rate=0.03, num_leaves=7, min_data_in_leaf=3000,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, num_threads=12,
              verbose=-1, lambdarank_truncation_level=300, label_gain=[0, 1], seed=1)
rounds = [25, 50, 100, 200, 400]
oof = {r: np.empty(len(D.s)) for r in rounds}
for f in range(5):
    trr = D.fold_row != f
    ter = ~trr
    tr_sz = D.sz[D.fold_feed != f]
    ds = lgb.Dataset(X[trr], D.y[trr], group=tr_sz, init_score=D.s[trr], categorical_feature=[feats.index("cat")] if "cat" in feats else "auto")
    bst = lgb.train(params, ds, num_boost_round=max(rounds))
    for r in rounds:
        oof[r][ter] = D.s[ter] + bst.predict(X[ter], num_iteration=r)
for r in rounds:
    m, se = D.delta(oof[r])
    print(f"{which} rounds {r}: OOF delta {m:+.5f} se {se:.5f} z {m / se:.2f}")
bst = lgb.train(params, lgb.Dataset(X, D.y, group=D.sz, init_score=D.s), num_boost_round=200)
imp = bst.feature_importance("gain")
for i in np.argsort(-imp):
    print("  %-18s %.1f" % (feats[i], imp[i]))
