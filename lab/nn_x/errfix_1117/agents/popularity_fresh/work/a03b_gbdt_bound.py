"""Bound probe v2: no score/rank as features; lambdarank or binary(init = calibrated logit). Cross-fitted."""
import sys
import lightgbm as lgb
from lib import *

D = Data()
xp = D.col("xp_imp_c50"); f4 = D.col("f4")
derived = {
    "f4_rel": f4 - D.feedwise(f4, np.median),
    "xp_rel": xp - D.feedwise(xp, np.mean),
    "cold_share": D.feedwise((xp == 0).astype(float), np.mean),
}
lens = ["xp_imp_c50", "it_hist_all_luo", "it_hist_et1_luo", "it_hist_et2_luo", "it_hist_et3_luo", "it_hist_et4_luo",
        "seq_cv_invd20", "f4", "f5", "pre2_fe_z", "pre2_new_z"]
seg = ["M", "H", "guest", "cat"]
sets = {"lens": lens, "lens+seg": lens + seg, "lens+seg+der": lens + seg + list(derived),
        "age_f5_cold+seg": ["f4", "f5", "xp_imp_c50", "M", "H", "guest", "cat"] + list(derived)}
obj, which = sys.argv[1], sys.argv[2]
feats = sets[which]
X = np.column_stack([derived[c] if c in derived else D.col(c) for c in feats]).astype(np.float32)
npos = np.add.reduceat(D.y, D.off)
p = np.clip(softmax_p(D.s, D.off, D.sz, npos, 1.0), 1e-5, 0.95)
init = np.log(p / (1 - p)) if obj == "binary" else D.s
params = dict(objective=obj, learning_rate=0.02, num_leaves=5, min_data_in_leaf=5000,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, num_threads=12,
              verbose=-1, seed=1)
if obj == "lambdarank":
    params.update(lambdarank_truncation_level=300, label_gain=[0, 1])
rounds = [25, 50, 100, 200, 400]
oof = {r: np.empty(len(D.s)) for r in rounds}
cf = [feats.index("cat")] if "cat" in feats else "auto"
for f in range(5):
    trr = D.fold_row != f; ter = ~trr
    kw = dict(group=D.sz[D.fold_feed != f]) if obj == "lambdarank" else {}
    ds = lgb.Dataset(X[trr], D.y[trr], init_score=init[trr], categorical_feature=cf, **kw)
    bst = lgb.train(params, ds, num_boost_round=max(rounds))
    for r in rounds:
        oof[r][ter] = D.s[ter] + bst.predict(X[ter], num_iteration=r, raw_score=True)
for r in rounds:
    m, se = D.delta(oof[r])
    print(f"{obj} {which} rounds {r}: OOF delta {m:+.5f} se {se:.5f} z {m / se:.2f}")
