"""Sweep of the LightGBM residual bound: rounds x init scale, category features vs all features."""
import sys
import numpy as np
import lightgbm as lgb
import cu
sys.argv = [sys.argv[0], sys.argv[1] if len(sys.argv) > 1 else "cat"]
exec(open("a13_lgb_bound.py").read().split("for params in")[0])
for init_scale in (1.0, 3.0):
    for nr in (25, 75, 200):
        preds = np.zeros(len(y))
        for k in range(5):
            tr = fold_row != k
            te = ~tr
            dtr = lgb.Dataset(X[tr], y[tr], group=sz[ffold != k], init_score=s[tr] * init_scale,
                              categorical_feature=[names.index("cat"), names.index("vert"), names.index("newest_vert")])
            p = dict(objective="lambdarank", num_threads=12, verbose=-1, feature_fraction=0.8, bagging_fraction=0.8,
                     bagging_freq=1, seed=k, num_leaves=7, learning_rate=0.03, min_data_in_leaf=1000, lambda_l2=30.0)
            b = lgb.train(p, dtr, num_boost_round=nr)
            preds[te] = s[te] + b.predict(X[te], raw_score=True) / init_scale
        m, se, z = cu.delta_stats(C.d(preds + 1e-9 * s))
        print(f"{featset} init {init_scale} rounds {nr}: OOF {m:+.5f} se {se:.5f} z {z:+.2f}", flush=True)
