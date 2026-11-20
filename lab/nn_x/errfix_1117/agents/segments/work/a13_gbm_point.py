"""Headroom diagnostic 2: pointwise binary GBM on residuals, init = logit of a clogit-calibrated probability.
Cross-fitted by user folds; new score = init + gbm margin; NDCG delta vs model."""
import fx, kit, numpy as np, polars as pl, lightgbm as lgb, sys
df = kit.load()
c = fx.Ctx(df)
p = np.load("p_clogit.npy").clip(1e-5, 0.999)
init = np.log(p / (1 - p))
drop = {"label", "fidx", "user_id", "item_id", "row0", "row", "score", "s_ft", "s_to", "gap_next"}
feat_cols = [x for x in df.columns if x not in drop]
X = df.select(feat_cols).with_columns([pl.col(x).cast(pl.Float32) for x in feat_cols]).to_numpy()
y = c.y
nr = int(sys.argv[1]) if len(sys.argv) > 1 else 100
lr = float(sys.argv[2]) if len(sys.argv) > 2 else 0.02
leaves = int(sys.argv[3]) if len(sys.argv) > 3 else 7
mdl = int(sys.argv[4]) if len(sys.argv) > 4 else 2000
oof = {n: np.zeros(len(y)) for n in (nr // 4, nr // 2, nr)}
for k in range(5):
    tr = c.fold_row != k; te = ~tr
    dtr = lgb.Dataset(X[tr], y[tr], init_score=init[tr], feature_name=feat_cols)
    params = dict(objective="binary", learning_rate=lr, num_leaves=leaves, min_data_in_leaf=mdl, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=50, num_threads=12, verbose=-1)
    m = lgb.train(params, dtr, num_boost_round=nr)
    for n in oof:
        oof[n][te] = init[te] + m.predict(X[te], num_iteration=n, raw_score=True, num_threads=12)
    if k == 0:
        imp = sorted(zip(m.feature_importance("gain"), feat_cols), reverse=True)[:15]
        print([(n_, round(g)) for g, n_ in imp])
print("init-only", c.stats(init))
for n, o in oof.items():
    print("rounds", n, c.stats(o))
