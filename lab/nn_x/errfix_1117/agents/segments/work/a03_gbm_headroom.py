"""Diagnostic: how much does a cross-fitted LightGBM residual ranker (score as init) gain? Upper-ish bound."""
import fx, kit, numpy as np, polars as pl, lightgbm as lgb, sys
df = kit.load()
c = fx.Ctx(df)
feat_cols = [x for x in df.columns if x not in ("label", "fidx", "user_id", "item_id", "row0", "row", "score", "s_ft", "s_to", "rank", "gap_top", "gap_next", "cat_rank_in_feed")]
X = df.select(feat_cols).with_columns([pl.col(x).cast(pl.Float32) for x in feat_cols]).to_numpy()
y = c.y
init_mult = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0
oof = np.zeros(len(y))
for k in range(5):
    trf = c.fold_row != k
    tef = ~trf
    # groups
    def grp(mask):
        fs = c.feed[mask]
        return np.bincount(fs)[np.unique(fs)]
    dtr = lgb.Dataset(X[trf], y[trf], group=grp(trf), init_score=init_mult * c.base[trf], feature_name=feat_cols)
    params = dict(objective="lambdarank", learning_rate=0.03, num_leaves=15, min_data_in_leaf=500,
                  feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10,
                  num_threads=12, verbose=-1, lambdarank_truncation_level=300, label_gain=[0, 1])
    m = lgb.train(params, dtr, num_boost_round=int(sys.argv[2]) if len(sys.argv) > 2 else 150)
    oof[tef] = init_mult * c.base[tef] + m.predict(X[tef], num_threads=12)
    if k == 0:
        imp = sorted(zip(m.feature_importance("gain"), feat_cols), reverse=True)[:20]
        print([(n, round(g)) for g, n in imp])
print("init_mult", init_mult, c.stats(oof))
