"""Diagnostic: how much headroom does a cross-fitted LightGBM stacker on all columns find?
Also produces calibrated expectation E (from model-output-only features) for O/E analysis."""
import lightgbm as lgb
from util import *

df = load()
c = Ctx(df)
fl = pl.read_parquet("<repo>/lab/nn_x/errfix_1117/agents/deep_dive/scripts/feedlevel.parquet")
df = df.with_columns((pl.col("rank") / pl.col("M")).alias("rrel"), pl.col("M").log().alias("logM"),
                     (pl.col("s_ft") - pl.col("s_to")).alias("sdiff"))
OUTF = ["score", "gap_top", "rank", "rrel", "logM", "gap_next"]
drop = {"fidx", "user_id", "row0", "row", "label", "item_id"}
ALLF = [x for x in df.columns if x not in drop]
X_all = df.select([pl.col(x).cast(pl.Float32) for x in ALLF]).to_numpy()
X_out = df.select([pl.col(x).cast(pl.Float32) for x in OUTF]).to_numpy()
y = c.y
fold = c.folds_row


def cv(X, names, obj="binary", rounds=300, lr=0.05, leaves=31):
    pred = np.zeros(len(y))
    for f in range(5):
        tr, te = fold != f, fold == f
        params = dict(objective=obj, learning_rate=lr, num_leaves=leaves, min_data_in_leaf=200,
                      feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10,
                      num_threads=12, verbose=-1)
        if obj == "lambdarank":
            # group sizes
            ftr = c.fid[tr]
            grp = np.bincount(ftr)[np.unique(ftr)]
            ds = lgb.Dataset(X[tr], y[tr], group=grp, feature_name=names)
            params["lambdarank_truncation_level"] = 300
        else:
            ds = lgb.Dataset(X[tr], y[tr], feature_name=names)
        m = lgb.train(params, ds, rounds)
        pred[te] = m.predict(X[te], raw_score=True)
    return pred, m


p_out, _ = cv(X_out, OUTF)
print("calib-only binary model ndcg delta", c.delta(p_out + 1e-6 * c.base))
np.save("<repo>/lab/nn_x/errfix_1117/agents/deep_dive/scripts/E_calib.npy", 1 / (1 + np.exp(-p_out)))
p_all, m = cv(X_all, ALLF)
print("all-feature binary stacker ndcg delta", c.delta(p_all + 1e-6 * c.base))
imp = sorted(zip(m.feature_importance("gain"), ALLF), reverse=True)[:30]
for g, n in imp:
    print(f"  {n:20s} {g:.0f}")
p_rk, m2 = cv(X_all, ALLF, obj="lambdarank", rounds=300, lr=0.03)
print("all-feature lambdarank stacker ndcg delta", c.delta(p_rk + 1e-6 * c.base))
np.save("<repo>/lab/nn_x/errfix_1117/agents/deep_dive/scripts/p_stack_bin.npy", p_all)
np.save("<repo>/lab/nn_x/errfix_1117/agents/deep_dive/scripts/p_stack_rk.npy", p_rk)
