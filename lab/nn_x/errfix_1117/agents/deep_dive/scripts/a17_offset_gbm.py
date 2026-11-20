"""Residual boosting from the model: LightGBM lambdarank / binary with init_score = k*score (offset), few shallow trees.
If this cannot gain out-of-fold, simple feature rules are unlikely to gain either. Also shows which features carry residual signal."""
import sys
import lightgbm as lgb
from util import *

df = load()
c = Ctx(df)
s = c.base
y = c.y
# slope of within-feed softmax on score
from scipy.optimize import minimize_scalar


def nll(k):
    z = k * s
    mx = np.maximum.reduceat(z, c.off)
    lse = np.log(np.add.reduceat(np.exp(z - mx[c.fid]), c.off)) + mx
    return -(np.sum(y * z) - np.sum(c.npos * lse))


k = minimize_scalar(nll, bounds=(0.1, 5), method="bounded").x
print("softmax slope k =", k)


def zf(a):
    a = a.copy(); a[~np.isfinite(a)] = np.nanmean(a)
    m = np.bincount(c.fid, weights=a) / c.sz
    v = np.bincount(c.fid, weights=a * a) / c.sz - m * m
    return (a - m[c.fid]) / np.sqrt(np.maximum(v[c.fid], 1e-6))


drop = {"fidx", "user_id", "row0", "row", "label", "item_id", "score", "rank", "gap_next", "gap_top", "cat_rank_in_feed"}
feats = [x for x in df.columns if x not in drop]
X = df.select([pl.col(x).cast(pl.Float32) for x in feats]).to_numpy().astype(np.float64)
# add within-feed z versions of item features
item_feats = ["f1", "f2", "f3", "f4", "f5", "ucos", "hmax", "h1", "h5", "h10", "h32", "hmean5", "xp_imp_c50", "it_hist_all_luo",
              "pre2_fe_z", "c_contact", "c_vfeed", "cand_maxcos", "cat_views", "cat_hist_share"]
Z = np.column_stack([zf(df[x].cast(pl.Float64).to_numpy()) for x in item_feats])
X = np.column_stack([X, Z])
names = feats + [f"z_{x}" for x in item_feats]
fold = c.folds_row
mode = sys.argv[1] if len(sys.argv) > 1 else "binary"
for rounds, lr, leaves, mdl in [(50, 0.02, 7, 2000), (100, 0.02, 7, 2000), (200, 0.02, 15, 1000)]:
    pred = np.zeros(len(y))
    imp = np.zeros(len(names))
    for f in range(5):
        tr, te = fold != f, fold == f
        params = dict(objective="lambdarank" if mode == "rank" else "binary", learning_rate=lr, num_leaves=leaves,
                      min_data_in_leaf=mdl, feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=50,
                      num_threads=12, verbose=-1, seed=f)
        if mode == "rank":
            ftr = c.fid[tr]
            grp = np.bincount(ftr)[np.unique(ftr)]
            ds = lgb.Dataset(X[tr], y[tr], group=grp, init_score=k * s[tr], feature_name=names)
            params["lambdarank_truncation_level"] = 300
        else:
            # binary with offset; centre offset per feed so the intercept is learnable
            ds = lgb.Dataset(X[tr], y[tr], init_score=k * s[tr] - 4.0, feature_name=names)
        m = lgb.train(params, ds, rounds)
        pred[te] = m.predict(X[te], raw_score=True)
        imp += m.feature_importance("gain")
    new = k * s + pred
    dm, se = c.delta(new + 1e-9 * s)
    print(f"{mode} rounds={rounds} lr={lr} leaves={leaves} mdl={mdl}: OOF delta {dm:+.5f} se {se:.5f} z {dm / se:+.2f}", flush=True)
    top = np.argsort(-imp)[:15]
    print("   top residual features:", ", ".join(f"{names[i]}({imp[i]:.0f})" for i in top))
