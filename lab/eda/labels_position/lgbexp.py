"""LightGBM helpers for the train-vs-val experiments."""
from common import *
import lightgbm as lgb, time, json

BASIC = ['f1', 'f2', 'f3', 'f4', 'f5', 'category_id']
RICH = BASIC + ['dot', 'cos', 'inorm', 'hcat_cnt', 'hcat_share', 'hcat_recent', 'hcat_cnt_ne1', 'hsim_max', 'hsim_mean', 'hsim_max_ne1',
                'f1_r', 'f3_r', 'f4_r', 'f5_r', 'dot_r', 'M', 'H']

def groups(df):
    return df.group_by('fid', maintain_order=True).len()['len'].to_numpy()

def offsets(gr):
    return np.concatenate([[0], np.cumsum(gr)[:-1]]).astype(np.int64)

def params_for(kind='rank', small=False, trunc=300, lr=0.05, threads=40):
    p = dict(objective='lambdarank' if kind == 'rank' else 'binary', learning_rate=lr,
             num_leaves=31 if small else 63, min_data_in_leaf=200 if small else 500, feature_fraction=0.8,
             bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, num_threads=threads, verbose=-1, seed=0,
             max_cat_to_onehot=4, cat_smooth=10, max_bin=255)
    if kind == 'rank':
        p.update(lambdarank_truncation_level=trunc, label_gain=[0, 1])
    return p

def fit(dtr, feats, params, rounds, weight=None):
    X = dtr.select(feats).to_numpy().astype(np.float32)
    ds = lgb.Dataset(X, dtr['target'].to_numpy(), group=groups(dtr) if params['objective'] == 'lambdarank' else None,
                     weight=weight, feature_name=feats, categorical_feature=['category_id'] if 'category_id' in feats else 'auto', free_raw_data=True)
    t0 = time.time()
    bst = lgb.train(params, ds, num_boost_round=rounds)
    return bst, time.time() - t0

def evaluate(bst, dva, feats, checkpoints, override=None):
    """NDCG (pessimistic, exact metric) on dva at each checkpoint; override: dict col->constant"""
    d = dva
    if override:
        d = d.with_columns([pl.lit(v).cast(pl.Float32).alias(k) for k, v in override.items()])
    X = d.select(feats).to_numpy().astype(np.float32)
    gr = groups(dva); off = offsets(gr); t = dva['target'].to_numpy()
    out = {}; preds = {}
    for c in checkpoints:
        s = bst.predict(X, num_iteration=c, num_threads=40)
        out[c] = float(ndcg_pf(s, t, off, gr).mean()); preds[c] = s.astype(np.float32)
    evaluate.last_preds = preds
    return out, s

def ndcg_detail(s, dva):
    gr = groups(dva); off = offsets(gr); t = dva['target'].to_numpy()
    pess = ndcg_pf(s, t, off, gr); exp_ = ndcg_pf(s, t, off, gr, mode=2)
    s32 = s.astype(np.float32).astype(np.float64)
    p32 = ndcg_pf(s32, t, off, gr)
    return dict(pess=float(pess.mean()), expected=float(exp_.mean()), float32_pess=float(p32.mean()),
                at30=float(ndcg_pf(s, t, off, gr, k=30).mean())), pess
