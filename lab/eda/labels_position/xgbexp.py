"""XGBoost (GPU) ranking helpers - used instead of LightGBM because the shared CPUs were oversubscribed (load avg >500)."""
from common import *
import xgboost as xgb, time, json

BASIC = ['f1', 'f2', 'f3', 'f4', 'f5', 'category_id']
RICH = BASIC + ['dot', 'cos', 'inorm', 'hcat_cnt', 'hcat_share', 'hcat_recent', 'hcat_cnt_ne1', 'hsim_max', 'hsim_mean', 'hsim_max_ne1',
                'f1_r', 'f3_r', 'f4_r', 'f5_r', 'dot_r', 'M', 'H']

def groups(df):
    return df.group_by('fid', maintain_order=True).len()['len'].to_numpy()

def offsets(gr):
    return np.concatenate([[0], np.cumsum(gr)[:-1]]).astype(np.int64)

def dmat(df, feats, label=True, group=True, gweight=None, override=None):
    if override:
        df = df.with_columns([pl.lit(v).cast(pl.Float32).alias(k) for k, v in override.items()])
    X = df.select(feats).to_numpy().astype(np.float32)
    ft = ['c' if f == 'category_id' else 'q' for f in feats]
    dm = xgb.DMatrix(X, label=df['target'].to_numpy() if label else None, feature_names=feats, feature_types=ft, enable_categorical=True, nthread=16)
    if group:
        dm.set_group(groups(df))
    if gweight is not None:
        dm.set_weight(gweight)
    return dm

def params(kind='rank', eta=0.05, depth=8, mcw=50):
    p = dict(device='cuda', tree_method='hist', eta=eta, max_depth=depth, min_child_weight=mcw, subsample=0.8, colsample_bytree=0.9,
             reg_lambda=10, max_bin=256, nthread=16, max_cat_to_onehot=1, seed=0)
    if kind == 'rank':
        p.update(objective='rank:ndcg', lambdarank_pair_method='topk', lambdarank_num_pair_per_sample=300, ndcg_exp_gain=False)
    else:
        p.update(objective='binary:logistic')
    return p

def evaluate(bst, dva, feats, checkpoints, override=None):
    dm = dmat(dva, feats, group=False, override=override)
    gr = groups(dva); off = offsets(gr); t = dva['target'].to_numpy()
    out = {}; preds = {}
    for c in checkpoints:
        s = bst.predict(dm, iteration_range=(0, c))
        out[c] = float(ndcg_pf(s, t, off, gr).mean()); preds[c] = s
    return out, preds

def ndcg_detail(s, dva):
    gr = groups(dva); off = offsets(gr); t = dva['target'].to_numpy()
    s = s.astype(np.float64)
    pess = ndcg_pf(s, t, off, gr)
    s32 = s.astype(np.float32).astype(np.float64)
    return dict(pess=float(pess.mean()), expected=float(ndcg_pf(s, t, off, gr, mode=2).mean()),
                float32_pess=float(ndcg_pf(s32, t, off, gr).mean()), at30=float(ndcg_pf(s, t, off, gr, k=30).mean()),
                se=float(pess.std() / np.sqrt(len(pess)))), pess
