"""5-fold val CV (fold seeds given on argv, default 7,8) with XGBoost rank:ndcg on GPU (eta 0.1, 200 rounds): lift of embedding
feature sets over f1..f5+category. Usage: 05c_xgb_valcv.py "set1,set2" "7,8" ; saves per-feed OOF NDCG to cache/xgbcv_<set>.npy,
merge/report with 05d_merge_xgbcv.py. (LightGBM lambdarank with truncation 300 was too slow on the contended CPU.)"""
import os
import sys
import numpy as np
import xgboost as xgb
from common import ndcg_per_feed, feeds, arr

assert os.environ.get('CUDA_VISIBLE_DEVICES') in ('4', '5', '6', '7')
E = '<repo>/eda/embeddings'
names = open(f'{E}/val_feat_names.txt').read().split('\n')
F = np.load(f'{E}/val_feats.npy')
fv = feeds('val'); M = fv['M'].to_numpy(); off = fv['cand_off'].to_numpy()
y = np.asarray(arr('val', 'cand_target')).astype(np.float64)
cat = np.asarray(arr('val', 'cand_category_id')).astype(np.float32)
B = np.stack([np.asarray(arr('val', f'cand_f{k}')) for k in range(1, 6)] + [cat], 1).astype(np.float32)
attn = np.load(f'{E}/cache/val_pred_attn_full.npy').astype(np.float32)
bil = np.load(f'{E}/cache/val_pred_scaled_bilinear.npy').astype(np.float32)
compact = ['user_dot', 'rec_hl5', 'last5_cos', 'max_sim', 'smax_t0.1', 'centrality_loo', 'hmean_cos', 'type1_max']
allemb = [n for n in names if n not in ('item_norm', 'neg_user_l2', 'user_cos', 'cand_mean_cos_raw')]
col = lambda L: F[:, [names.index(c) for c in L]]
fid = np.repeat(np.arange(len(M)), M)


def within_feed_rank(X):
    """percentile rank within feed for each column (NaN kept)"""
    out = np.empty_like(X)
    tb = np.random.default_rng(0).random(len(fid))
    for j in range(X.shape[1]):
        s = np.where(np.isfinite(X[:, j]), X[:, j], -1e18)
        o = np.lexsort((tb, s, fid)); r = np.empty(len(s)); r[o] = np.arange(len(s))
        out[:, j] = (r - np.repeat(off, M)) / np.repeat(np.maximum(M - 1, 1), M)
        out[~np.isfinite(X[:, j]), j] = np.nan
    return out


C8 = col(compact)
sets = {
    'base': B,
    'base+user_dot': np.c_[B, col(['user_dot'])],
    'base+compact8': np.c_[B, C8],
    'base+compact8+ranks': np.c_[B, C8, within_feed_rank(C8)],
    'base+all_emb': np.c_[B, col(allemb)],
    'base+attn': np.c_[B, attn],
    'base+compact8+attn': np.c_[B, C8, attn],
    'base+compact8+attn+bilinear': np.c_[B, C8, attn, bil],
    'emb_only(all_emb)': col(allemb),
}
params = dict(objective='rank:ndcg', tree_method='hist', device='cuda', eta=0.1, max_depth=7, min_child_weight=20,
              subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, lambdarank_pair_method='mean',
              lambdarank_num_pair_per_sample=8, eval_metric='ndcg')
rt = np.random.default_rng(0).random(len(y)) * 1e-9
SEL = sys.argv[1].split(',') if len(sys.argv) > 1 else list(sets)
SEEDS = [int(x) for x in sys.argv[2].split(',')] if len(sys.argv) > 2 else [7, 8]
sets = {k: v for k, v in sets.items() if k in SEL}
res = {k: [] for k in sets}
for seed in SEEDS:
    folds = np.random.default_rng(seed).integers(0, 5, len(M)); cf = np.repeat(folds, M)
    for nm, X in sets.items():
        oof = np.zeros(len(y))
        for k in range(5):
            tr = cf != k
            d = xgb.DMatrix(X[tr], label=y[tr], qid=fid[tr])
            bst = xgb.train(dict(params, seed=seed * 10 + k), d, num_boost_round=200)
            oof[~tr] = bst.predict(xgb.DMatrix(X[~tr]))
        res[nm].append(ndcg_per_feed(oof + rt, y, off, M))
    print('seed', seed, ' | '.join(f'{k}:{v[-1].mean():.4f}' for k, v in res.items()), flush=True)
for k, v in res.items():
    np.save(f'{E}/cache/xgbcv_{k}.npy', np.array(v))
# reporting is done by 05d_merge_xgbcv.py
