"""5-fold val CV LightGBM: does a learned (train-fitted) history-attention score add over handcrafted embedding features?
Features: base (f1..f5,cat) [+ compact8 handcrafted] [+ attn_full score from 06c] [+ scaled bilinear score from 06d].
Repeated with 3 fold seeds to reduce CV noise."""
import numpy as np
import lightgbm as lgb
from common import ndcg_per_feed, feeds, arr

E = '<repo>/eda/embeddings'
names = open(f'{E}/val_feat_names.txt').read().split('\n')
F = np.load(f'{E}/val_feats.npy')
fv = feeds('val'); M = fv['M'].to_numpy(); off = fv['cand_off'].to_numpy()
y = np.asarray(arr('val', 'cand_target')).astype(np.float64)
B = np.stack([np.asarray(arr('val', f'cand_f{k}')) for k in range(1, 6)] + [np.asarray(arr('val', 'cand_category_id')).astype(np.float32)], 1)
attn = np.load(f'{E}/cache/val_pred_attn_full.npy')
bil = np.load(f'{E}/cache/val_pred_scaled_bilinear.npy')
compact = ['user_dot', 'rec_hl5', 'last5_cos', 'max_sim', 'smax_t0.1', 'centrality_loo', 'hmean_cos', 'type1_max']
C8 = F[:, [names.index(c) for c in compact]]
rt = np.random.default_rng(0).random(len(y)) * 1e-9
sets = {
    'base': (B, ['f1', 'f2', 'f3', 'f4', 'f5', 'category_id']),
    'base+compact8': (np.c_[B, C8], ['f1', 'f2', 'f3', 'f4', 'f5', 'category_id'] + compact),
    'base+attn': (np.c_[B, attn], ['f1', 'f2', 'f3', 'f4', 'f5', 'category_id', 'attn']),
    'base+compact8+attn': (np.c_[B, C8, attn], ['f1', 'f2', 'f3', 'f4', 'f5', 'category_id'] + compact + ['attn']),
    'base+compact8+attn+bilinear': (np.c_[B, C8, attn, bil], ['f1', 'f2', 'f3', 'f4', 'f5', 'category_id'] + compact + ['attn', 'bilinear']),
}
pv = dict(objective='lambdarank', learning_rate=0.03, num_leaves=31, min_data_in_leaf=100, feature_fraction=0.8,
          bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, lambdarank_truncation_level=300, num_threads=32, verbose=-1, seed=1)
res = {k: [] for k in sets}
for seed in (7, 8, 9):
    folds = np.random.default_rng(seed).integers(0, 5, len(M)); cf = np.repeat(folds, M)
    for nm, (X, cols) in sets.items():
        oof = np.zeros(len(y))
        for k in range(5):
            tr = cf != k
            d = lgb.Dataset(X[tr], y[tr], group=M[folds != k], feature_name=cols, categorical_feature=['category_id'])
            bst = lgb.train(pv, d, num_boost_round=400)
            oof[~tr] = bst.predict(X[~tr], num_threads=32)
        res[nm].append(ndcg_per_feed(oof + rt, y, off, M))
    print('seed', seed, ' '.join(f'{k}:{v[-1].mean():.4f}' for k, v in res.items()), flush=True)
base = np.mean(res['base'], 0)
print('\nmean over 3 CV seeds (per-feed averaged), paired vs base:')
for nm, v in res.items():
    pf = np.mean(v, 0); d = pf - base
    print(f'  {nm:30s} {pf.mean():.4f}  vs base {d.mean():+.4f} (SE {d.std(ddof=1) / np.sqrt(len(d)):.4f})')
pf_c8 = np.mean(res['base+compact8'], 0)
for nm in ['base+compact8+attn', 'base+compact8+attn+bilinear']:
    d = np.mean(res[nm], 0) - pf_c8
    print(f'  {nm} vs base+compact8: {d.mean():+.4f} (SE {d.std(ddof=1) / np.sqrt(len(d)):.4f})')
