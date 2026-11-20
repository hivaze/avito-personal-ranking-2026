"""LightGBM lambdarank lift of embedding features over f1..f5+category.
(a) train on 200K-feed train subsample -> eval val (exact metric); (b) 5-fold CV on val."""
import numpy as np
import lightgbm as lgb
import time
from common import ndcg_per_feed, feeds, arr

C = '<repo>/eda/embeddings/cache'
names = open(f'{C}/feat_names.txt').read().split('\n')
Ftr = np.load(f'{C}/train_feats.npy'); Btr = np.load(f'{C}/train_base.npy')
ytr = np.load(f'{C}/train_target.npy'); Mtr = np.load(f'{C}/train_M.npy')
Fva = np.load('<repo>/eda/embeddings/val_feats.npy')
fv = feeds('val'); Mva = fv['M'].to_numpy(); offva = fv['cand_off'].to_numpy()
Bva = np.stack([np.asarray(arr('val', f'cand_{c}')) for c in ['f1', 'f2', 'f3', 'f4', 'f5']] +
               [np.asarray(arr('val', 'cand_category_id')).astype(np.float32)], 1).astype(np.float32)
yva = np.asarray(arr('val', 'cand_target')).astype(np.float64)
bnames = ['f1', 'f2', 'f3', 'f4', 'f5', 'category_id']
rand_t = np.random.default_rng(0).random(len(yva)) * 1e-9

compact = ['user_dot', 'rec_hl5', 'last5_cos', 'max_sim', 'smax_t0.1', 'centrality_loo', 'hmean_cos', 'type1_max']
sets = {
    'base': [],
    'base+user_dot': ['user_dot'],
    'base+hist(rec_hl5,max_sim)': ['rec_hl5', 'max_sim'],
    'base+compact8': compact,
    'base+all_emb': [n for n in names if n not in ('item_norm', 'neg_user_l2', 'user_cos', 'cand_mean_cos_raw')],
    'emb_only(all)': None,
}
params = dict(objective='lambdarank', learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              lambdarank_truncation_level=300, eval_at=[300], num_threads=40, verbose=-1, seed=1)


def mk(F, B, sel):
    if sel is None:
        cols = [n for n in names if n not in ('item_norm', 'neg_user_l2', 'user_cos', 'cand_mean_cos_raw')]
        return F[:, [names.index(c) for c in cols]], cols, []
    X = np.concatenate([B, F[:, [names.index(c) for c in sel]]], 1)
    return X, bnames + sel, ['category_id']


res = {}
base_pf = None
t0 = time.time()
for nm, sel in sets.items():
    Xtr, cols, cat = mk(Ftr, Btr, sel)
    Xva, _, _ = mk(Fva, Bva, sel)
    dtr = lgb.Dataset(Xtr, ytr, group=Mtr, feature_name=cols, categorical_feature=cat, free_raw_data=False)
    bst = lgb.train(params, dtr, num_boost_round=1000)
    out = []
    for it in (200, 500, 1000):
        p = bst.predict(Xva, num_iteration=it, num_threads=40)
        pf = ndcg_per_feed(p + rand_t, yva, offva, Mva)
        out.append(pf)
    pf = out[1]
    if base_pf is None:
        base_pf = pf
    d = pf - base_pf
    res[nm] = pf
    imp = bst.feature_importance('gain'); top = np.argsort(-imp)[:10]
    print(f'[train-fit] {nm:28s} val NDCG@200/500/1000 it: {out[0].mean():.4f} {out[1].mean():.4f} {out[2].mean():.4f} | '
          f'vs base (500it) {d.mean():+.4f} (SE {d.std(ddof=1) / np.sqrt(len(d)):.4f})  [{time.time() - t0:.0f}s]', flush=True)
    print('    top gain:', ', '.join(f'{cols[i]}:{imp[i] / imp.sum():.3f}' for i in top), flush=True)

# 5-fold CV on val (unbiased traffic)
folds = np.random.default_rng(7).integers(0, 5, len(Mva))
cand_fold = np.repeat(folds, Mva)
pv = dict(params, learning_rate=0.03, num_leaves=31, min_data_in_leaf=100)
base_cv = None
for nm in ['base', 'base+user_dot', 'base+compact8', 'base+all_emb']:
    sel = sets[nm]
    X, cols, cat = mk(Fva, Bva, sel)
    oof = np.zeros(len(yva))
    for k in range(5):
        tr = cand_fold != k; te = ~tr
        dtr = lgb.Dataset(X[tr], yva[tr], group=Mva[folds != k], feature_name=cols, categorical_feature=cat)
        bst = lgb.train(pv, dtr, num_boost_round=400)
        oof[te] = bst.predict(X[te], num_threads=40)
    pf = ndcg_per_feed(oof + rand_t, yva, offva, Mva)
    if base_cv is None:
        base_cv = pf
    d = pf - base_cv
    print(f'[val-5fold] {nm:28s} OOF NDCG {pf.mean():.4f} vs base {d.mean():+.4f} (SE {d.std(ddof=1) / np.sqrt(len(d)):.4f})', flush=True)
print('done', time.time() - t0)
