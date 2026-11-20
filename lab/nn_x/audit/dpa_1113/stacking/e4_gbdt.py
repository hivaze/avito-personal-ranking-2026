"""E4 (b): LightGBM second level, residual on the pool (init_score = 2 x pool z), nested on the FT folds, 3 seeds,
fixed rounds (predictions saved at 100 and 300). Usage: e4_gbdt.py ARM FORM  (ARM B0..B3, FORM full|to).
full: val OOF vs G3 + test preds of a full-val fit. to: val OOF vs TO pool + holdout/test preds of a full-val fit."""
from lib import *
import time, json
import lightgbm as lgb
from feats import build

arm, form = sys.argv[1], sys.argv[2]
T0 = time.time()
V = S('val')
X, names, pool = build(V, arm, form)
y = V.y
RND = [100, 300]; C = 2.0
def params(seed):
    return dict(objective='lambdarank', lambdarank_truncation_level=300, learning_rate=0.03, num_leaves=15, min_data_in_leaf=1000,
                lambda_l2=100.0, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, seed=seed, num_threads=12,
                verbose=-1, metric='none', deterministic=True, force_row_wise=True)
oof = {(s, r): np.zeros(len(y)) for s in (1, 2, 3) for r in RND}
rows = lambda m: np.repeat(m, V.M)
for k in range(5):
    tr = V.fold != k; te = V.fold == k; rtr = rows(tr); rte = rows(te)
    ds = lgb.Dataset(X[rtr], y[rtr], group=V.M[tr], init_score=C * pool[rtr], feature_name=names, free_raw_data=False)
    for s in (1, 2, 3):
        b = lgb.train(params(s), ds, num_boost_round=max(RND))
        for r in RND:
            oof[(s, r)][rte] = C * pool[rte] + b.predict(X[rte], num_iteration=r)
    print(f'[{time.time()-T0:.0f}s] fold {k} done', flush=True)
tag = f'{arm}_{form}'
for (s, r), v in oof.items():
    np.save(CACHE / f'oof_gbdt_{tag}_s{s}_r{r}.npy', V.ev(v))
    np.save(CACHE / f'scr_gbdt_{tag}_s{s}_r{r}_val.npy', v.astype(np.float32))
# full-val fits -> holdout (to) and test
ds = lgb.Dataset(X, y, group=V.M, init_score=C * pool, feature_name=names, free_raw_data=False)
targets = [S('test')] + ([S('hold')] if form == 'to' else [])
imp = None
for s in (1, 2, 3):
    b = lgb.train(params(s), ds, num_boost_round=max(RND))
    g = b.feature_importance('gain'); imp = g if imp is None else imp + g
    for sp in targets:
        Xs, _, ps = build(sp, arm, form)
        for r in RND:
            sc = C * ps + b.predict(Xs, num_iteration=r)
            np.save(CACHE / f'scr_gbdt_{tag}_s{s}_r{r}_{sp.name}.npy', sc.astype(np.float32))
            if sp.name == 'hold':
                np.save(CACHE / f'hold_gbdt_{tag}_s{s}_r{r}.npy', sp.ev(sc))
order = np.argsort(-imp)
json.dump({names[i]: float(imp[i]) for i in order}, open(OUT / f'imp_gbdt_{tag}.json', 'w'), indent=1)
print(f'[{time.time()-T0:.0f}s] done {tag}; top gain:', [(names[i], round(float(imp[i] / imp.sum()), 3)) for i in order[:8]], flush=True)
