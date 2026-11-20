"""Position-aware train model (pos as feature) - sweep of the constant pos used at inference on val. Same sample as s07/s10."""
from common import *
from featlib import subset
import lightgbm as lgb
rng = np.random.default_rng(7)
ftr = feeds('train'); dd = ftr['date'].cast(pl.Int32).to_numpy()
sel = rng.choice(np.where(dd >= dd.max() - 3)[0], 60000, replace=False)
T = subset('train', sel); V = subset('val')
f = T['f']; M = T['M']; co = f['cand_off'].to_numpy()
cidx = np.repeat(co, M) + (np.arange(M.sum()) - np.repeat(np.cumsum(M) - M, M))
T['C']['pos'] = np.asarray(cand('train', 'pos'))[cidx].astype(np.float32)
def offs(Mx): return np.r_[0, np.cumsum(Mx)[:-1]]
def prank(d, v):
    r = pl.DataFrame({'f': d['cf'], 'v': v}).with_columns(pl.col('v').rank('average').over('f').alias('r'))['r'].to_numpy() - 1
    return r / np.maximum(d['M'][d['cf']] - 1, 1)
ALL = ['f1', 'f2', 'f3', 'f4', 'f5']
def X(d, pos):
    out = {c: d['C'][c].astype(np.float32) for c in ALL}
    for c in ALL: out[c + '_r'] = prank(d, d['C'][c].astype(np.float64)).astype(np.float32)
    out['cat'] = d['C']['category_id'].astype(np.int32); out['pos'] = pos
    return pl.DataFrame(out).to_pandas()
PAR = dict(objective='binary', learning_rate=0.05, num_leaves=31, min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=4)
tv = V['C']['target'].astype(float); NZ = rng.random(len(tv)) * 1e-9
f1b = ndcg_per_feed(V['C']['f1'] + NZ, tv, offs(V['M']), V['M'])
m = lgb.train(PAR, lgb.Dataset(X(T, T['C']['pos']), T['C']['target'], categorical_feature=['cat']), 300)
Xv = X(V, np.zeros(len(tv), np.float32))
print(f'val f1 alone {f1b.mean():.4f}')
for c in [0, 5, 10, 20, 40, 80, 150, 250]:
    Xv['pos'] = np.float32(c)
    s = ndcg_per_feed(m.predict(Xv) + NZ, tv, offs(V['M']), V['M']); d = s - f1b
    print(f'  infer pos={c:3d}: val NDCG {s.mean():.4f} (vs f1 {d.mean():+.4f} +- {d.std()/np.sqrt(len(d)):.4f})', flush=True)
