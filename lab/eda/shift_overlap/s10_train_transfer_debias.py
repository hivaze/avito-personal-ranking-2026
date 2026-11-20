"""Does a train-fit model transfer to val? Effect of dropping f4 (sign-flipped feature) and of position-debiasing
(pos as a training feature, set to a constant at inference). Same 60k-feed train sample as s07 (Sep 17-19)."""
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
def X(d, cols, with_pos=None, ranks=True):
    out = {c: d['C'][c].astype(np.float32) for c in cols}
    if ranks:
        for c in cols: out[c + '_r'] = prank(d, d['C'][c].astype(np.float64)).astype(np.float32)
    out['cat'] = d['C']['category_id'].astype(np.int32)
    if with_pos is not None:
        out['pos'] = with_pos
    return pl.DataFrame(out).to_pandas()
PAR = dict(objective='binary', learning_rate=0.05, num_leaves=31, min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=4)
tv = V['C']['target'].astype(float); NZ = rng.random(len(tv)) * 1e-9
f1b = ndcg_per_feed(V['C']['f1'] + NZ, tv, offs(V['M']), V['M'])
print(f'val f1 alone (random tie-break): {f1b.mean():.4f}')
res = {}
def run(name, cols, pos_mode=None):
    if pos_mode is None:
        Xt = X(T, cols); Xv = X(V, cols)
    else:
        Xt = X(T, cols, with_pos=T['C']['pos']); Xv = X(V, cols, with_pos=np.full(len(tv), pos_mode, np.float32))
    m = lgb.train(PAR, lgb.Dataset(Xt, T['C']['target'], categorical_feature=['cat']), 300)
    p = m.predict(Xv)
    s = ndcg_per_feed(p + NZ, tv, offs(V['M']), V['M']); res[name] = s
    d = s - f1b
    print(f'  {name:55s} val NDCG {s.mean():.4f}  (vs f1 alone {d.mean():+.4f} +- {d.std()/np.sqrt(len(d)):.4f})', flush=True)
ALL = ['f1', 'f2', 'f3', 'f4', 'f5']
run('f1..f5 raw+rank + cat', ALL)
run('f1,f2,f3,f5 raw+rank + cat (drop f4)', ['f1', 'f2', 'f3', 'f5'])
run('f1..f5 raw+rank + cat + pos (infer pos=0)', ALL, 0.0)
run('f1..f5 raw+rank + cat + pos (infer pos=20)', ALL, 20.0)
run('f1,f2,f3,f5 + cat + pos (infer pos=0)', ['f1', 'f2', 'f3', 'f5'], 0.0)
# f4 relation after conditioning on pos: positive rate by f4 decile in train for pos<10 vs pos>=50
e = np.quantile(V['C']['f4'], np.linspace(0, 1, 11)[1:-1])
for lo, hi in [(0, 10), (10, 50), (50, 400)]:
    m = (T['C']['pos'] >= lo) & (T['C']['pos'] < hi)
    b = np.digitize(T['C']['f4'][m], e); t = T['C']['target'][m]
    r = np.bincount(b, weights=t, minlength=10) / np.maximum(np.bincount(b, minlength=10), 1) / t.mean()
    print(f'  train f4 decile pos-rate ratio, pos in [{lo},{hi}): {np.round(r,2).tolist()}')
# where do low-f4 items sit in production order?
b = np.digitize(T['C']['f4'], e)
print('  train mean pos by f4 decile:', [round(float(T['C']['pos'][b == k].mean()), 1) for k in range(10)])
print('  train share of candidates by f4 decile (val-defined edges):', np.round(np.bincount(b, minlength=10) / len(b) * 100, 1).tolist())
