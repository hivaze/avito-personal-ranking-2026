"""Blend of train-trained model and val-CV (OOF) model: within-feed rank average. Both are out-of-sample for every val feed."""
from xgbexp import *
dva = pl.read_parquet(f'{OUT}/feat_val.parquet', columns=['fid', 'target'])
gv = groups(dva); ov = offsets(gv); tv = dva['target'].to_numpy()
def load(fn, ck): return np.load(f'{OUT}/{fn}.npz')[str(ck)].astype(np.float64)
def rnk(s):
    return pl.DataFrame({'fid': dva['fid'], 's': s}).with_columns((pl.col('s').rank('average').over('fid') / pl.len().over('fid')).alias('r'))['r'].to_numpy()
A = rnk(load('pred_a_rich_rank', 200)); B = rnk(load('pred_b_rich_pos_rkend_rank__pos20_rkend50', 600)); C = rnk(load('pred_c_val_rich_binary', 200))
res = {}
def zs(s):
    return pl.DataFrame({'fid': dva['fid'], 's': s}).with_columns(((pl.col('s') - pl.col('s').mean().over('fid')) / (pl.col('s').std().over('fid') + 1e-9)).alias('z'))['z'].to_numpy()
Az = zs(load('pred_a_rich_rank', 200)); Bz = zs(load('pred_b_rich_pos_rkend_rank__pos20_rkend50', 600)); Cz = zs(np.log(load('pred_c_val_rich_binary', 200) + 1e-9))
cands = [('a_rich', A), ('b_rich_pos_rkend', B), ('c_val_binary', C),
         ('rankavg 0.5a+0.5c (ties!)', 0.5 * A + 0.5 * C), ('rankavg 0.5b+0.5c (ties!)', 0.5 * B + 0.5 * C),
         ('rankavg 0.5b+0.5c +1e-6*Bz tiebreak', 0.5 * B + 0.5 * C + 1e-6 * Bz),
         ('zavg 0.5a+0.5c', 0.5 * Az + 0.5 * Cz), ('zavg 0.5b+0.5c', 0.5 * Bz + 0.5 * Cz), ('zavg 0.7b+0.3c', 0.7 * Bz + 0.3 * Cz)]
for nm, s in cands:
    pf = ndcg_pf(s, tv, ov, gv); res[nm] = pf
    res[nm + ' [expected-ties]'] = ndcg_pf(s, tv, ov, gv, mode=2)
    print(nm, round(pf.mean(), 4), 'expected-ties', round(res[nm + ' [expected-ties]'].mean(), 4))
for a, b in [('zavg 0.5b+0.5c', 'b_rich_pos_rkend'), ('zavg 0.5a+0.5c', 'a_rich'), ('zavg 0.5b+0.5c', 'c_val_binary'), ('zavg 0.7b+0.3c', 'b_rich_pos_rkend')]:
    m, se = paired(res[a], res[b]); print(f'{a} - {b}: {m:+.4f} +- {se:.4f}')
json.dump({k: float(v.mean()) for k, v in res.items()}, open(f'{OUT}/10b_blend.json', 'w'), indent=1)
