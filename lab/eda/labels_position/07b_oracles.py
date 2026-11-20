"""Rough oracle bounds on val: knowing the positive's category, category+f1, number of positives, etc."""
from common import *
import json, datetime
nb.set_num_threads(16)
F = feeds('val'); off = F['cand_off'].to_numpy(); M = F['M'].to_numpy()
t = np.asarray(cand('val', 'target')); cat = np.asarray(cand('val', 'category_id')).astype(np.int64)
f1 = np.asarray(cand('val', 'f1')).astype(np.float64); f5 = np.asarray(cand('val', 'f5')).astype(np.float64)
fid = np.repeat(np.arange(len(M)), M)
df = pl.DataFrame({'fid': fid, 'cat': cat, 't': t})
poscat = df.filter(pl.col('t') == 1).select('fid', 'cat').unique().with_columns(pl.lit(1).alias('isposcat'))
df = df.join(poscat, on=['fid', 'cat'], how='left').with_columns(pl.col('isposcat').fill_null(0))
df = df.sort('fid', maintain_order=True) if False else df
# join may reorder -> recompute in original order
key = pl.DataFrame({'fid': fid, 'cat': cat, 'row': np.arange(len(t))})
ip = key.join(poscat, on=['fid', 'cat'], how='left').sort('row')['isposcat'].fill_null(0).to_numpy()
rng = np.random.default_rng(0)
res = {}
res['oracle_category_random_within'] = float(ndcg_pf(ip + 1e-3 * rng.random(len(t)), t, off, M).mean())
res['oracle_category_then_f1'] = float(ndcg_pf(ip + 0.5 * f1 + 1e-4 * f5, t, off, M).mean())
res['share_cands_in_pos_category'] = float(ip.mean())
res['f1_tiebreak_f5'] = float(ndcg_pf(f1 + 1e-3 * f5, t, off, M).mean())
# number of distinct categories per feed and size of positive category block
nc = df.group_by('fid').agg(pl.col('cat').n_unique().alias('nc'))['nc']
res['distinct_cats_per_feed_mean'] = float(nc.mean())
print(res)
T = feeds('test').sort(['timestamp', 'feed_id']); cut = int(0.3 * len(T))
ts = int(T['timestamp'][cut - 1])
res['public_cutoff_msk'] = str(datetime.datetime.utcfromtimestamp(ts + 3 * 3600))
res['test_range_msk'] = [str(datetime.datetime.utcfromtimestamp(int(T['timestamp'][0]) + 3 * 3600)), str(datetime.datetime.utcfromtimestamp(int(T['timestamp'][-1]) + 3 * 3600))]
V = feeds('val'); res['val_range_msk'] = [str(datetime.datetime.utcfromtimestamp(int(V['timestamp'].min()) + 3 * 3600)), str(datetime.datetime.utcfromtimestamp(int(V['timestamp'].max()) + 3 * 3600))]
TR = feeds('train'); res['train_range_msk'] = [str(datetime.datetime.utcfromtimestamp(int(TR['timestamp'].min()) + 3 * 3600)), str(datetime.datetime.utcfromtimestamp(int(TR['timestamp'].max()) + 3 * 3600))]
# val gap: find largest time gaps
for nm, X in [('val', V), ('train', TR)]:
    tsv = np.sort(X['timestamp'].to_numpy()); g = np.diff(tsv); ii = np.argsort(-g)[:3]
    res[f'{nm}_largest_gaps'] = [[str(datetime.datetime.utcfromtimestamp(int(tsv[i]) + 3 * 3600)), round(g[i] / 3600, 1)] for i in ii]
print(res)
json.dump(res, open(f'{OUT}/07b_oracles.json', 'w'), indent=1)
