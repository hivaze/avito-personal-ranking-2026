"""Target rate vs within-feed position rank for likely-random (p>0.9) vs likely-production (p<0.05) train feeds, M>=50.
In random feeds the curve is (examination x truncation) only; in production feeds it adds ranker relevance."""
from common import *
import json
nb.set_num_threads(16)
d = np.load(f'{OUT}/train_prandom.npz'); p = d['p_random']
F = feeds('train'); M = F['M'].to_numpy(); off = F['cand_off'].to_numpy()
res = {}
for nm, mask in [('random_like', (M >= 50) & (p > 0.9)), ('prod_like', (M >= 50) & (p < 0.05))]:
    sel = np.where(mask)[0]
    dd, o2, M2 = gather('train', sel, F, ['pos', 'target'])
    fid = np.repeat(np.arange(len(sel)), M2)
    df = pl.DataFrame({'fid': fid, 'pos': dd['pos'], 't': dd['target'], 'M': np.repeat(M2, M2)})
    df = df.with_columns((pl.col('pos').rank('ordinal').over('fid') - 1).alias('rk')).with_columns((pl.col('M') - 1 - pl.col('rk')).alias('rkend'))
    a = df.group_by('rk').agg(pl.col('t').mean().alias('rate'), pl.len()).sort('rk')
    b = df.group_by('rkend').agg(pl.col('t').mean().alias('rate'), pl.len()).sort('rkend')
    base = float(df['t'].mean())
    pick = [0, 1, 2, 3, 4, 5, 7, 10, 15, 20, 30, 50, 100, 150]
    ra = dict(zip(a['rk'].to_list(), a['rate'].to_list())); rb = dict(zip(b['rkend'].to_list(), b['rate'].to_list()))
    res[nm] = dict(n_feeds=len(sel), base_rate=base, rate_by_rank={k: ra[k] for k in pick if k in ra}, rate_by_rank_from_end={k: rb[k] for k in [0, 1, 2, 3, 4, 5, 10, 20] if k in rb},
                   ndcg_pos_asc=float(ndcg_pf(-dd['pos'].astype(float), dd['target'], o2, M2).mean()),
                   ndcg_random=float(ndcg_pf(np.random.default_rng(0).random(len(fid)), dd['target'], o2, M2).mean()))
    print(nm, json.dumps(res[nm]))
json.dump(res, open(f'{OUT}/04i_exam_curve.json', 'w'), indent=1)
