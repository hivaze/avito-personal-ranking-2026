"""Label structure: positive rate, positives/feed, M distribution, rate vs M, H, hour, day. Train (full) vs val (test where label-free)."""
from common import *
import json, time

res = {}
for split in ['train', 'val', 'test']:
    t0 = time.time()
    F = feeds(split)
    off = F['cand_off'].to_numpy(); M = F['M'].to_numpy(); H = F['H'].to_numpy(); ts = F['timestamp'].to_numpy()
    d = F.with_columns(
        (pl.from_epoch(pl.col('timestamp') + 3 * 3600, time_unit='s')).alias('dt_msk'))
    d = d.with_columns(pl.col('dt_msk').dt.hour().alias('hour'), pl.col('dt_msk').dt.date().alias('day'),
                       pl.col('dt_msk').dt.weekday().alias('wd'))
    r = {'n_feeds': len(F), 'n_cands': int(M.sum()), 'M_mean': float(M.mean()),
         'M_q': {q: float(np.quantile(M, q)) for q in [0, .01, .1, .25, .5, .75, .9, .99, 1]},
         'H_mean': float(H.mean()), 'H_q': {q: float(np.quantile(H, q)) for q in [0, .1, .25, .5, .75, .9, 1]},
         'H_eq0': float((H == 0).mean()), 'H_eq256': float((H == 256).mean()),
         'M_ge299': float((M >= 299).mean()), 'M_eq300': float((M == 300).mean()), 'M_max': int(M.max()),
         'users': int(F['user_id'].n_unique()), 'feeds_per_user_mean': len(F) / F['user_id'].n_unique()}
    if split != 'test':
        t = cand(split, 'target')
        npos = np.add.reduceat(np.asarray(t), off, dtype=np.int64)
        d = d.with_columns(pl.Series('npos', npos))
        r['pos_rate'] = float(npos.sum() / M.sum())
        r['npos_mean'] = float(npos.mean())
        vc = np.bincount(np.minimum(npos, 10))
        r['npos_dist'] = {str(i) if i < 10 else '10+': float(vc[i] / len(npos)) for i in range(len(vc))}
        r['feed_rate_mean'] = float((npos / M).mean())
        # rate vs M bins
        mb = [5, 10, 20, 30, 50, 75, 100, 150, 200, 250, 299, 301]
        d = d.with_columns(pl.col('M').cut(mb[1:-1], left_closed=True).alias('Mbin'),
                           pl.col('H').cut([1, 16, 64, 128, 256], left_closed=True).alias('Hbin'))
        agg = lambda col: (d.group_by(col).agg(pl.len().alias('feeds'), (pl.len() / len(d)).alias('share'),
                                              pl.col('M').mean().alias('M_mean'), pl.col('npos').mean().alias('npos_mean'),
                                              (pl.col('npos').sum() / pl.col('M').sum()).alias('pos_rate'),
                                              (pl.col('npos') == 1).mean().alias('frac_1pos'), pl.col('H').mean().alias('H_mean'))
                           .sort(col))
        for col in ['Mbin', 'Hbin', 'hour', 'day']:
            tb = agg(col)
            print(split, col); print(tb)
            r[f'by_{col}'] = tb.with_columns(pl.col(col).cast(pl.Utf8)).to_dicts()
        # npos vs M correlation
        r['corr_npos_M'] = float(np.corrcoef(npos, M)[0, 1])
        r['corr_rate_M_spearman'] = float(pl.DataFrame({'a': npos / M, 'b': M}).select(pl.corr('a', 'b', method='spearman')).item())
    else:
        for col in ['hour', 'day']:
            tb = d.group_by(col).agg(pl.len().alias('feeds'), pl.col('M').mean().alias('M_mean')).sort(col)
            r[f'by_{col}'] = tb.with_columns(pl.col(col).cast(pl.Utf8)).to_dicts()
            print(split, col); print(tb)
    res[split] = r
    print(split, {k: v for k, v in r.items() if not k.startswith('by_')}, f'{time.time() - t0:.1f}s', flush=True)

# user overlap
u = {s: set(feeds(s)['user_id'].to_list()) for s in ['train', 'val', 'test']}
res['user_overlap'] = {'val_in_train': len(u['val'] & u['train']) / len(u['val']),
                       'test_in_train': len(u['test'] & u['train']) / len(u['test']),
                       'test_in_val': len(u['test'] & u['val']) / len(u['test'])}
print(res['user_overlap'])
json.dump(res, open(f'{OUT}/01_labels.json', 'w'), indent=1, default=str)
