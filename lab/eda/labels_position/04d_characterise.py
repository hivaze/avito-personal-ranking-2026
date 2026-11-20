"""Characterise likely-random (mixture posterior) vs likely-production train feeds (04b held-out sample, M>=50)."""
from common import *
import json
nb.set_num_threads(40)
d = np.load(f'{OUT}/04b_detect.npz'); post = d['post']; sel = d['sel']; M = d['M']; z = d['z']
F = feeds('train')
Fs = F[sel].with_columns(pl.Series('post', post), pl.Series('z', z),
                         pl.from_epoch(pl.col('timestamp') + 3 * 3600, time_unit='s').dt.date().cast(pl.Utf8).alias('day'),
                         pl.from_epoch(pl.col('timestamp') + 3 * 3600, time_unit='s').dt.hour().alias('hour'))
dd, off, Mg = gather('train', sel, F, ['target', 'pos', 'f1', 'f3', 'f4', 'f5'])
t = dd['target']
npos = np.add.reduceat(t, off, dtype=np.int64)
Fs = Fs.with_columns(pl.Series('npos', npos))
nd = {'pos_asc': ndcg_pf(-dd['pos'].astype(float), t, off, Mg)}
for k in ['f1', 'f3', 'f4', 'f5']:
    nd[k] = ndcg_pf(dd[k], t, off, Mg, mode=2); nd[k + '_neg'] = ndcg_pf(-dd[k], t, off, Mg, mode=2)
nd['random'] = ndcg_pf(np.random.default_rng(0).random(len(t)), t, off, Mg)
# pos-0 hit: target of min-pos item
@nb.njit(cache=True)
def top_hit(pos, t, off, M):
    out = np.zeros(len(M))
    for f in range(len(M)):
        o = off[f]; m = M[f]; b = o
        for j in range(o, o + m):
            if pos[j] < pos[b]:
                b = j
        out[f] = t[b]
    return out
th = top_hit(dd['pos'].astype(np.int64), t, off, Mg)
mean_f = {k: np.add.reduceat(dd[k].astype(np.float64), off) / Mg for k in ['f1', 'f4', 'f5']}
Fs = Fs.with_columns(pl.Series('tophit', th), *[pl.Series('nd_' + k, v) for k, v in nd.items()], *[pl.Series('mean_' + k, v) for k, v in mean_f.items()])
res = {}
G = Fs.filter((pl.col('M') >= 50) & pl.col('post').is_not_null())
G = G.with_columns(pl.when(pl.col('post') > 0.7).then(pl.lit('random_like')).when(pl.col('post') < 0.05).then(pl.lit('prod_like')).otherwise(pl.lit('mid')).alias('grp'),
                   pl.col('M').cut([100, 200], left_closed=True).cast(pl.Utf8).alias('Mb'))
cols = [pl.len().alias('feeds'), pl.col('M').mean().alias('M'), (pl.col('npos').sum() / pl.col('M').sum()).alias('pos_rate'),
        pl.col('npos').mean().alias('npos'), (pl.col('tophit').mean() / (pl.col('npos') / pl.col('M')).mean()).alias('top_lift'),
        pl.col('H').mean().alias('H')] + [pl.col('nd_' + k).mean().alias('nd_' + k) for k in nd] + [pl.col('mean_' + k).mean().alias('mean_' + k) for k in mean_f]
tb = G.group_by(['Mb', 'grp']).agg(cols).sort(['Mb', 'grp'])
pl.Config.set_tbl_cols(30); pl.Config.set_tbl_width_chars(400)
print(tb)
res['groups'] = tb.to_dicts()
# soft-weighted (posterior) estimate for all M>=50
w = G['post'].to_numpy()
for nm, ww in [('random_w', w), ('prod_w', 1 - w)]:
    res[nm] = {c: float((G[c].to_numpy() * ww).sum() / ww.sum()) for c in ['M', 'H', 'tophit'] + ['nd_' + k for k in nd] + ['mean_' + k for k in mean_f]}
    res[nm]['pos_rate'] = float((G['npos'].to_numpy() * ww).sum() / (G['M'].to_numpy() * ww).sum())
    print(nm, {k: round(v, 4) for k, v in res[nm].items()})
# mean posterior by day / hour
byday = G.group_by('day').agg(pl.col('post').mean(), pl.len()).sort('day'); print(byday)
byhour = G.group_by('hour').agg(pl.col('post').mean(), pl.len()).sort('hour'); print(byhour.to_dicts())
res['post_by_day'] = byday.to_dicts(); res['post_by_hour'] = byhour.to_dicts()
json.dump(res, open(f'{OUT}/04d_characterise.json', 'w'), indent=1, default=str)
