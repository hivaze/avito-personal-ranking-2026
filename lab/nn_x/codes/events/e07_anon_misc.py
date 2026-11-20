"""e07: (a) anonymous (user_id 0) vs logged-in: which types exist, type 10 context; (b) type mix of the newest events by
train day (logging changes?); (c) missing item embedding by type controlled for hpos (closed / sold listing proxy);
(d) user-level co-occurrence of types.  Val + test + the e03 200K train sample."""
import sys, os
import numpy as np, polars as pl
sys.path.insert(0, os.path.dirname(__file__))
from common_ev import hist_frame

pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(250); pl.Config.set_tbl_cols(-1); pl.Config.set_float_precision(4)
OUT = '<repo>/nn_x/codes/events/out'
samp = np.load(f'{OUT}/train_sample200k_fidx.npy')
vh, _ = hist_frame('val'); th, _ = hist_frame('test'); trh, _ = hist_frame('train', samp)
allh = pl.concat([vh.with_columns(pl.lit('val').alias('sp')), th.with_columns(pl.lit('test').alias('sp')), trh.with_columns(pl.lit('train').alias('sp'))])
allh = allh.with_columns((pl.col('user_id') == 0).alias('anon'))
# (a) type-10 context
it = allh.group_by(['sp', 'fidx', 'item_id']).agg(pl.col('et').unique().alias('ts'), pl.col('anon').first(),
                                                  pl.col('cat').first(), pl.col('es').filter(pl.col('et') == 1).first().alias('src_of_view'),
                                                  pl.col('hpos').filter(pl.col('et') == 10).min().alias('h10'),
                                                  pl.col('hpos').filter(pl.col('et').is_in([2, 5, 8, 9])).min().alias('h_login'))
it10 = it.filter(pl.col('ts').list.contains(10))
print('[A1] items with a type-10 event: anon vs logged-in; share with a view, with login-only types (2/5/8/9), with 3/4/7')
print(it10.group_by('anon').agg(pl.len(), pl.col('ts').list.contains(1).mean().alias('has1'),
                               pl.col('ts').list.eval(pl.element().is_in([2, 5, 8, 9])).list.any().mean().alias('has_2589'),
                               pl.col('ts').list.contains(3).mean().alias('has3'), pl.col('ts').list.contains(4).mean().alias('has4'),
                               pl.col('ts').list.contains(7).mean().alias('has7'),
                               (pl.col('h_login') < pl.col('h10')).mean().alias('login_type_newer_than_10')))
print('[A2] source of the type-1 view on items with type 10 vs all viewed items (anon / logged-in)')
v = it.filter(pl.col('ts').list.contains(1)).with_columns(pl.col('ts').list.contains(10).alias('has10'),
                                                          pl.when(pl.col('src_of_view') <= 4).then(pl.col('src_of_view')).otherwise(99).alias('s'))
print(v.group_by(['anon', 'has10', 's']).agg(pl.len()).with_columns((pl.col('len') / pl.col('len').sum().over(['anon', 'has10'])).alias('share'))
      .pivot(on='s', index=['anon', 'has10'], values='share').sort(['anon', 'has10']))
print('[A3] category distribution: type-10 events vs type-1 events (top 15 cats by type-10 count), all splits')
c10 = allh.filter(pl.col('et') == 10).group_by('cat').agg(pl.len().alias('n10'))
c1 = allh.filter(pl.col('et') == 1).group_by('cat').agg(pl.len().alias('n1'))
cc = c10.join(c1, on='cat').with_columns((pl.col('n10') / pl.col('n10').sum()).alias('share10'), (pl.col('n1') / pl.col('n1').sum()).alias('share1'))
print(cc.with_columns((pl.col('share10') / pl.col('share1')).alias('lift')).sort('n10', descending=True).head(15))
print('[A4] anon feeds: type mix per split; anon items: type sets top 15')
print(allh.filter(pl.col('anon')).group_by('et').agg(pl.len()).with_columns((pl.col('len') / pl.col('len').sum()).alias('share')).sort('et'))
an_it = it.filter(pl.col('anon')).with_columns(pl.col('ts').list.sort().cast(pl.List(pl.Utf8)).list.join(',').alias('k'))
print(an_it.group_by('k').agg(pl.len()).with_columns((pl.col('len') / an_it.height).alias('share')).sort('len', descending=True).head(15))
# (b) newest-event type mix by train day
tr = trh.filter(pl.col('hpos') < 20).with_columns(((pl.col('timestamp') - 1788652800) // 86400).alias('day'))
print('[B1] train sample: type mix of the 20 newest events by feed day (day 0 = Sep 6 2026 UTC)')
x = tr.group_by(['day', 'et']).agg(pl.len()).pivot(on='et', index='day', values='len').fill_null(0).sort('day')
cols = [c for c in x.columns if c != 'day']
x = x.with_columns(pl.sum_horizontal(cols).alias('tot'))
print(x.select('day', 'tot', *[(pl.col(c) / pl.col('tot')).round(4).alias('e' + c) for c in sorted(cols, key=int)]))
for nm, df in [('val', vh), ('test', th)]:
    y = df.filter(pl.col('hpos') < 20).group_by('et').agg(pl.len()).with_columns((pl.col('len') / pl.col('len').sum()).round(4).alias('share')).sort('et')
    print(f'   {nm} newest-20 mix:', dict(zip(y['et'].to_list(), y['share'].to_list())))
# (c) no embedding by type x hpos bin
print('[C1] share of events whose item has NO embedding, by type x hpos bin (64 wide), val+test')
vt = pl.concat([vh, th]).with_columns((pl.col('hpos') // 64).alias('hb'), (pl.col('emb_idx') < 0).alias('noemb'))
print(vt.group_by(['et', 'hb']).agg(pl.col('noemb').mean()).pivot(on='hb', index='et', values='noemb').sort('et'))
# (d) per-feed co-occurrence of types (phi correlation of presence), val+test
fe = pl.concat([vh, th]).group_by(['fidx', 'user_id', 'H']).agg(*[(pl.col('et') == t).any().alias(f'p{t}') for t in range(1, 11)])
fe = fe.filter(pl.col('user_id') != 0)
P = fe.select([f'p{t}' for t in range(1, 11)]).to_numpy().astype(float)
C = np.corrcoef(P.T)
print('[D1] presence correlation across logged-in feeds (val+test, by feed)')
print('      ' + ' '.join(f'{t:>6d}' for t in range(1, 11)))
for i in range(10):
    print(f'{i+1:>5d} ' + ' '.join(f'{C[i,j]:6.2f}' for j in range(10)))
