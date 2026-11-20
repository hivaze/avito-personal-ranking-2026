"""e07b: category profile of each event type for anonymous (user_id 0) vs logged-in users (val+test+train200K):
lift = share of the type's events in category c / share of type-1 events in category c (same user group)."""
import sys, os
import numpy as np, polars as pl
sys.path.insert(0, os.path.dirname(__file__))
from common_ev import hist_frame
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(250); pl.Config.set_tbl_cols(-1); pl.Config.set_float_precision(3)
OUT = '<repo>/nn_x/codes/events/out'
samp = np.load(f'{OUT}/train_sample200k_fidx.npy')
h = pl.concat([hist_frame('val')[0], hist_frame('test')[0], hist_frame('train', samp)[0]]).with_columns((pl.col('user_id') == 0).alias('anon'))
for anon in (True, False):
    d = h.filter(pl.col('anon') == anon)
    base = d.filter(pl.col('et') == 1).group_by('cat').agg(pl.len().alias('n1')).with_columns((pl.col('n1') / pl.col('n1').sum()).alias('s1'))
    rows = []
    for t in [3, 4, 6, 7, 10] if anon else [2, 3, 4, 7, 9, 10]:
        x = d.filter(pl.col('et') == t).group_by('cat').agg(pl.len().alias('n')).with_columns((pl.col('n') / pl.col('n').sum()).alias('s'))
        x = x.join(base, on='cat').with_columns((pl.col('s') / pl.col('s1')).alias('lift')).sort('n', descending=True).head(8)
        rows.append(f"type {t} (n={d.filter(pl.col('et') == t).height}): " + ', '.join(f"c{c}:{s:.2f}(x{l:.1f})" for c, s, l in zip(x['cat'], x['s'], x['lift'])))
    print(f'[K] anon={anon}: top categories per type, share (lift vs type-1 share in the same user group)')
    print('\n'.join(rows))
