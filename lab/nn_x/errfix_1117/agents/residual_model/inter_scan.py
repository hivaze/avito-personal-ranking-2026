"""Observed/expected positives for interactions of item category with feed context (hour, weekday, newest event,
feed size, history length). expected = per-feed softmax(c*score) * n_pos. Prints the cells with |z| >= 3."""
import numpy as np, polars as pl
import prep
from prep import kit
df = kit.load()
y = df['label'].to_numpy().astype(float); s = df['score'].to_numpy().astype(float)
off, sz = kit.feed_layout(df); fid = np.repeat(np.arange(len(sz)), sz)
c = prep.softmax_scale(s, y, off, sz)
e = np.exp(c*s - np.maximum.reduceat(c*s, off)[fid]); p = e/np.add.reduceat(e, off)[fid]
ex = p*np.add.reduceat(y, off)[fid]
df = df.with_columns(pl.Series('ex', ex), pl.Series('yf', y),
                     (pl.col('hour')//4).alias('hour4'), (pl.col('weekday')>=5).alias('wkend'),
                     pl.col('M').cut([30,100,200]).cast(pl.Utf8).alias('Mb'), pl.col('H').cut([10,50,200]).cast(pl.Utf8).alias('Hb'),
                     (pl.col('cat')==pl.col('newest_cat')).alias('cat_is_newest'),
                     pl.col('rank').cut([3,10,30,100]).cast(pl.Utf8).alias('rkb'))
tot = 0
for a in ['hour4','wkend','Mb','Hb','newest_type','newest_src','cat_is_newest','guest','rkb','n_hist_cats32']:
    t = df.group_by(['cat', a]).agg(pl.len().alias('n'), pl.col('yf').sum().alias('obs'), pl.col('ex').sum().alias('exp'))
    t = t.filter(pl.col('exp') >= 10).with_columns(((pl.col('obs')-pl.col('exp'))/pl.col('exp').sqrt()).alias('z'), (pl.col('obs')/pl.col('exp')).alias('ratio'))
    tot += t.height
    hit = t.filter(pl.col('z').abs() >= 2.8).sort('z')
    if hit.height: print(a, hit.to_pandas().to_string(index=False))
print('cells tested', tot)
