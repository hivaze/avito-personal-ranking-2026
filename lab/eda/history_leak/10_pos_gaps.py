"""10) Train 'pos' gaps = candidates removed because the user had already viewed them (README).
Quantify how many were removed per feed and whether removal correlates with history size."""
import numpy as np
import polars as pl
from common import feeds, arr
f = feeds('train')
rng = np.random.default_rng(0)
sel = np.sort(rng.choice(len(f), 300_000, replace=False))
F = f.sort('fidx')[sel]
coff = F['cand_off'].to_numpy(); M = F['M'].to_numpy()
pos = arr('train', 'cand_pos'); tg = arr('train', 'cand_target')
mx = np.array([pos[o:o + m].max() for o, m in zip(coff, M)])
mn = np.array([pos[o:o + m].min() for o, m in zip(coff, M)])
missing = mx + 1 - M
df = pl.DataFrame({'M': M, 'max_pos': mx, 'min_pos': mn, 'missing': missing, 'H': F['H'].to_numpy()})
print('feeds with gaps (max_pos+1 > M):', (missing > 0).mean(), ' mean missing', missing.mean(), ' median', np.median(missing))
print('min_pos==0 share', (mn == 0).mean())
print('missing share of original list (missing/(max_pos+1)) mean', (missing / (mx + 1)).mean())
print(df.with_columns(pl.col('H').cut([0, 50, 100, 200, 255]).alias('Hb')).group_by('Hb').agg(
    pl.len(), pl.col('missing').mean(), (pl.col('missing') / (pl.col('max_pos') + 1)).mean().alias('miss_share')).sort('Hb'))
# target rate by original position (position bias)
idx = np.concatenate([np.arange(o, o + m) for o, m in zip(coff, M)])
P = np.asarray(pos[idx]); T = np.asarray(tg[idx])
d = pl.DataFrame({'p': P, 't': T}).with_columns(pl.col('p').cut([0, 4, 9, 19, 49, 99, 199]).alias('pb'))
print(d.group_by('pb').agg(pl.len(), pl.col('t').mean()).sort('pb'))
