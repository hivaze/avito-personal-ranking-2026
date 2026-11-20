"""Is the best item of each category over-predicted once the score->probability curve is fitted flexibly?
Base: per-feed softmax of a flexible function of (score, log rank, rank_pct, gap_top); obs/exp by cat_rank_in_feed
within global-rank bins."""
import numpy as np, polars as pl
from scipy.optimize import minimize
import prep
from prep import kit
df = prep.derive(kit.load())
y = df['label'].to_numpy().astype(float); s = df['score'].to_numpy().astype(float)
off, sz = kit.feed_layout(df); fid = np.repeat(np.arange(len(sz)), sz); npos = np.add.reduceat(y, off)
rk = df['rank'].to_numpy().astype(float)
B = np.column_stack([s, np.log(rk), df['rank_pct'].to_numpy(), df['gap_top'].to_numpy(), s**2, (rk<=1), (rk<=2), (rk<=3), (rk<=5)]).astype(float)
def obj(w):
    z = B @ w; mx = np.maximum.reduceat(z, off); e = np.exp(z - mx[fid]); se = np.add.reduceat(e, off)
    p = e / se[fid]
    return -((y * z).sum() - (npos * (np.log(se) + mx)).sum()), -(B.T @ (y - npos[fid] * p))
w = minimize(obj, np.r_[1, np.zeros(B.shape[1]-1)], jac=True, method='L-BFGS-B').x
z = B @ w; e = np.exp(z - np.maximum.reduceat(z, off)[fid]); p = e / np.add.reduceat(e, off)[fid]; ex = p * npos[fid]
df = df.with_columns(pl.Series('ex', ex), pl.Series('yf', y))
print('weights', np.round(w, 3))
for c in ['cat_rank_in_feed', 'n_cand_dup95', 'cat_user_rank']:
    t = df.with_columns(pl.col('rank').cut([1, 3, 10, 30]).cast(pl.Utf8).alias('rkb'), pl.col(c).clip(0, 5).alias('v')).group_by(['rkb', 'v']).agg(
        pl.len().alias('n'), pl.col('yf').sum().alias('obs'), pl.col('ex').sum().alias('exp')).with_columns(
        (pl.col('obs') / pl.col('exp')).alias('ratio'), ((pl.col('obs') - pl.col('exp')) / pl.col('exp').sqrt()).alias('z')).sort(['rkb', 'v'])
    print(c); print(t.filter(pl.col('exp') > 15).to_pandas().to_string(index=False, float_format=lambda v: f'{v:.2f}'))
