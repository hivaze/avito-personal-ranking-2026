import hi, numpy as np, polars as pl
D = hi.Data(); df = D.df
E = hi.clogit_expect(np.c_[D.s], np.array([0.9825]), D.y, D.off, D.sz)
df = df.with_columns(pl.Series('E', E), (pl.col('cat_n_in_feed') / pl.col('M')).alias('cshare'))
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_cols(14); pl.Config.set_tbl_width_chars(200)
def b(c, edges):
    e = pl.col(c).cast(pl.Float64)
    out = pl.when(e.is_nan() | e.is_null()).then(pl.lit('nan'))
    for i in range(len(edges) - 1):
        out = out.when((e >= edges[i]) & (e < edges[i + 1])).then(pl.lit(f'{edges[i]:06.2f}'))
    return out.otherwise(pl.lit('zz'))
def piv(expr, name, by, frame=None):
    f = df if frame is None else frame
    t = (f.with_columns(expr.alias(name)).group_by(by, name)
         .agg(pl.col('label').sum().alias('pos'), pl.col('E').sum().alias('exp'))
         .with_columns(pl.format('{}/{}', ((pl.col('pos') / pl.col('exp')) * 100).round(0).cast(pl.Int32), pl.col('pos')).alias('v'))
         .pivot(on=name, index=by, values='v').sort(by))
    cols = [by] + sorted([c for c in t.columns if c != by])
    print(t.select(cols))
r = pl.col('rank')
df = df.with_columns(pl.when(r <= 3).then(pl.lit('a1-3')).when(r <= 10).then(pl.lit('b4-10')).when(r <= 30).then(pl.lit('c11-30')).otherwise(pl.lit('d31+')).alias('rb'))
print('cat share in feed (cat_n_in_feed/M), rows same_cat_newest')
piv(b('cshare', [0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.01]), 'cshare', 'same_cat_newest')
print('cat_rank_in_feed, rows same_cat_newest')
piv(b('cat_rank_in_feed', [1, 2, 3, 5, 10, 30, 1000]), 'crk', 'same_cat_newest')
print('rank bucket x cshare for same_cat_newest items')
piv(b('cshare', [0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.01]), 'cshare', 'rb', df.filter(pl.col('same_cat_newest')))
print('rank bucket x cshare for other items')
piv(b('cshare', [0, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 1.01]), 'cshare', 'rb', df.filter(~pl.col('same_cat_newest')))
