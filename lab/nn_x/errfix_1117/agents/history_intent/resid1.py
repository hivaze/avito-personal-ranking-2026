import hi, numpy as np, polars as pl
D = hi.Data(); df = D.df
E = hi.clogit_expect(np.c_[D.s], np.array([0.9825]), D.y, D.off, D.sz)
df = df.with_columns(pl.Series('E', E))
pl.Config.set_tbl_rows(40); pl.Config.set_tbl_cols(12)

def tab(expr, name, frame=None):
    f = df if frame is None else frame
    t = (f.with_columns(expr.alias(name)).group_by(name)
         .agg(pl.len().alias('n'), pl.col('label').sum().alias('pos'), pl.col('E').sum().alias('exp'),
              pl.col('rank').median().alias('med_rank'))
         .with_columns((pl.col('pos') / pl.col('exp')).alias('O/E'),
                       ((pl.col('pos') - pl.col('exp')) / pl.col('exp').sqrt()).alias('zres'))
         .sort(name))
    print(t)

def cut(c, edges):
    e = pl.col(c)
    out = pl.when(e.is_nan() | e.is_null()).then(pl.lit('nan'))
    for i in range(len(edges) - 1):
        out = out.when((e >= edges[i]) & (e < edges[i + 1])).then(pl.lit(f'{i:02d}[{edges[i]},{edges[i+1]})'))
    return out.otherwise(pl.lit('zz'))

edges = [-1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.01]
for c in ['h1', 'h5', 'h10', 'hmean5', 'hmax', 'c_contact', 'c_order', 'c_fav', 'c_lead', 'c_vfeed', 'c_vsearch']:
    print('====', c)
    tab(cut(c, edges), c)
print('==== cat_newest_idx')
cni = pl.col('cat_newest_idx')
tab(pl.when(cni < 0).then(pl.lit('-1')).when(cni == 0).then(pl.lit('0')).when(cni < 3).then(pl.lit('1-2'))
    .when(cni < 10).then(pl.lit('3-9')).when(cni < 32).then(pl.lit('10-31')).when(cni < 100).then(pl.lit('32-99')).otherwise(pl.lit('100+')), 'cni')
tab(pl.col('same_cat_newest'), 'same_cat_newest')
