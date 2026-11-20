import hi, numpy as np, polars as pl
D = hi.Data(); df = D.df
E = hi.clogit_expect(np.c_[D.s], np.array([0.9825]), D.y, D.off, D.sz)
df = df.with_columns(pl.Series('E', E))
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_cols(14); pl.Config.set_tbl_width_chars(200)
def b(c, edges):
    e = pl.col(c).cast(pl.Float64)
    out = pl.when(e.is_nan() | e.is_null()).then(pl.lit('nan'))
    for i in range(len(edges) - 1):
        out = out.when((e >= edges[i]) & (e < edges[i + 1])).then(pl.lit(f'{edges[i]:05.2f}'))
    return out.otherwise(pl.lit('zz'))
def piv(expr, name, by, frame=None):
    f = df if frame is None else frame
    t = (f.with_columns(expr.alias(name)).group_by(by, name)
         .agg(pl.col('label').sum().alias('pos'), pl.col('E').sum().alias('exp'))
         .with_columns(pl.format('{}/{}', ((pl.col('pos') / pl.col('exp')) * 100).round(0).cast(pl.Int32), pl.col('pos')).alias('v'))
         .pivot(on=name, index=by, values='v').sort(by))
    cols = [by] + sorted([c for c in t.columns if c != by])
    print(t.select(cols))
print('hmax bins x hmax_src')
piv(b('hmax', [-1, 0.7, 0.8, 0.9, 0.95, 0.99, 1.01]), 'hmax', 'hmax_src')
print('hmax bins x hmax_type')
piv(b('hmax', [-1, 0.7, 0.8, 0.9, 0.95, 0.99, 1.01]), 'hmax', 'hmax_type')
hi_ = pl.col('hmax_idx')
df = df.with_columns(pl.when(hi_ < 1).then(pl.lit('a0')).when(hi_ < 5).then(pl.lit('b1-4')).when(hi_ < 20).then(pl.lit('c5-19')).when(hi_ < 80).then(pl.lit('d20-79')).otherwise(pl.lit('e80+')).alias('hidx'))
print('for hmax>=0.95 (near-copy of a history item): hmax_idx x hmax_src')
piv(pl.col('hmax_src').cast(pl.Utf8), 'src', 'hidx', df.filter(pl.col('hmax') >= 0.95))
print('for hmax 0.85-0.95: hmax_idx x hmax_src')
piv(pl.col('hmax_src').cast(pl.Utf8), 'src', 'hidx', df.filter((pl.col('hmax') >= 0.85) & (pl.col('hmax') < 0.95)))
