import hi, numpy as np, polars as pl
D = hi.Data(); df = D.df
E = hi.clogit_expect(np.c_[D.s], np.array([0.9825]), D.y, D.off, D.sz)
df = df.with_columns(pl.Series('E', E))
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_cols(14); pl.Config.set_tbl_width_chars(200)
r = pl.col('rank')
rb = (pl.when(r == 1).then(pl.lit('a1')).when(r <= 3).then(pl.lit('b2-3')).when(r <= 10).then(pl.lit('c4-10'))
      .when(r <= 30).then(pl.lit('d11-30')).when(r <= 100).then(pl.lit('e31-100')).otherwise(pl.lit('f101+')))
df = df.with_columns(rb.alias('rb'))
def b(c, edges):
    e = pl.col(c)
    out = pl.when(e.is_nan() | e.is_null()).then(pl.lit('nan'))
    for i in range(len(edges) - 1):
        out = out.when((e >= edges[i]) & (e < edges[i + 1])).then(pl.lit(f'{edges[i]}'))
    return out.otherwise(pl.lit('zz'))
def piv(expr, name, by='rb', frame=None):
    f = df if frame is None else frame
    t = (f.with_columns(expr.alias(name)).group_by(by, name)
         .agg(pl.col('label').sum().alias('pos'), pl.col('E').sum().alias('exp'), pl.len().alias('n'))
         .with_columns(pl.format('{}/{}/{}', ((pl.col('pos') / pl.col('exp')) * 100).round(0).cast(pl.Int32), pl.col('pos'), (pl.col('pos')/pl.col('n')*1000).round(0).cast(pl.Int32)).alias('v'))
         .pivot(on=name, index=by, values='v').sort(by))
    cols = [by] + sorted([c for c in t.columns if c != by])
    print(t.select(cols))
print('cells: O/E*100 / positives / pos per mille; rows = model rank bucket')
for c in ['h1', 'h5', 'h10', 'hmean5', 'c_contact']:
    print('====', c)
    piv(b(c, [-1, 0.5, 0.7, 0.8, 0.9, 0.95, 1.01]), c)
