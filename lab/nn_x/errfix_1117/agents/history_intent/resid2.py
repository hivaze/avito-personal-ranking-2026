import hi, numpy as np, polars as pl
D = hi.Data(); df = D.df
E = hi.clogit_expect(np.c_[D.s], np.array([0.9825]), D.y, D.off, D.sz)
nt, ns = pl.col('newest_type'), pl.col('newest_src')
ng = (pl.when(nt < 0).then(pl.lit('empty'))
      .when((nt == 1) & (ns == 1)).then(pl.lit('v_feed'))
      .when((nt == 1) & (ns == 2)).then(pl.lit('v_search'))
      .when((nt == 1) & (ns == 0)).then(pl.lit('v_src0'))
      .when(nt == 1).then(pl.lit('v_other'))
      .when(nt.is_in([2, 3])).then(pl.lit('contact'))
      .when(nt == 4).then(pl.lit('fav'))
      .when(nt.is_in([5, 6, 8])).then(pl.lit('order'))
      .when(nt == 7).then(pl.lit('lead'))
      .when(nt == 9).then(pl.lit('done'))
      .otherwise(pl.lit('guest')))
df = df.with_columns(pl.Series('E', E), ng.alias('ng'))
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_cols(14); pl.Config.set_tbl_width_chars(200)

def b(c, edges):
    e = pl.col(c)
    out = pl.when(e.is_nan() | e.is_null()).then(pl.lit('nan'))
    for i in range(len(edges) - 1):
        out = out.when((e >= edges[i]) & (e < edges[i + 1])).then(pl.lit(f'{edges[i]}'))
    return out.otherwise(pl.lit('zz'))

def piv(expr, name, by='ng', frame=None):
    f = df if frame is None else frame
    t = (f.with_columns(expr.alias(name)).group_by(by, name)
         .agg(pl.col('label').sum().alias('pos'), pl.col('E').sum().alias('exp'))
         .with_columns(pl.format('{}/{}', ((pl.col('pos') / pl.col('exp')) * 100).round(0).cast(pl.Int32), pl.col('pos')).alias('v'))
         .pivot(on=name, index=by, values='v').sort(by))
    cols = [by] + sorted([c for c in t.columns if c != by])
    print(t.select(cols))

fd = df[D.off]
print(fd.group_by('ng').agg(pl.len()).sort('ng'))
print('O/E*100 / positives, rows=newest group')
for c in ['h1', 'h5', 'hmean5', 'c_contact', 'c_fav', 'c_order', 'c_vfeed', 'c_vsearch']:
    print('====', c)
    piv(b(c, [-1, 0.6, 0.8, 0.9, 0.95, 1.01]), c)
cni = pl.col('cat_newest_idx')
print('==== cat_newest_idx')
piv(pl.when(cni < 0).then(pl.lit('a-1')).when(cni == 0).then(pl.lit('b0')).when(cni < 5).then(pl.lit('c1-4'))
    .when(cni < 32).then(pl.lit('d5-31')).otherwise(pl.lit('e32+')), 'cni')
