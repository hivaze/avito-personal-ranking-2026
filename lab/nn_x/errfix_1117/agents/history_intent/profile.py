import hi, numpy as np, polars as pl
D = hi.Data(); df = D.df
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_cols(20); pl.Config.set_tbl_width_chars(220)
r = pl.col('rank')
pos = df.filter(pl.col('label') == 1).with_columns(
    pl.when(r <= 3).then(pl.lit('a1-3')).when(r <= 10).then(pl.lit('b4-10')).when(r <= 30).then(pl.lit('c11-30')).otherwise(pl.lit('d31+')).alias('rb'))
neg_top = df.filter((pl.col('label') == 0) & (r <= 3)).with_columns(pl.lit('neg_top3').alias('rb'))
allp = pl.concat([pos, neg_top])
t = allp.group_by('rb').agg(pl.len().alias('n'),
    (pl.col('cat_newest_idx') == 0).mean().alias('cni0'), (pl.col('cat_newest_idx') < 0).mean().alias('cni-1'),
    ((pl.col('cat_newest_idx') >= 0) & (pl.col('cat_newest_idx') < 10)).mean().alias('cni<10'),
    pl.col('h1').median().alias('h1'), pl.col('h5').median().alias('h5'), pl.col('hmean5').median().alias('hmean5'), pl.col('hmax').median().alias('hmax'),
    (pl.col('hmax') >= 0.95).mean().alias('hmax>=.95'), pl.col('ucos').median().alias('ucos'), pl.col('M').median().alias('M'),
    pl.col('xp_imp_c50').median().alias('imp'), (pl.col('xp_imp_c50') == 0).mean().alias('cold'), pl.col('cat_n_in_feed').median().alias('catn'),
    pl.col('cat_rank_in_feed').median().alias('catrk')).sort('rb')
print(t)
