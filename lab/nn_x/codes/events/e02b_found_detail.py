"""e02b: details on A candidates found in B (from out/e02_found.parquet): view presence with contact-type events,
category dependence of 'no view' positives, and target rate per first event by category group."""
import polars as pl
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(240); pl.Config.set_float_precision(4)
D = pl.read_parquet('<repo>/nn_x/codes/events/out/e02_found.parquet')
NODELIV = [2, 5, 9, 10, 11, 12, 19, 25, 26, 27, 30, 31, 34, 35, 36, 37, 38, 48, 49, 50, 51]
D = D.with_columns(pl.col('cat').is_in(NODELIV).alias('nodeliv_cat'))
print('[B1] found positives: P(has type-1 event | has type t), positives vs negatives')
rows = []
for t in range(2, 11):
    for lab in (1, 0):
        d = D.filter((pl.col('t') == lab) & pl.col(f'h{t}'))
        rows.append({'type': t, 'label': lab, 'n': d.height, 'P(has1)': d['h1'].mean(), 'P(first is 1)': (d['first_t'] == 1).mean()})
print(pl.DataFrame(rows))
print('[B2] found positives without any type-1 event: share by category (top 20 by count) vs all found positives')
pos = D.filter(pl.col('t') == 1)
a = pos.group_by('cat').agg(pl.len().alias('n'), (~pl.col('h1')).mean().alias('no_view_share'),
                            (pl.col('first_t').is_in([2, 3])).mean().alias('first_is_2or3'),
                            (pl.col('first_t').is_in([5, 6])).mean().alias('first_is_5or6'),
                            (pl.col('first_t') == 4).mean().alias('first_is_4'), (pl.col('first_t') == 7).mean().alias('first_is_7'))
print(a.sort('n', descending=True).head(25))
print('[B3] by delivery group (NODELIV = categories with zero 5/6/8 in histories)')
print(pos.group_by('nodeliv_cat').agg(pl.len(), (~pl.col('h1')).mean().alias('no_view_share'), pl.col('h2').mean(), pl.col('h3').mean(),
                                      pl.col('h4').mean(), pl.col('h5').mean(), pl.col('h7').mean(), pl.col('h9').mean()))
print('[B4] gap dependence of no-view positives')
print(pos.with_columns(pl.col('gb')).group_by('gb').agg(pl.len(), (~pl.col('h1')).mean().alias('no_view_share')).sort('gb'))
