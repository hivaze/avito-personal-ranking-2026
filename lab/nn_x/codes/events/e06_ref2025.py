"""e06: reference = Avito ML Cup 2025 clickstream (public release of the 2025 competition data:
<public repository, link removed> releases data-v1; 68.8M events, 19 event codes with is_contact).
Per 2025 event: share, is_contact, surfaces, platforms, per-(cookie,item) repetition, same-second ties with other events,
category profile (2025 'category', 53 codes). Run with python -I; path of the downloaded files is argv[1]."""
import sys
import numpy as np, polars as pl

pl.Config.set_tbl_rows(80); pl.Config.set_tbl_width_chars(260); pl.Config.set_tbl_cols(60); pl.Config.set_float_precision(4)
D = sys.argv[1]
OUT = '<repo>/nn_x/codes/events/out'
ev = pl.read_parquet(f'{sys.argv[2]}')  # events.pq (event, is_contact)
c = pl.read_parquet(f'{D}/clickstream.pq')
items = pl.read_parquet(f'{D}/cat_features.pq', columns=['item', 'category'])
c = c.join(items, on='item', how='left').join(ev, on='event', how='left')
print('rows', c.height, 'cookies', c['cookie'].n_unique(), 'items', c['item'].n_unique(), 'missing category', c['category'].null_count())
print(c.select(pl.col('event_date').min().alias('t_min'), pl.col('event_date').max().alias('t_max')))
# 1. per event profile
g = c.group_by(['cookie', 'item', 'event']).agg(pl.len().alias('n'))
rep = g.group_by('event').agg(pl.col('n').mean().alias('ev_per_cookie_item'), (pl.col('n') > 1).mean().alias('pairs_repeated'))
prof = c.group_by('event').agg(pl.len().alias('n'), pl.col('is_contact').first(), pl.col('surface').n_unique().alias('n_surfaces'),
                               pl.col('surface').mode().first().alias('top_surface'), (pl.col('surface') == pl.col('surface').mode().first()).mean().alias('top_surface_share'),
                               pl.col('platform').mode().first().alias('top_platform'), pl.col('cookie').n_unique().alias('cookies'))
prof = prof.join(rep, on='event').with_columns((pl.col('n') / c.height).alias('share')).sort('n', descending=True)
print('[R1] 2025 events'); print(prof)
prof.write_csv(f'{OUT}/e06_2025_events.csv')
# 2. surface x event
se = c.group_by(['event', 'surface']).agg(pl.len()).pivot(on='surface', index='event', values='len').fill_null(0)
cols = [x for x in se.columns if x != 'event']
se = se.with_columns(pl.sum_horizontal(cols).alias('tot'))
print('[R2] surface distribution per event (rows)')
print(se.select('event', 'tot', *[(pl.col(x) / pl.col('tot')).round(3).alias('s' + x) for x in sorted(cols, key=int)]).sort('tot', descending=True))
pe = c.group_by(['event', 'platform']).agg(pl.len()).pivot(on='platform', index='event', values='len').fill_null(0)
cols = [x for x in pe.columns if x != 'event']
pe = pe.with_columns(pl.sum_horizontal(cols).alias('tot'))
print('[R3] platform distribution per event (rows)')
print(pe.select('event', 'tot', *[(pl.col(x) / pl.col('tot')).round(3).alias('p' + x) for x in sorted(cols, key=int)]).sort('tot', descending=True))
# 3. same-second ties: for each (cookie,item,event_date) set of events
t = c.group_by(['cookie', 'item', 'event_date']).agg(pl.col('event').unique().sort().alias('evs'))
t = t.filter(pl.col('evs').list.len() > 1).with_columns(pl.col('evs').cast(pl.List(pl.Utf8)).list.join(',').alias('k'))
print('[R4] same (cookie,item,timestamp) multi-event sets, top 25')
print(t.group_by('k').agg(pl.len()).sort('len', descending=True).head(25))
# 4. item-level first-occurrence precedence among contact-ish events
fo = c.group_by(['cookie', 'item', 'event']).agg(pl.col('event_date').min().alias('t0'))
w = fo.pivot(on='event', index=['cookie', 'item'], values='t0')
E = [17, 11, 12, 10, 15, 5, 3, 19, 8, 4, 13, 16, 14, 2, 0]
print('[R5] P(first a earlier than first b | cookie-item has both), ties = 0.5;  rows a, cols b')
print('      ' + ' '.join(f'{b:>6d}' for b in E))
for a in E:
    row = []
    for b in E:
        if a == b or str(a) not in w.columns or str(b) not in w.columns:
            row.append('     .'); continue
        x = w.select(pl.col(str(a)).alias('a'), pl.col(str(b)).alias('b')).drop_nulls()
        if x.height < 50:
            row.append('     .'); continue
        p = ((x['a'] < x['b']).sum() + 0.5 * (x['a'] == x['b']).sum()) / x.height
        row.append(f'{p:6.2f}')
    print(f'{a:>5d} ' + ' '.join(row))
# 5. event x category
cm = c.group_by(['category', 'event']).agg(pl.len()).pivot(on='event', index='category', values='len').fill_null(0).sort('category')
cols = [x for x in cm.columns if x != 'category']
cm = cm.with_columns(pl.sum_horizontal(cols).alias('tot'))
cmn = cm.select('category', 'tot', *[(pl.col(x) / pl.col('tot')).round(4).alias('e' + x) for x in sorted(cols, key=int)])
print('[R6] 2025 event mix per category (rows), sorted by category code')
print(cmn)
cm.write_csv(f'{OUT}/e06_2025_event_x_category_counts.csv')
# category share of all events and of items
cs = c.group_by('category').agg(pl.len().alias('events'), pl.col('item').n_unique().alias('items')).sort('category')
cs.write_csv(f'{OUT}/e06_2025_category_sizes.csv')
print(cs.with_columns((pl.col('events') / pl.col('events').sum()).alias('ev_share')))
