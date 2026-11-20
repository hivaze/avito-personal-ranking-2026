"""1) Semantics of history event_type / event_source: frequencies, hpos, categories, repeats."""
import numpy as np
import polars as pl
from common import feeds, arr

pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(40)
pl.Config.set_tbl_width_chars(250)


def hist_df(split, sample_feeds=None, seed=0):
    f = feeds(split)
    if sample_feeds is not None:
        rng = np.random.default_rng(seed)
        sel = np.sort(rng.choice(len(f), sample_feeds, replace=False))
        f = f[sel]
    off = f['hist_off'].to_numpy(); H = f['H'].to_numpy()
    idx = np.concatenate([np.arange(o, o + h) for o, h in zip(off, H)]) if sample_feeds else np.arange(H.sum())
    cols = {}
    for c in ['fidx', 'item_id', 'category_id', 'event_type', 'event_source', 'hpos', 'emb_idx']:
        a = arr(split, 'hist_' + c)
        cols[c] = np.asarray(a[idx]) if sample_feeds else np.asarray(a)
    return pl.DataFrame(cols), f


def main():
    out = {}
    for split, samp in [('val', None), ('test', None), ('train', 200_000)]:
        h, f = hist_df(split, samp)
        print(f'\n===== {split}: {len(f)} feeds, {len(h)} events =====')
        n = len(h)
        et = h.group_by('event_type').agg(
            pl.len().alias('n'),
            (pl.len() / n).alias('share'),
            pl.col('hpos').mean().alias('mean_hpos'),
            (pl.col('hpos') < 10).mean().alias('frac_hpos<10'),
            (pl.col('category_id') == 0).mean().alias('cat0'),
            (pl.col('emb_idx') < 0).mean().alias('no_emb'),
            pl.col('fidx').n_unique().alias('n_feeds_with'),
            pl.col('event_source').mode().first().alias('top_source'),
        ).sort('event_type')
        et = et.with_columns((pl.col('n_feeds_with') / len(f)).alias('frac_feeds_with'))
        print(et)
        es = h.group_by('event_source').agg(
            pl.len().alias('n'), (pl.len() / n).alias('share'),
            pl.col('hpos').mean().alias('mean_hpos'),
            pl.col('event_type').mode().first().alias('top_type'),
            pl.col('event_type').n_unique().alias('n_types')).sort('event_source')
        print(es)
        if split == 'val':
            ct = h.group_by(['event_type', 'event_source']).len().pivot(
                on='event_source', index='event_type', values='len', sort_columns=True).sort('event_type').fill_null(0)
            print('type x source counts (val)')
            print(ct)
        # repeated items within a feed's history
        g = h.group_by(['fidx', 'item_id']).agg(pl.len().alias('cnt'),
                                                  pl.col('event_type').unique().sort().alias('types'))
        rep = g.filter(pl.col('cnt') > 1)
        print(f'distinct (feed,item) pairs {len(g)}, with >1 events {len(rep)} ({len(rep)/len(g):.3f}); '
              f'events on repeated items {rep["cnt"].sum()/n:.3f}')
        # per event type: share of its events whose item repeats in same history
        hh = h.join(g.select(['fidx', 'item_id', 'cnt']), on=['fidx', 'item_id'])
        print('frac of events whose item has >1 event in the same history, by type:')
        print(hh.group_by('event_type').agg((pl.col('cnt') > 1).mean().alias('rep_frac'),
                                            pl.col('cnt').mean().alias('mean_cnt')).sort('event_type'))
        # transitions: for same (feed,item), event (older) -> next newer event type
        hs = h.sort(['fidx', 'item_id', 'hpos'], descending=[False, False, True])  # older first
        same = (hs['fidx'].to_numpy()[1:] == hs['fidx'].to_numpy()[:-1]) & (hs['item_id'].to_numpy()[1:] == hs['item_id'].to_numpy()[:-1])
        older = hs['event_type'].to_numpy()[:-1][same]; newer = hs['event_type'].to_numpy()[1:][same]
        tr = pl.DataFrame({'older': older, 'newer': newer}).group_by(['older', 'newer']).len().pivot(
            on='newer', index='older', values='len', sort_columns=True).sort('older').fill_null(0)
        print('same-item transitions older->newer (rows=older type, cols=newer type)')
        print(tr)
        # first event type of an item (oldest) vs set
        # adjacency in hpos: is same item consecutive?
        dpos = hs['hpos'].to_numpy()[:-1][same] - hs['hpos'].to_numpy()[1:][same]
        print('hpos gap between consecutive same-item events: quantiles', np.quantile(dpos, [0.1, 0.25, 0.5, 0.75, 0.9]))
        # categories per event type (top categories)
        if split == 'val':
            cat = h.group_by(['event_type', 'category_id']).len().sort('len', descending=True)
            top = cat.group_by('event_type').agg(pl.col('category_id').head(5).alias('top_cats'),
                                                 (pl.col('len').head(5).sum() / pl.col('len').sum()).alias('top5_share')).sort('event_type')
            print(top)
            # history length distribution and event type composition by H
            print('H==256 share', (f['H'] == 256).mean(), 'H==0 share', (f['H'] == 0).mean())
            # position-type profile: share of each type by hpos bucket
            prof = h.with_columns((pl.col('hpos') // 32).alias('b')).group_by(['b', 'event_type']).len().pivot(
                on='event_type', index='b', values='len', sort_columns=True).sort('b').fill_null(0)
            print('event type counts by hpos bucket (32-wide)')
            print(prof)


if __name__ == '__main__':
    main()
