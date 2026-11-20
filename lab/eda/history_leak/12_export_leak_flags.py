"""12) Export per-candidate cross-feed leak flags for val and test (keys feed_id,item_id) for downstream use.
leak_pos = item appears in the history of a LATER feed of the same (non-zero) user  (=> clicked after this feed)
leak_neg = item appears as a candidate of a LATER feed of the same user (=> not viewed before that feed)
earlier_hist = item appears in the history of an EARLIER feed of the same user (mild positive)."""
import numpy as np
import polars as pl
from common import feeds
OUT = '<repo>/eda/history_leak/'
for s in ['val', 'test']:
    X = pl.read_parquet(OUT + f'out_crossfeed_{s}.parquet')
    f = feeds(s).select(['fidx', 'feed_id'])
    X = X.join(f, on='fidx', how='left')
    E = X.select(['feed_id', 'item_id',
                  pl.col('later_hist_dt').is_not_null().cast(pl.Int8).alias('leak_pos'),
                  pl.col('later_hist_type').fill_null(0).alias('leak_pos_event_type'),
                  pl.col('later_hist_dt').fill_null(-1).alias('leak_pos_dt_sec'),
                  pl.col('later_cand_dt').is_not_null().cast(pl.Int8).alias('leak_neg'),
                  pl.col('earlier_hist_dt').is_not_null().cast(pl.Int8).alias('earlier_hist'),
                  pl.col('earlier_hist_type').fill_null(0).alias('earlier_hist_event_type'),
                  (pl.col('n_later_feeds') > 0).cast(pl.Int8).alias('user_has_later_feed')])
    E.write_parquet(OUT + f'leak_flags_{s}.parquet')
    print(s, E.shape, E.select(pl.col('leak_pos').sum(), pl.col('leak_neg').sum(), pl.col('earlier_hist').sum(),
                               pl.col('feed_id').filter(pl.col('leak_pos') == 1).n_unique().alias('feeds_leak_pos'),
                               pl.col('feed_id').filter(pl.col('user_has_later_feed') == 1).n_unique().alias('feeds_with_later')).to_dicts())
