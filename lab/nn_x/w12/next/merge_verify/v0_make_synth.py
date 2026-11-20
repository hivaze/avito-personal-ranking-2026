"""V0. Synthetic partner files for the reader checks of section 2.3 / 2.5 (written to my cache only; tool inputs, not candidates; seed 5).
  cd <repo> && CUDA_VISIBLE_DEVICES="" POLARS_MAX_THREADS=12 .venv/bin/python -B nn_x/w12/next/merge_verify/v0_make_synth.py"""
import os; os.environ['CUDA_VISIBLE_DEVICES'] = ''; os.environ.setdefault('POLARS_MAX_THREADS', '12')
import polars as pl, numpy as np
C = '<repo>/nn_x/w12/next/merge_verify/cache/'
df = pl.read_parquet('<repo>/subs/cand_Q2a8_ftbag3_memeq.parquet'); f = df['feed_id'].to_numpy(); uf = np.unique(f); rng = np.random.default_rng(5)
for share, nm in ((0.30, 'V_partial_inversion_30pct'), (0.10, 'V_partial_inversion_10pct')):
    inv = rng.choice(uf, int(share * len(uf)), replace=False); m = np.isin(f, inv); s = df['score'].to_numpy().astype(np.float32).copy(); s[m] = -s[m]
    pl.DataFrame({'feed_id': df['feed_id'], 'item_id': df['item_id'], 'score': s}).write_parquet(C + nm + '.parquet'); print(nm, int(m.sum()), 'rows in', len(inv), 'feeds negated')
g = pl.DataFrame({'feed_id': df['feed_id'], 'item_id': df['item_id'], 'score': df['score'], 'k': rng.random(len(df))}); bad = rng.choice(uf, int(0.12 * len(uf)), replace=False)
a = g.filter(pl.col('feed_id').is_in(bad)); b = g.filter(~pl.col('feed_id').is_in(bad))
a2 = a.sort(['feed_id', 'k']).with_columns(pl.col('score').alias('s2')); a1 = a.sort(['feed_id', 'item_id']); a = a1.with_columns(a2['s2'].alias('score'))   # scores of a feed re-assigned to its items at random
pl.concat([b.select('feed_id', 'item_id', 'score'), a.select('feed_id', 'item_id', 'score')]).write_parquet(C + 'V_shuffled_12pct_feeds.parquet'); print('V_shuffled_12pct_feeds', len(bad), 'feeds shuffled')
d2 = pl.read_parquet('<repo>/nn_x/w12/next/merge/cache/NOTSUB_c3_test_scores.parquet'); s = d2['score'].to_numpy().astype(np.float64).copy(); s[[10, 500000, 2000000]] = 1e200
pl.DataFrame({'feed_id': d2['feed_id'], 'item_id': d2['item_id'], 'score': s}).write_parquet(C + 'V_huge_values_3rows.parquet'); print('V_huge_values_3rows')
