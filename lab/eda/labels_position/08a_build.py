"""Build & cache candidate feature tables: val (all feeds) and a random 200K-feed train sample (seed 21)."""
from featlib import *
import time, sys
nb.set_num_threads(40)
t0 = time.time()
def finish(df, split):
    if split == 'train':
        df = df.with_columns((pl.col('pos').rank('ordinal').over('fid') - 1).cast(pl.Int32).alias('prk'))
        df = df.with_columns((pl.col('M') - 1 - pl.col('prk')).alias('rkend'))
    return df
which = sys.argv[1] if len(sys.argv) > 1 else 'both'
if which in ('both', 'val'):
    df, off, M, sel = build('val', None)
    df = add_hist_sim(df, 'val', sel, off, M)
    df = finish(df, 'val').with_columns(pl.lit(0).alias('dummy'))
    df = df.with_columns(pl.Series('user_id', np.repeat(feeds('val')['user_id'].to_numpy(), M)))
    df.write_parquet(f'{OUT}/feat_val.parquet'); print('val', df.shape, time.time() - t0, flush=True)
if which in ('both', 'train'):
    F = feeds('train')
    sel = subsample_feeds(F, 200_000, seed=21)
    df, off, M, sel = build('train', sel)
    df = add_hist_sim(df, 'train', sel, off, M)
    df = finish(df, 'train')
    df = df.with_columns(pl.Series('fidx', np.repeat(sel, M).astype(np.int32)))
    df.write_parquet(f'{OUT}/feat_train200k.parquet'); print('train', df.shape, time.time() - t0, flush=True)
