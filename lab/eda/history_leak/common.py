"""Shared helpers for the history/leak EDA."""
import sys
import numpy as np
import polars as pl

sys.path.insert(0, '<repo>/src')
from metrics import ndcg, ndcg_per_feed  # noqa: E402

ROOT = '<repo>/data/flat'


def feeds(split):
    return pl.read_parquet(f'{ROOT}/{split}/feeds.parquet')


def arr(split, name, mmap=True):
    return np.load(f'{ROOT}/{split}/{name}.npy', mmap_mode='r' if mmap else None)


def load_split(split, cand_cols=('item_id', 'category_id', 'f1', 'target'),
               hist_cols=('item_id', 'category_id', 'event_type', 'event_source', 'hpos')):
    f = feeds(split)
    d = {'feeds': f}
    for c in cand_cols:
        try:
            d['c_' + c] = np.asarray(arr(split, 'cand_' + c))
        except FileNotFoundError:
            pass
    for c in hist_cols:
        d['h_' + c] = np.asarray(arr(split, 'hist_' + c))
    d['c_fidx'] = np.asarray(arr(split, 'cand_fidx'))
    d['h_fidx'] = np.asarray(arr(split, 'hist_fidx'))
    return d


def eval_score(score, d, name='', base=None, verbose=True):
    """NDCG of score on a split dict d (val). Returns per-feed array."""
    f = d['feeds']
    off = f['cand_off'].to_numpy()
    M = f['M'].to_numpy()
    pf = ndcg_per_feed(score, d['c_target'], off, M)
    if verbose:
        msg = f'{name:50s} NDCG={pf.mean():.4f}'
        if base is not None:
            diff = pf - base
            se = diff.std(ddof=1) / np.sqrt(len(diff))
            msg += f'  delta={diff.mean():+.4f} (paired SE {se:.4f}, t={diff.mean()/max(se,1e-12):.1f})'
        print(msg, flush=True)
    return pf


def rank_in_feed(x, fidx):
    """Within-feed percentile rank of x (0..1), ties averaged."""
    import scipy.stats as ss
    df = pl.DataFrame({'f': fidx, 'x': x})
    r = df.select(pl.col('x').rank('average').over('f') / pl.len().over('f'))['x'].to_numpy()
    return r
