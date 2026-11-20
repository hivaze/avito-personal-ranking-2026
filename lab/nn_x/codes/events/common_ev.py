import numpy as np, polars as pl
ROOT = '<repo>/data/flat'


def feeds(split):
    return pl.read_parquet(f'{ROOT}/{split}/feeds.parquet').sort('fidx')


def arr(split, name):
    return np.load(f'{ROOT}/{split}/{name}.npy', mmap_mode='r')


def hist_frame(split, fidx=None, seed=0):
    """History events as a polars frame. For train, fidx = array of feed indices (sample)."""
    F = feeds(split)
    if fidx is None:
        rows = slice(None)
        Fs = F
    else:
        Fs = F.filter(pl.col('fidx').is_in(fidx)).sort('fidx')
        off = Fs['hist_off'].to_numpy(); H = Fs['H'].to_numpy()
        rows = np.concatenate([np.arange(o, o + h) for o, h in zip(off, H)])
    d = {}
    for c in ['fidx', 'item_id', 'category_id', 'event_type', 'event_source', 'hpos', 'emb_idx']:
        x = arr(split, 'hist_' + c)
        d[c] = np.asarray(x[rows]) if not isinstance(rows, slice) else np.asarray(x)
    df = pl.DataFrame(d).rename({'event_type': 'et', 'event_source': 'es', 'category_id': 'cat'})
    df = df.join(Fs.select('fidx', 'user_id', 'H', 'timestamp'), on='fidx', how='left')
    return df, Fs


def cand_frame(split, fidx=None):
    F = feeds(split)
    if fidx is None:
        rows = None; Fs = F
    else:
        Fs = F.filter(pl.col('fidx').is_in(fidx)).sort('fidx')
        off = Fs['cand_off'].to_numpy(); M = Fs['M'].to_numpy()
        rows = np.concatenate([np.arange(o, o + m) for o, m in zip(off, M)])
    d = {}
    for c in ['fidx', 'item_id', 'category_id', 'emb_idx', 'f1', 'f2', 'f3', 'f4', 'f5', 'target']:
        try:
            x = arr(split, 'cand_' + c)
        except FileNotFoundError:
            continue
        d[c] = np.asarray(x) if rows is None else np.asarray(x[rows])
    return pl.DataFrame(d).rename({'category_id': 'cat'}), Fs
