import json
import sys
import numpy as np
import polars as pl

sys.path.insert(0, '<repo>/src')
from metrics import ndcg, ndcg_per_feed  # noqa: E402

FLAT = '<repo>/data/flat'
COMP = '<repo>/data/competition'


def feeds(split):
    return pl.read_parquet(f'{FLAT}/{split}/feeds.parquet')


def arr(split, name, mmap=True):
    return np.load(f'{FLAT}/{split}/{name}.npy', mmap_mode='r' if mmap else None)


def item_table(split):
    meta = json.load(open(f'{COMP}/{split}/item_embeddings/data.meta'))
    return np.memmap(f'{COMP}/{split}/item_embeddings/data.memmap', dtype=np.int8, mode='r',
                     shape=tuple(meta['shape']))


def gather_emb(table, idx, out_dtype=np.float32):
    """Gather rows; missing (-1) -> zero vector."""
    idx = np.asarray(idx)
    out = np.zeros((len(idx), table.shape[1]), dtype=out_dtype)
    m = idx >= 0
    if m.any():
        ii = idx[m]
        order = np.argsort(ii, kind='stable')
        rows = np.asarray(table[ii[order]], dtype=out_dtype)
        tmp = np.empty_like(rows)
        tmp[order] = rows
        out[m] = tmp
    return out


def eval_ndcg(scores, split='val', per_feed=False):
    f = feeds(split)
    M = f['M'].to_numpy()
    off = f['cand_off'].to_numpy()
    t = arr(split, 'cand_target', mmap=False)
    if per_feed:
        return ndcg_per_feed(scores, t, off, M)
    return ndcg(scores, t, off, M)


def paired(a_pf, b_pf):
    d = a_pf - b_pf
    return d.mean(), d.std(ddof=1) / np.sqrt(len(d))
