import sys, os, numpy as np, polars as pl
sys.path.insert(0, '<repo>/src')
from metrics import ndcg, ndcg_per_feed  # noqa
FLAT = '<repo>/data/flat'
OUT = '<repo>/eda/shift_overlap/out'
SPLITS = ['train', 'val', 'test']
MSK = 3 * 3600  # timestamps: day boundaries align with 21:00 UTC => Moscow time

def feeds(s):
    f = pl.read_parquet(f'{FLAT}/{s}/feeds.parquet')
    t = f['timestamp'].to_numpy().astype(np.int64) + MSK
    f = f.with_columns(
        pl.Series('ts_msk', t),
        pl.Series('day', (t // 86400)).cast(pl.Int64),
        pl.Series('hour', ((t % 86400) // 3600)).cast(pl.Int8),
    )
    f = f.with_columns(pl.from_epoch(pl.col('ts_msk'), time_unit='s').dt.date().alias('date'))
    return f

def cand(s, c):
    return np.load(f'{FLAT}/{s}/cand_{c}.npy', mmap_mode='r')

def hist(s, c):
    return np.load(f'{FLAT}/{s}/hist_{c}.npy', mmap_mode='r')

def js_div(p, q, eps=1e-12):
    p = np.asarray(p, float); q = np.asarray(q, float)
    p = p / p.sum(); q = q / q.sum(); m = 0.5 * (p + q)
    def kl(a, b):
        mk = a > 0
        return np.sum(a[mk] * np.log2(a[mk] / (b[mk] + eps)))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)

def psi(p, q, eps=1e-6):
    p = np.asarray(p, float); q = np.asarray(q, float)
    p = p / p.sum() + eps; q = q / q.sum() + eps
    return float(np.sum((p - q) * np.log(p / q)))

def val_eval(scores, s='val'):
    f = pl.read_parquet(f'{FLAT}/{s}/feeds.parquet')
    M = f['M'].to_numpy(); off = f['cand_off'].to_numpy()
    t = cand(s, 'target')
    return ndcg(np.asarray(scores), np.asarray(t), off, M)
