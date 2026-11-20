import sys, os, time
import numpy as np
import polars as pl

sys.path.insert(0, '<repo>/src')
from metrics import ndcg_per_feed  # noqa

FLAT = '<repo>/data/flat'
OUT = '<repo>/eda/collab_stats/out'
os.makedirs(OUT, exist_ok=True)
NITEM = 111319575 + 1


def feeds(split):
    return pl.read_parquet(f'{FLAT}/{split}/feeds.parquet')


def arr(split, name, mmap=True):
    return np.load(f'{FLAT}/{split}/{name}.npy', mmap_mode='r' if mmap else None)


class Val:
    """Holds val structure for evaluation."""
    def __init__(self, split='val'):
        f = feeds(split)
        self.split = split
        self.f = f
        self.M = f['M'].to_numpy().astype(np.int64)
        self.off = f['cand_off'].to_numpy().astype(np.int64)
        self.n = len(self.M)
        self.target = np.asarray(arr(split, 'cand_target')).astype(np.float64)
        self.fid = np.repeat(np.arange(self.n), self.M)

    def per_feed(self, s):
        s = np.asarray(s, dtype=np.float64)
        s = np.nan_to_num(s, nan=-1e18)
        return ndcg_per_feed(s, self.target, self.off, self.M)

    def ndcg(self, s):
        return float(self.per_feed(s).mean())

    def paired(self, s, base):
        a = self.per_feed(s); b = self.per_feed(base)
        d = a - b
        return a.mean(), b.mean(), d.mean(), d.std(ddof=1) / np.sqrt(len(d))


def tlog(t0, msg):
    print(f'[{time.time()-t0:7.1f}s] {msg}', flush=True)


def dense_rank(s):
    s = np.nan_to_num(np.asarray(s, dtype=np.float64), nan=-1e18)
    u, inv = np.unique(s, return_inverse=True)
    return inv.astype(np.float64)


def tiebreak(s, tb):
    """primary key s, secondary key tb (any array); returns a score with no extra ties."""
    r = dense_rank(s)
    t = dense_rank(tb)
    return r + t / (t.max() + 1.0)


def _patch():
    def ndcg_rt(self, s, seed=0):
        rnd = np.random.RandomState(seed).rand(len(self.target))
        return self.ndcg(tiebreak(s, rnd))
    Val.ndcg_rt = ndcg_rt
_patch()


def _patch2():
    def pf_rt(self, s, seeds=(0, 1, 2)):
        """per-feed NDCG averaged over random tie-breaks"""
        out = 0
        for sd in seeds:
            rnd = np.random.RandomState(1000 + sd).rand(len(self.target))
            out = out + self.per_feed(tiebreak(s, rnd))
        return out / len(seeds)
    def cmp(self, s, base_pf):
        a = self.pf_rt(s); d = a - base_pf
        return a.mean(), d.mean(), d.std(ddof=1) / np.sqrt(len(d))
    def rand_pf(self, nseeds=20):
        out = 0
        for sd in range(nseeds):
            out = out + self.per_feed(np.random.RandomState(5000 + sd).rand(len(self.target)))
        return out / nseeds
    Val.pf_rt = pf_rt; Val.cmp = cmp; Val.rand_pf = rand_pf
_patch2()
