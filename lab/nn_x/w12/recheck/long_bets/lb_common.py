"""Shared loaders for the long_bets probes (read-only on preds). Follows nn_x/w12/kd/pool_read.py conventions."""
import sys, os
os.environ.setdefault("OMP_NUM_THREADS", "2"); os.environ.setdefault("MKL_NUM_THREADS", "2"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("POLARS_MAX_THREADS", "2")
sys.path.insert(0, '<repo>/src')
from pathlib import Path
import numpy as np
from common import feed_meta, FLAT
from metrics import ndcg_per_feed
from stack_val import feed_rank_z
ROOT = Path('<repo>'); W = ROOT / 'nn_x/w5/preds'; P = ROOT / 'preds'
fv, Mv, ov = feed_meta('val'); yv = np.load(FLAT / 'val/cand_target.npy')
Mh, oh = np.load(W / 'holdout/holdout_meta.npy'); Mh = Mh.astype(np.int64); oh = oh.astype(np.int64); yh = np.load(W / 'holdout/holdout_y.npy')
ev = lambda s: ndcg_per_feed(np.asarray(s, np.float32), yv, ov, Mv)
eh = lambda s: ndcg_per_feed(np.asarray(s, np.float32), yh, oh, Mh)
zv = lambda x: feed_rank_z(ov, Mv, np.asarray(x, np.float32))[:, 1]
zh = lambda x: feed_rank_z(oh, Mh, np.asarray(x, np.float32))[:, 1]
se = lambda d: d.std() / np.sqrt(len(d))
f = lambda d: f'{d.mean():+.5f}+-{se(d):.5f}'
def ld(v, suf=''):
    """(val z, holdout z) of a w5 train-only run."""
    return zv(np.load(W / f'nn_{v}{suf}_trainonly_val.npy')), zh(np.load(W / f'holdout/nn_{v}{suf}_holdout.npy'))
def pool(ms):
    return np.mean([m[0] for m in ms], 0), np.mean([m[1] for m in ms], 0)
def wcorr(a, b):
    """mean within-feed Pearson is ~ Pearson of per-feed z vectors (z has mean 0 per feed); use global corr of z arrays."""
    return float(np.corrcoef(a, b)[0, 1])
S4 = (951, 952, 953, 954)
