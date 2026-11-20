"""Loaders for the feasibility_value skeptic checks (CPU, read-only; modelled on nn_x/w12/kd/pool_read.py)."""
import os, sys
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
sys.path.insert(0, '<repo>/src')
from pathlib import Path
import numpy as np
from common import FLAT, feed_meta
from metrics import ndcg_per_feed
from stack_val import feed_rank_z

W = Path('<repo>/nn_x/w5/preds')
fv, Mv, ov = feed_meta('val'); ov = ov.astype(np.int64); yv = np.load(FLAT / 'val/cand_target.npy')
Mh, oh = np.load(W / 'holdout/holdout_meta.npy'); Mh = Mh.astype(np.int64); oh = oh.astype(np.int64)
yh = np.load(W / 'holdout/holdout_y.npy')
ev = lambda s: ndcg_per_feed(np.asarray(s, np.float32), yv, ov, Mv)
eh = lambda s: ndcg_per_feed(np.asarray(s, np.float32), yh, oh, Mh)
zv = lambda x: feed_rank_z(ov, Mv, np.asarray(x, np.float32))[:, 1]
zh = lambda x: feed_rank_z(oh, Mh, np.asarray(x, np.float32))[:, 1]
se = lambda d: d.std() / np.sqrt(len(d))
fmt = lambda d: f'{d.mean():+.5f} +- {se(d):.5f}'


def ld(v, suf=''):
    """(per-feed z on val, per-feed z on the holdout) of a w5 variant; suf '' = selected checkpoint, '@last' = last."""
    return zv(np.load(W / f'nn_{v}{suf}_trainonly_val.npy')), zh(np.load(W / f'holdout/nn_{v}{suf}_holdout.npy'))


def fam(arm, seeds, suf=''):
    return [ld(f'w5_e10_{arm}_s{s}', suf) for s in seeds]


def pool(X, i):
    return np.mean([x[i] for x in X], 0)


def done(arm, seed):
    v = f'w5_e10_{arm}_s{seed}'
    return (W / f'nn_{v}_trainonly_val.npy').exists() and (W / f'holdout/nn_{v}_holdout.npy').exists() and \
        Path(f'<repo>/nn_x/w5/logs/{v}.json').exists()
