"""gateB shared loader (searcher B, 2026-11-15). Re-implements build_r4.py's pool / branch / final forms; reads only.
Pools = member-equal mean of per-feed z (src/stack_val.feed_rank_z[:, 1]); branch = (1 - w) z(base) + w z(pop); final = w_ft z(FT branch) + (1 - w_ft) z(TO branch)."""
import os, sys
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
sys.dont_write_bytecode = True
sys.path.insert(0, '<repo>/src')
import numpy as np, polars as pl
from pathlib import Path
from common import feed_meta, FLAT
from metrics import ndcg_per_feed
from stack_val import feed_rank_z

ROOT = Path('<repo>'); P = ROOT / 'preds'; HERE = ROOT / 'nn_x/w12/mix/gateB'; CACHE = HERE / 'cache'
CACHE.mkdir(exist_ok=True)
S8 = range(951, 959); S4 = range(951, 955)
SUF = {'val': ('n2cpv3s3_val_oof', 'trainonly_val'), 'test': ('n2cpv3s3_test', 'trainonly_test')}
MEM = {
    'FD16': [f'w5_fd10_{a}_s{s}' for a in ('kd05w192', 'kd05t25w192') for s in S8],
    'FDP': [f'abl_fd10_pop1kd05w192_s{s}' for s in S4],
    'W16': [f'w5_e10_{a}_s{s}' for a in ('kd05w192', 'kd05t25w192') for s in S8],
    'P1w': [f'abl_e10_pop1kd05w192_s{s}' for s in S4],
    'P1d': [f'abl_e10_pop1kd05t25_s{s}' for s in S4],
}
_META = {}


def meta(sp):
    if sp not in _META:
        f, M, off = feed_meta(sp); _META[sp] = (f, M, off)
    return _META[sp]


def Z(sp, x):
    f, M, off = meta(sp)
    return feed_rank_z(off, M, np.asarray(x, np.float32))[:, 1]


def member_z(sp, name, branch):
    """branch 0 = fine-tuned (OOF on val), 1 = train-only."""
    return Z(sp, np.load(P / f'nn_{name}_{SUF[sp][branch]}.npy'))


def pool(sp, names):
    """(FT pool, TO pool): member-equal mean of per-feed z, exactly build_r4.pool."""
    return (np.mean([member_z(sp, m, 0) for m in names], 0), np.mean([member_z(sp, m, 1) for m in names], 0))


def pools4(sp, base, pop, tag=None):
    """dict of the four pool arrays zB_ft, zB_to, zP_ft, zP_to (each RE-standardised per feed as build_r4 does before mixing)."""
    if tag is not None and (CACHE / f'{tag}_{sp}.npz').exists():
        d = np.load(CACHE / f'{tag}_{sp}.npz'); return {k: d[k] for k in d.files}
    b = pool(sp, base); p = pool(sp, pop)
    out = {'Bft': Z(sp, b[0]), 'Bto': Z(sp, b[1]), 'Pft': Z(sp, p[0]), 'Pto': Z(sp, p[1])}
    if tag is not None:
        np.savez(CACHE / f'{tag}_{sp}.npz', **out)
    return out


def branches(sp, q, wrow):
    """wrow: scalar or per-row float32 weight of the popularity-light pool."""
    return Z(sp, (1 - wrow) * q['Bft'] + wrow * q['Pft']), Z(sp, (1 - wrow) * q['Bto'] + wrow * q['Pto'])


def final(sp, q, wrow, wft=0.5):
    bf, bt = branches(sp, q, wrow)
    return wft * bf + (1 - wft) * bt


def labels():
    return np.load(FLAT / 'val/cand_target.npy')


def ev(s, y=None):
    f, M, off = meta('val')
    return ndcg_per_feed(np.asarray(s, np.float32), labels() if y is None else y, off, M)


def groups():
    """user-grouped ids for val feeds; user_id 0 feeds are singleton groups."""
    f, M, off = meta('val'); u = f['user_id'].to_numpy().astype(np.int64)
    g = u.copy(); z = u == 0; g[z] = -(np.arange(z.sum()) + 1)
    return np.unique(g, return_inverse=True)[1]


def cluster_se(d, g):
    """mean and user-cluster-robust SE of a per-feed array."""
    d = np.asarray(d, np.float64); m = d.mean(); s = np.bincount(g, d - m)
    return m, np.sqrt((s ** 2).sum()) / len(d)
