"""why/noise (2026-11-19): shared loaders for the noise / shared-reference / diversity analysis of the w12/moe screen.
CPU only, read-only on every prediction file. Member score = per-feed z of the train-only (selected-checkpoint) prediction, exactly as
arm_read8_moe.py / pc_common (feed_rank_z[:, 1]); pool = member mean of z; per-feed NDCG = pc_common.ndcg_fast (pessimistic ties).
z arrays and member per-feed NDCG are cached under the session scratchpad (never inside the repo)."""
import os, sys
for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'POLARS_MAX_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[k] = '12'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, '<repo>/nn_x/w12/recheck/pool_composition')
from pc_common import Meta, ndcg_fast, paths as _paths_orig, ROOT  # noqa: E402
import numpy as np  # noqa: E402
from pathlib import Path  # noqa: E402

MOE_OUT = ROOT / 'lab/nn_x/w12/moe/out'
CACHE = Path('/tmp/scratchpad/why/noise_cache')
CACHE.mkdir(parents=True, exist_ok=True)
OUT = ROOT / 'lab/nn_x/w12/moe/why/noise'
ARMS = {'A': 'kd05w192moeffn', 'B': 'kd05w192moevo', 'C': 'kd05w192b3ff1'}
S4A, S4B, S8 = (951, 952, 953, 954), (955, 956, 957, 958), tuple(range(951, 959))
nm = lambda a, s: f'w5_e10_{a}_s{s}'


def paths(v, suf=''):
    if (MOE_OUT / 'preds' / f'nn_{v}{suf}_trainonly_val.npy').exists():
        Wm = MOE_OUT / 'preds'
        return Wm / f'nn_{v}{suf}_trainonly_val.npy', Wm / f'holdout/nn_{v}{suf}_holdout.npy', Wm / f'nn_{v}{suf}_trainonly_test.npy'
    return _paths_orig(v, suf)


def exists(v, suf=''):
    pv, ph, _ = paths(v, suf)
    return pv.exists() and ph.exists()


_m = None


def meta():
    global _m
    if _m is None:
        _m = Meta()
    return _m


_zc = {}


def Z(v, suf=''):
    """(z_val, z_hold) float32 of member v (cached in RAM and on scratchpad disk)."""
    key = v + suf
    if key in _zc:
        return _zc[key]
    f = CACHE / f'z_{key}.npz'
    if f.exists():
        d = np.load(f); r = (d['v'], d['h'])
    else:
        m = meta(); pv, ph, _ = paths(v, suf)
        r = (m.z('val', np.load(pv).astype(np.float32)).astype(np.float32), m.z('hold', np.load(ph).astype(np.float32)).astype(np.float32))
        np.savez(f, v=r[0], h=r[1])
    _zc[key] = r
    return r


_nc = {}


def member_ndcg(v, suf=''):
    """(per-feed val NDCG, per-feed holdout NDCG) of a single member."""
    key = v + suf
    if key in _nc:
        return _nc[key]
    f = CACHE / f'n_{key}.npz'
    if f.exists():
        d = np.load(f); r = (d['v'], d['h'])
    else:
        m = meta(); zv, zh = Z(v, suf)
        r = (ndcg_fast(zv, m.yv, m.ov, m.Mv), ndcg_fast(zh, m.yh, m.oh, m.Mh))
        np.savez(f, v=r[0], h=r[1])
    _nc[key] = r
    return r


def pool_ndcg(vs, suf=''):
    """(per-feed val NDCG, per-feed holdout NDCG) of the member-mean z pool."""
    m = meta()
    zv = np.mean([Z(v, suf)[0] for v in vs], 0); zh = np.mean([Z(v, suf)[1] for v in vs], 0)
    return ndcg_fast(zv, m.yv, m.ov, m.Mv), ndcg_fast(zh, m.yh, m.oh, m.Mh)


se = lambda d: float(np.std(d, ddof=1) / np.sqrt(len(d)))
fmt = lambda d: f'{np.mean(d):+.5f}+-{se(d):.5f}'
