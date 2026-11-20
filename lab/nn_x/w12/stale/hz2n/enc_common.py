"""hz2n: the row-wise encoder-score view = solvers/CPU2_stale_replay_pre2s_and_arms.py::build_pre2both, written once and used for
(1) the VAL rows (witness parity proof P2) and (2) all TRAIN rows of the three views. CPU only, label-free: no cand_target file is opened anywhere.

Construction (DECLARATION.txt D0), for a block of contiguous feeds (off, M) with stored config-mean columns fe / noid / fs / fz, per-config aligned scores
fes_c / noids_c and a bool row mask nc ("newly cold": view counter == 0, stored counter > 0, not a keep row):
    fe2  = where(nc, noid, fe)                                    (witness: fe2 = np.where(nc, noid, fe).astype(np.float32))
    fs2  = where(nc, float32((fe2 + 20) / 6), fs)                 (witness: ((fe2 + 20) / 6).astype(np.float32) on every row; its check (a) says the
                                                                   stored fs IS that expression on val, max|d| 0.0; here the stored bits are kept on non-nc rows)
    z2   = mean over configs of feed_rank_z(off, M, where(nc, noid_c, fe_c))[:, 1]     (witness, verbatim), used on every row of a feed that has an nc row;
           feeds without an nc row keep the stored fz bit for bit
    has2 = float32(~isnan(fe2))
"""
import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'POLARS_MAX_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[_k] = '12'
import sys

sys.dont_write_bytecode = True
import time

import numba as nb
import numpy as np
import polars as pl

ROOT = '<repo>'
HERE = os.path.dirname(os.path.abspath(__file__))
FLAT, F, FC, P, PT = f'{ROOT}/data/flat', f'{ROOT}/data/feats', f'{ROOT}/nn/feat_cache', f'{ROOT}/preds', f'{ROOT}/data/pretrain'
WS = f'{PT}/winscores'
HZ2 = f'{ROOT}/nn_x/w12/stale/hz2'
MSK, DAY, DAY0 = 3 * 3600, 86400, 20702
PUB_TS = 1790497920
CONFIGS = ['v3', 'v4']
WTAG, RTAG = 'w{k}r_{c}', '{c}t2'  # pretrain/assemble_train.py --wtag "w{k}r_{c}" --rtag "{c}t2" --prefix pre2_ (docs/02_SOLUTION.md row 9)
VAL_TAGS, VAL_CALIB = ['v3At2', 'v4At2'], ['v3t2', 'v4t2']  # CPU2 PRE_TAGS
N_TRAIN = 246745836
# base views: name -> (base dir of the view files, keep mask or None)
VIEWS = {'feats_n': (f'{HZ2}/feats', f'{HZ2}/feats/train/hz2_keep_rows.npy'),
         'feats_Ln': (f'{HZ2}/feats_L', f'{HZ2}/feats_L/train/hz2_keep_rows.npy'),
         'feats_hn': (f'{ROOT}/data/feats_new', None)}
NEW_FILES = ['pre2_hz_fe', 'pre2_hz_fe_s', 'pre2_hz_fe_z', 'pre2_hz_fe_has']
T0 = time.time()


def lg(*a):
    print(f'[{time.time() - T0:6.0f}s]', *a, flush=True)


def mm(path):
    return np.load(path, mmap_mode='r')


# src/stack_val.py::feed_rank_z, body copied verbatim (the function pretrain/assemble.py, assemble_train.py and build_pre2both call). Not imported from
# src/ because that module is compiled with cache=True (it would write into src/__pycache__) and imports lightgbm / sklearn at module level. Equality with the
# original is proven by P0 (stored train z reproduced on every window row) and P2 (the witness's val files reproduced).
@nb.njit(parallel=True)
def feed_rank_z(off, M, x):
    out = np.empty((len(x), 2), np.float32)
    for f in nb.prange(len(M)):
        o, m = off[f], M[f]
        v = x[o:o + m]
        r = np.argsort(np.argsort(v))
        mu = v.mean(); sd = v.std() + 1e-9
        for k in range(m):
            out[o + k, 0] = r[k] / (m - 1) if m > 1 else 0.5
            out[o + k, 1] = (v[k] - mu) / sd
    return out


def affine(x, ref):
    """pretrain/assemble.affine, verbatim (val-side alignment)."""
    mx, mr = (x > -2) & np.isfinite(x), (ref > -2) & np.isfinite(ref)
    a = ref[mr].astype(np.float64).std() / x[mx].astype(np.float64).std()
    b = ref[mr].astype(np.float64).mean() - a * x[mx].astype(np.float64).mean()
    fn = lambda v: np.where(v > -2, a * v + b, v).astype(np.float32)  # noqa: E731
    fn.slope = a
    return fn, (a, b)


def moments(x):
    """pretrain/assemble_train.moments, verbatim (train-window alignment)."""
    m = np.isfinite(x) & (x > -2)
    return x[m].astype(np.float64).mean(), x[m].astype(np.float64).std()


def bits_eq(a, b):
    """Row-wise bitwise equality of two float32 arrays (NaN payloads included)."""
    return np.ascontiguousarray(a, np.float32).view(np.int32) == np.ascontiguousarray(b, np.float32).view(np.int32)


def feed_any(mask, off, M):
    """per feed: any(mask) over its rows (off / M contiguous, covering mask)."""
    cs = np.concatenate([[0], np.cumsum(mask, dtype=np.int64)])
    return (cs[off + M] - cs[off]) > 0


def mix_rows(nc, fe, noid, fs, fz, fes_c, noids_c, off, M):
    """The row-wise construction (module docstring). Returns fe2, fs2, z2, has2 (float32) and the per-feed flag 'has an nc row'."""
    nc = np.asarray(nc, bool)
    fe2 = np.where(nc, noid, fe).astype(np.float32)
    fs2 = np.where(nc, ((fe2 + 20) / 6).astype(np.float32), fs).astype(np.float32)
    zc = [feed_rank_z(off, M, np.where(nc, n_, f_).astype(np.float32))[:, 1] for f_, n_ in zip(fes_c, noids_c)]
    z_new = np.mean(zc, 0).astype(np.float32)
    fnc = feed_any(nc, off, M)
    z2 = np.where(np.repeat(fnc, M), z_new, fz).astype(np.float32)
    has2 = (~np.isnan(fe2)).astype(np.float32)
    return fe2, fs2, z2, has2, fnc


def val_configs(family=''):
    """Per-config aligned fe / fe_noid on the val rows, exactly as build_pre2both builds them."""
    fes, noids = [], []
    for t, c in zip(VAL_TAGS, VAL_CALIB):
        L0 = lambda k: np.load(f'{P}/pre_{t}_{k}_val.npy').astype(np.float32)  # noqa: E731
        R = lambda k: np.load(f'{P}/pre_{c}_{k}_val.npy').astype(np.float32)  # noqa: E731
        ffe, _ = affine(L0('fe'), R('fe'))
        fno, _ = affine(L0('fe_noid'), R('fe_noid'))
        fes.append(ffe(L0('fe')))
        noids.append(ffe(L0('fe_noid')) if family == 'P' else fno(L0('fe_noid')))  # P (ADDENDUM 1): the fe key's own map on the raw no-ID score
    return fes, noids


def window_configs(k, family=''):
    """Per-config aligned fe / fe_noid on the rows of train window k, exactly as pretrain/assemble_train.window_feats builds them
    (the L lambda is its expression verbatim). Returns feeds, rows, off, M, fes_c, noids_c, calib."""
    f = pl.read_parquet(f'{FLAT}/train/feeds.parquet')
    M_all = f['M'].to_numpy().astype(np.int64)
    feeds = np.load(f'{PT}/win{k}/window_feeds.npy')
    rows = np.load(f'{PT}/win{k}/cand_rows.npy')
    s_rows = np.load(f'{PT}/win{k}/cand_rows_s60000.npy')
    pos = np.searchsorted(rows, s_rows)
    assert (rows[pos] == s_rows).all()
    M = M_all[feeds]
    off = np.concatenate([[0], np.cumsum(M)[:-1]]).astype(np.int64)
    coff = f['cand_off'].to_numpy().astype(np.int64)
    assert len(rows) == int(M.sum()) and (rows[off] == coff[feeds]).all() and (rows[off + M - 1] == coff[feeds] + M - 1).all()
    fes, noids, calib = [], [], {}
    for c in CONFIGS:
        L0 = lambda key: np.load(f'{WS}/pre_{WTAG.format(k=k, c=c)}_{key}_win{k}.npy').astype(np.float32)  # noqa: E731
        fns = {}
        for key in ('fe', 'fe_noid'):
            x = L0(key)
            mx, sx = moments(x[pos])
            mr, sr = moments(np.load(f'{WS}/pre_{RTAG.format(c=c)}_{key}_win{k}_s60000.npy'))
            aa = sr / sx; bb = mr - aa * mx
            fns[key] = (aa, bb)
            calib[f'{c}_{key}'] = (round(float(aa), 4), round(float(bb), 4))
        L = lambda key: np.where(L0(key) > -2, fns[key][0] * L0(key) + fns[key][1], L0(key)).astype(np.float32) if key in fns else L0(key)  # noqa: E731
        fes.append(L('fe'))
        if family == 'P':  # ADDENDUM 1: the fe key's own affine map applied to the raw no-ID score (same expression as L('fe') with the raw array swapped)
            noids.append(np.where(L0('fe_noid') > -2, fns['fe'][0] * L0('fe_noid') + fns['fe'][1], L0('fe_noid')).astype(np.float32))
        else:
            noids.append(L('fe_noid'))
    return feeds, rows, off, M, fes, noids, calib
