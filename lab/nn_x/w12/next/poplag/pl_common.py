"""poplag: shared constants / loaders (CPU only, read-only on every existing file).
KEY = poplag (LAGGED-POPULARITY HOLDOUT TEST, nn_x/w5/logs/E_kd_predeclared.txt 2026-11-14T22:17:17Z)."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'POLARS_MAX_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[_k] = '12'
import json
from pathlib import Path
import numpy as np
import polars as pl

ROOT = Path('<repo>')
FLAT = ROOT / 'data/flat'
FEATS = ROOT / 'data/feats'
FC = ROOT / 'nn/feat_cache'
HERE = ROOT / 'nn_x/w12/next/poplag'
CACHE = HERE / 'cache'
LOGS = HERE / 'logs'
MSK = 3 * 3600
DAY = 86400
CUT_VT = 1789851600          # 2026-09-20 00:00 MSK = end of the dense (train) data
DATA0 = CUT_VT - 14 * DAY    # 2026-09-06 00:00 MSK = start of the dense data
PUB_TS = 1790497920          # public / private split of test (src/calib.py)
K_KNOTS = 64
F4_CAP = 63 / 64
SPARSE_PAT = (2, 4, 6, 7, 8, 9)   # day offsets (from the dense cutoff, Sep 20) of the days that carry val / test snapshots
W8 = [f'w5_e10_{a}_s{s}' for a in ('kd05w192', 'kd05t25w192') for s in (951, 952, 953, 954)]
POP1KD = [f'abl_e10_pop1kd05t25_s{s}' for s in (951, 952, 953, 954)]
LUO = ['it_hist_all_luo', 'it_hist_et1_luo', 'it_hist_et2_luo', 'it_hist_et3_luo', 'it_hist_et4_luo']
SEQ = ['seq_cv_invd20', 'seq_cv_pop', 'seq_cv_norm']
XP = ['xp_imp_c50']
LAGCOLS = SEQ + LUO + XP


def ckpt_path(v):
    return ROOT / ('nn_x/w5/ckpt' if v.startswith('w5_') else 'nn_x/abl/ckpt') / f'{v}.pt'


def stored_holdout(v):
    return ROOT / ('nn_x/w5/preds' if v.startswith('w5_') else 'nn_x/abl/preds') / 'holdout' / f'nn_{v}_holdout.npy'


def feeds(split):
    return pl.read_parquet(FLAT / split / 'feeds.parquet')


def day_of(ts):
    return (np.asarray(ts, dtype=np.int64) + MSK) // DAY


def holdout_feeds(ftr=None):
    """Exactly nn_x/w5/train.py --holdout 30000 (l.284-292): p_random > 0.9 (NaN -> 0, M < 50 -> 0), M >= 50, rng 12345."""
    f = feeds('train') if ftr is None else ftr
    M = f['M'].to_numpy().astype(np.int64)
    pr = np.load(ROOT / 'eda/labels_position/out/train_prandom.npz')
    assert (pr['fidx'] == np.arange(len(f))).all()
    p = np.nan_to_num(pr['p_random'].astype(np.float32), nan=0.0)
    p[M < 50] = 0.0
    cand = np.where((p > 0.9) & (M >= 50))[0]
    hold = np.sort(np.random.default_rng(12345).choice(cand, min(30000, len(cand)), replace=False))
    return hold, p


def rows_of(off, M):
    return np.repeat(off - np.concatenate([[0], np.cumsum(M)[:-1]]), M) + np.arange(int(M.sum()))


def age_hours(x, knots):
    """Piecewise-linear f4 -> listing age in hours (copy of feat_new/hz_common.py::age_hours); NaN at the cap."""
    x = x.astype(np.float64)
    u = x * K_KNOTS
    k = np.clip(np.floor(u).astype(int), 1, K_KNOTS - 2)
    w = u - k
    a = knots[k] * (1 - w) + knots[k + 1] * w
    return np.where(x < F4_CAP, a, np.nan)


def knots():
    return np.load(FEATS / 'life_f4_age_knots.npy')


AGE_EDGES_D = np.array([0, 1, 3, 7, 10, 24, 76, np.inf])
AGE_NAMES = ['<1d', '1-3d', '3-7d', '7-10d', '10-24d', '24-76d', '>76d(+cap)']


def age_bucket(age_h):
    """Bucket index 0..6 (nn_x/w5/hzcd/preflight.py buckets; capped f4 = NaN age -> last bucket)."""
    d = np.where(np.isnan(age_h), 1e9, age_h / 24.0)
    return np.clip(np.searchsorted(AGE_EDGES_D, d, side='right') - 1, 0, 6)


def xt_transform(name, x):
    """nn_x/w5/data.py SplitData.attach_xt transform of one raw column (same branches, same order), float32 numpy."""
    x = np.asarray(x).astype(np.float32)
    if name.startswith('has:') or name.endswith('_rk'):
        pass
    elif name.endswith('_z'):
        x = np.clip(x, -5, 5)
    elif name.startswith(('seq_cv', 'it_hist')):
        x = np.log1p(np.maximum(x, 0))
    else:
        x = np.clip(x, -10, 10)
    return np.nan_to_num(x, nan=0.0)


def se(d):
    d = np.asarray(d, np.float64)
    return float(d.std(ddof=1) / np.sqrt(len(d)))


def fmt(d):
    d = np.asarray(d, np.float64)
    return f'{d.mean():+.5f} +- {se(d):.5f}'
