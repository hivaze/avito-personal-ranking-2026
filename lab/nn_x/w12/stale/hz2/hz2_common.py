"""hz2: shared constants / loaders (CPU only; read-only on every existing file; writes only under nn_x/w12/stale/hz2/).
Design: DECLARATION.txt. Day slots: slot = MSK day - 20702 (slot 0 = Sep 6, slot 14 = Sep 20 = dense cutoff of val / test, slot 23 = Sep 29)."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'POLARS_MAX_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[_k] = '12'
import sys
sys.dont_write_bytecode = True
import time
from pathlib import Path
import numpy as np
import polars as pl

ROOT = Path('<repo>')
FLAT = ROOT / 'data/flat'
FEATS = ROOT / 'data/feats'
FC = ROOT / 'nn/feat_cache'
FNEW = ROOT / 'data/feats_new'
HERE = ROOT / 'nn_x/w12/stale/hz2'
CACHE = HERE / 'cache'
LOGS = HERE / 'logs'
OUT = HERE / 'feats'
MSK = 3 * 3600
DAY = 86400
DAY0 = 20702                 # MSK day index of Sep 6 = slot 0
NSLOT = 24                   # Sep 6 .. Sep 29
NTR = 15                     # train slots 0..14 (Sep 6 .. Sep 20)
CSLOT = 14                   # Sep 20 = dense cutoff of val / test rows
PUB_TS = 1790497920          # public / private split of test (src/calib.py)
K_KNOTS = 64
F4_CAP = 63 / 64
DPAT = np.array([-14, -13, -12, -11, -10, -9, -8, -7, -6, -3, -2, -1], np.int64)   # dense days of a val / test row relative to its cutoff
PAT = np.array([2, 4, 6, 7, 8, 9], np.int64)                                        # sparse days relative to the cutoff
DENSE_SLOTS = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13], np.int64)
DENSE_BITS = int(sum(1 << int(s) for s in DENSE_SLOTS))
ALL_TR_BITS = (1 << NTR) - 1
H_SP = np.uint64(0xC2B2AE3D27D4EB4F)     # feat_new/hz_common.py sparse_flag (shared 1/37 sample)
H_LAG = np.uint64(0xD6E8FEB86659FD93)    # poplag lag hash (nn_x/w12/next/poplag/build_lag2.py)
H_PART = np.uint64(0x94D049BB133111EB)   # hz2: public-like / private-like part of a train feed
LUO = ['it_hist_all_luo', 'it_hist_et1_luo', 'it_hist_et2_luo', 'it_hist_et3_luo', 'it_hist_et4_luo']
SEQ = ['seq_cv_invd20', 'seq_cv_pop', 'seq_cv_norm']
XP = ['xp_imp_c50']
COLS9 = SEQ + LUO + XP
# stored name -> hz2 file name (same naming scheme as data/feats_new/train/it_hist_hz_*.npy)
HZ2_NAME = {'it_hist_all_luo': 'it_hist_hz_all', 'it_hist_et1_luo': 'it_hist_hz_et1', 'it_hist_et2_luo': 'it_hist_hz_et2',
            'it_hist_et3_luo': 'it_hist_hz_et3', 'it_hist_et4_luo': 'it_hist_hz_et4', 'seq_cv_invd20': 'seq_cv_hz_invd20',
            'seq_cv_pop': 'seq_cv_hz_pop', 'seq_cv_norm': 'seq_cv_hz_norm', 'xp_imp_c50': 'xp_imp_hz_c50'}
AGE_EDGES_D = np.array([0, 1, 3, 7, 10, 24, 76, np.inf])
AGE_NAMES = ['<1d', '1-3d', '3-7d', '7-10d', '10-24d', '24-76d', '>76d(+cap)']

T0 = time.time()


def lg(*a):
    print(f'[{time.time() - T0:6.0f}s]', *a, flush=True)


def feeds(split):
    return pl.read_parquet(FLAT / split / 'feeds.parquet')


def day_of(ts):
    return (np.asarray(ts, dtype=np.int64) + MSK) // DAY


def sparse_flag(feed_id):
    h = (np.asarray(feed_id).astype(np.uint64) * H_SP) >> np.uint64(40)
    return (h % np.uint64(37)) == 0


def hash_uniform(feed_id, mult):
    return ((np.asarray(feed_id).astype(np.uint64) * mult) >> np.uint64(11)).astype(np.float64) / float(1 << 53)


def test_lag_dist():
    """Lag counts of the test feeds by part (label-free): {'pub': (lags, counts), 'priv': (lags, counts)}, share of public feeds."""
    fte = feeds('test')
    tts = fte['timestamp'].to_numpy().astype(np.int64)
    tL = day_of(tts) - (DAY0 + CSLOT)
    tpub = tts < PUB_TS
    out = {}
    for nm, sel in (('pub', tpub), ('priv', ~tpub)):
        u, c = np.unique(tL[sel], return_counts=True)
        out[nm] = (u.astype(np.int64), c.astype(np.int64))
    return out, float(tpub.mean())


def assign_lags(feed_id):
    """(part_pub bool, L int64) per train feed: part by H_PART with the public feed share; lag inside the part by the poplag hash and cdf."""
    dist, ppub = test_lag_dist()
    uni = hash_uniform(feed_id, H_LAG)
    part_pub = hash_uniform(feed_id, H_PART) < ppub
    L = np.zeros(len(uni), np.int64)
    for nm, sel in (('pub', part_pub), ('priv', ~part_pub)):
        u, c = dist[nm]
        cdf = np.cumsum(c) / c.sum()
        L[sel] = u[np.minimum(np.searchsorted(cdf, uni[sel], side='right'), len(u) - 1)]
    return part_pub, L, dist, ppub


def holdout_feeds(ftr=None):
    """Exactly nn_x/w5/train.py --holdout 30000: p_random > 0.9 (NaN -> 0, M < 50 -> 0), M >= 50, rng 12345. Returns (hold fidx sorted, p_random)."""
    f = feeds('train') if ftr is None else ftr
    M = f['M'].to_numpy().astype(np.int64)
    pr = np.load(ROOT / 'eda/labels_position/out/train_prandom.npz')
    assert (pr['fidx'] == np.arange(len(f))).all()
    p = np.nan_to_num(pr['p_random'].astype(np.float32), nan=0.0)
    p[M < 50] = 0.0
    cand = np.where((p > 0.9) & (M >= 50))[0]
    hold = np.sort(np.random.default_rng(12345).choice(cand, min(30000, len(cand)), replace=False))
    return hold, p


def knots():
    return np.load(FEATS / 'life_f4_age_knots.npy')


def age_hours(x, kn):
    """Piecewise-linear f4 -> listing age in hours (copy of feat_new/hz_common.py::age_hours); NaN at the cap."""
    x = x.astype(np.float64)
    u = x * K_KNOTS
    k = np.clip(np.floor(u).astype(int), 1, K_KNOTS - 2)
    w = u - k
    a = kn[k] * (1 - w) + kn[k + 1] * w
    return np.where(x < F4_CAP, a, np.nan)


def age_bucket(age_h):
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


def cload(name, mmap=True):
    a = np.load(CACHE / f'{name}.npy', mmap_mode='r' if mmap else None)
    return np.asarray(a)
