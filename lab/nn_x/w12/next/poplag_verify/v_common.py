"""poplag_verify: reviewer's own helpers (CPU only). Nothing here imports the study's scripts (nn_x/w12/next/poplag/*.py)."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
NT = str(min(12, int(os.environ.get('PV_THREADS', '12'))))   # at most 12; light helper jobs set PV_THREADS=4
for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS', 'POLARS_MAX_THREADS'):
    os.environ[_k] = NT
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import numpy as np
import polars as pl

ROOT = Path('<repo>')
FLAT = ROOT / 'data/flat'
FC = ROOT / 'nn/feat_cache'
STUDY = ROOT / 'nn_x/w12/next/poplag/cache'          # the study's cached arrays (read-only for me)
HERE = ROOT / 'nn_x/w12/next/poplag_verify'
MYC = HERE / 'cache'
TZ = 3 * 3600                                        # MSK
DAY = 86400
SEP6 = 20702                                         # (ts + TZ) // DAY of 2026-09-06 MSK (checked in v_gate.py against the data)
SEP20 = 20716
PUB_TS = 1790497920                                  # public / private split of test (src/calib.py), checked in v_gate.py
W8 = [f'w5_e10_{a}_s{s}' for a in ('kd05w192', 'kd05t25w192') for s in (951, 952, 953, 954)]
P4 = [f'abl_e10_pop1kd05t25_s{s}' for s in (951, 952, 953, 954)]


def stored_path(v):
    base = 'nn_x/w5/preds' if v.startswith('w5_') else 'nn_x/abl/preds'
    return ROOT / base / 'holdout' / f'nn_{v}_holdout.npy'


def my_holdout():
    """30,000 holdout feeds, re-derived from nn_x/w5/train.py l.285-292 (read by me): p_random > 0.9 after NaN -> 0 and M < 50 -> 0; rng 12345."""
    f = pl.read_parquet(FLAT / 'train/feeds.parquet')
    M = f['M'].to_numpy().astype(np.int64)
    pr = np.load(ROOT / 'eda/labels_position/out/train_prandom.npz')
    assert (pr['fidx'] == np.arange(len(f))).all()
    p = np.nan_to_num(pr['p_random'].astype(np.float32), nan=0.0)
    p[M < 50] = 0.0
    cand = np.where((p > 0.9) & (M >= 50))[0]
    hold = np.sort(np.random.default_rng(12345).choice(cand, min(30000, len(cand)), replace=False))
    return f, hold, p


def ndcg_feed(s, y, off, M):
    """Per-feed full-list NDCG, binary gain, pessimistic ties (negatives first among equal scores), no-positive feed = 1. Own implementation."""
    s = np.asarray(s, np.float64); y = np.asarray(y, np.float64)
    n = len(M); fid = np.repeat(np.arange(n), M)
    o = np.lexsort((y, -s, fid))
    pos_in_feed = np.arange(len(s)) - off[fid]           # rows are stored feed by feed, so the sorted order keeps the feed blocks
    w = 1.0 / np.log2(pos_in_feed + 2.0)
    dcg = np.bincount(fid, weights=y[o] * w, minlength=n)
    npos = np.bincount(fid, weights=y, minlength=n)
    ideal = np.bincount(fid, weights=w * (pos_in_feed < npos[fid]), minlength=n)
    out = np.ones(n)
    k = ideal > 0
    out[k] = dcg[k] / ideal[k]
    return out


def ndcg_feed_slow(s, y, off, M, feeds):
    """Plain per-feed loop, for a cross-check of ndcg_feed."""
    out = []
    for f in feeds:
        a = np.asarray(s[off[f]:off[f] + M[f]], np.float64); b = np.asarray(y[off[f]:off[f] + M[f]], np.float64)
        if b.sum() == 0:
            out.append(1.0); continue
        idx = sorted(range(len(a)), key=lambda i: (-a[i], b[i]))
        dcg = sum(b[i] / np.log2(r + 2.0) for r, i in enumerate(idx))
        idcg = sum(1.0 / np.log2(r + 2.0) for r in range(int(b.sum())))
        out.append(dcg / idcg)
    return np.array(out)


def zfeed(x, off, M):
    """Per-feed z as src/stack_val.py feed_rank_z[:, 1]: (x - mean) / (population sd + 1e-9), float32 result."""
    x = np.asarray(x, np.float32).astype(np.float64)
    fid = np.repeat(np.arange(len(M)), M)
    mu = np.add.reduceat(x, off) / M
    var = np.add.reduceat((x - mu[fid]) ** 2, off) / M
    return ((x - mu[fid]) / (np.sqrt(var)[fid] + 1e-9)).astype(np.float32)


def mse(d):
    d = np.asarray(d, np.float64)
    return d.mean(), d.std(ddof=1) / np.sqrt(len(d))


def f5(d):
    m, s = mse(d)
    return f'{m:+.5f} +- {s:.5f}'
