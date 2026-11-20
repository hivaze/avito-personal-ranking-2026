"""Critic: SE of 'slot examination on top of the last-item oracle' for SHORT (M<50) random-order feeds, which the
synthesis' holdout number (M>=50 only) does not cover.  Same estimator and inputs as synthesis/p_oracle.py (scores in-sample,
e from disjoint feeds).  Usage: python c_short_oracle.py -> out/c_short_oracle.txt"""
import sys
from pathlib import Path
import numpy as np, polars as pl
sys.path.insert(0, '<repo>')
from avito_ranker.metrics import ndcg_per_feed
ROOT = Path('<repo>'); FLAT = ROOT / 'artifacts/data/flat/train'
SYN = ROOT / 'lab/nn_x/w12/noise/synthesis/out'; OUT = ROOT / 'lab/nn_x/w12/noise/critic/out'
f = pl.read_parquet(FLAT / 'feeds.parquet'); M = f['M'].to_numpy().astype(np.int64); off = f['cand_off'].to_numpy().astype(np.int64)
tgt = np.load(FLAT / 'cand_target.npy', mmap_mode='r'); pos = np.load(FLAT / 'cand_pos.npy', mmap_mode='r')
p_v2 = np.load(ROOT / 'lab/eda/labels_position/out/train_prandom_v2.npz')['p_random'].astype(np.float32)
pool = np.flatnonzero((M < 50) & (p_v2 > 0.9))
BINS = [(0, 0), (1, 1), (2, 2), (3, 4), (5, 9), (10, 19), (20, 49)]; NB = len(BINS)
def layout(fids):
    rows = np.concatenate([np.arange(off[i], off[i] + M[i]) for i in fids]); fidx = np.repeat(np.arange(len(fids)), M[fids])
    order = np.lexsort((np.asarray(pos[rows]), fidx)); start = np.concatenate([[0], np.cumsum(M[fids])[:-1]])
    rk = np.empty(len(rows), np.int64); rk[order] = np.arange(len(rows)) - np.repeat(start, M[fids])
    b = np.full(len(rows), -1)
    for k, (lo, hi) in enumerate(BINS): b[(rk >= lo) & (rk <= hi)] = k
    return rows, b, rk == np.repeat(M[fids], M[fids]) - 1, start
z = np.load(SYN / 'p_token_w5_fd10_kd05w192_s951_rand_M<50.npz'); ids = z['ids']; s1 = z['s1'].astype(np.float64)
rows, b, last, start = layout(ids); y = np.asarray(tgt[rows], np.float64)
ref = np.setdiff1d(pool, ids); rr, rb, rl, rst = layout(ref); ry = np.asarray(tgt[rr], np.float64)
fr = np.add.reduceat(ry * ~rl, rst) / (M[ref] - 1); ex = np.repeat(fr, M[ref])
e = np.array([ry[(rb == k) & ~rl].sum() / ex[(rb == k) & ~rl].sum() for k in range(NB)])
lift = np.log(ry[rl].mean() / ry[~rl].mean()); st = start; Mi = M[ids]
lo = ndcg_per_feed(s1 + np.where(last, lift, 0.0), y, st, Mi); bo = ndcg_per_feed(s1 + np.log(e)[b] + np.where(last, lift, 0.0), y, st, Mi)
base = ndcg_per_feed(s1, y, st, Mi); so = ndcg_per_feed(s1 + np.log(e)[b], y, st, Mi)
d = bo - lo; d2 = so - base
L = [f'rand M<50 (v2 p>0.9), {len(ids)} scored feeds (seen in training), e from {len(ref):,} disjoint feeds: e {np.round(e,3).tolist()}',
     f'  slot only: {d2.mean():+.5f} ± {d2.std(ddof=1)/np.sqrt(len(d2)):.5f}; slot on top of last-item oracle: {d.mean():+.5f} ± {d.std(ddof=1)/np.sqrt(len(d)):.5f}']
(OUT / 'c_short_oracle.txt').write_text('\n'.join(L) + '\n'); print('\n'.join(L))
