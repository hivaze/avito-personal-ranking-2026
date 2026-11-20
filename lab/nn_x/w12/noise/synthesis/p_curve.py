"""Within-feed relative click rate by display slot (non-last rows; observed / expected under the feed's own non-last rate)
for random-order train pools of different purity, M >= 50.  No model.  -> out/p_curve.txt"""
from pathlib import Path
import numpy as np, polars as pl

ROOT = Path('<repo>'); FLAT = ROOT / 'artifacts/data/flat/train'; OUT = ROOT / 'lab/nn_x/w12/noise/synthesis/out'
f = pl.read_parquet(FLAT / 'feeds.parquet'); M = f['M'].to_numpy().astype(np.int64); off = f['cand_off'].to_numpy().astype(np.int64)
tgt = np.load(FLAT / 'cand_target.npy', mmap_mode='r'); pos = np.load(FLAT / 'cand_pos.npy', mmap_mode='r')
old = np.nan_to_num(np.load(ROOT / 'artifacts/exploration/train_prandom.npz')['p_random'].astype(np.float32), nan=-1)
v2 = np.load(ROOT / 'lab/eda/labels_position/out/train_prandom_v2.npz'); pr, ps, z = v2['p_random'], v2['p_set'], v2['z']
BINS = [(0, 0), (1, 1), (2, 2), (3, 4), (5, 9), (10, 19), (20, 49), (50, 99), (100, 199), (200, 10**6)]
L = M >= 50
pools = {'old p>0.9': L & (old > 0.9), 'old p>0.9 & v2 p>0.9': L & (old > 0.9) & (pr > 0.9), 'v2 p>0.9': L & (pr > 0.9),
         'v2 p>0.9 & p_set>0.9': L & (pr > 0.9) & (ps > 0.9), 'v2 p>0.9 & |z|<1': L & (pr > 0.9) & (np.abs(z) < 1),
         'old p>0.9 & v2 p<0.5 (contaminants)': L & (old > 0.9) & (pr < 0.5), 'production old p<0.05': L & (old >= 0) & (old < 0.05)}
rng = np.random.default_rng(0)
lines = [f'bins {BINS}']
for name, m in pools.items():
    fids = np.flatnonzero(m)
    if len(fids) > 100000: fids = np.sort(rng.choice(fids, 100000, replace=False))
    rows = np.concatenate([np.arange(off[i], off[i] + M[i]) for i in fids]); fidx = np.repeat(np.arange(len(fids)), M[fids])
    order = np.lexsort((np.asarray(pos[rows]), fidx)); start = np.concatenate([[0], np.cumsum(M[fids])[:-1]])
    rk = np.empty(len(rows), np.int64); rk[order] = np.arange(len(rows)) - np.repeat(start, M[fids])
    last = rk == np.repeat(M[fids], M[fids]) - 1; y = tgt[rows].astype(np.float64)
    rate = np.add.reduceat(y * ~last, start) / (M[fids] - 1); er = np.repeat(rate, M[fids])
    e = [y[(rk >= lo) & (rk <= hi) & ~last].sum() / er[(rk >= lo) & (rk <= hi) & ~last].sum() for lo, hi in BINS]
    # bootstrap-free SE of slot 0 ratio: Poisson approx
    n0 = y[(rk == 0) & ~last].sum()
    lines.append(f'{name:38s} feeds {int(m.sum()):>9,}: e = {np.round(e, 2).tolist()}  (slot-0 clicks {int(n0):,}; last-item click rate {y[last].mean():.3f})')
(OUT / 'p_curve.txt').write_text('\n'.join(lines) + '\n'); print('\n'.join(lines))
