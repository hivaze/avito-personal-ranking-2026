"""a02: the placebo of a01 with 300 draws and other seeds (the NDCG part is dropped). Critic re-read of lane A (click strength), CPU only, <= 12 threads, read-only on every file outside reports/NEXT_1116/critic/.
One re-read of the failing gates g1 / g2 / g4 on the stored arrays of the lane, with my own code for everything but the lens of record:
  (a) the entered value recomputed from the stored item statistic and the stored exposure with my own strata / centring code (pandas), against the stored array;
  (b) cz of the stored array under the null model of record (condlib.Null: pool + control block + nx3), my own call;
  (c) an independent Wald read under a different null (pool only): lens.fit + user-clustered sandwich;
  (d) my own exposure-preserving placebo: the item statistic permuted inside (feed, exposure bin) with my own permutation code and seeds, re-centred, scored (60 draws);
  (e) NDCG with the organisers' metric copy src/metrics.ndcg_per_feed (NOT the lens kernel): in-sample oracle weight of the form on top of the fitted null score
      (an upper bound of what a nested fit can return), on T all feeds and on the lag-view holdout feeds.
usage: a01_reread.py   -> out/a02_placebo300.txt"""
import os, sys, time
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS', 'POLARS_MAX_THREADS'):
    os.environ[_k] = '12'
CR = '<repo>/reports/NEXT_1116/critic'
os.environ['NUMBA_CACHE_DIR'] = CR + '/out/numba_cache'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True
import numba as nb
import numpy as np
import pandas as pd
from pathlib import Path
nb.set_num_threads(12)
sys.path.insert(0, '<repo>/reports/INPUTS_1116/residual_rows'); sys.path.insert(0, '<repo>/src')
_argv = sys.argv; sys.argv = [sys.argv[0], 'val', 'TO16']
import lens, condlib
sys.argv = _argv
nb.set_num_threads(12)
from metrics import ndcg_per_feed

ROOT = Path('<repo>'); ARR = ROOT / 'reports/NEXT_1116/clickstrength/out/arr'; NX3 = ROOT / 'nn_x/w12/nx3'
T0 = time.time(); LOG = open(CR + '/out/a02_placebo300.txt', 'x')


def P(*a):
    s = f'[{time.time() - T0:6.0f}s] ' + ' '.join(str(x) for x in a)
    print(s, flush=True); LOG.write(s + '\n'); LOG.flush()


def my_strata(expo, view, fid):
    """8 within-feed bins of the average-tie percentile rank of the exposure over the rows with a view (my code, pandas)"""
    s = pd.Series(np.where(view, expo, np.nan))
    g = s.groupby(fid)
    r = g.rank(method='average') - 1.0; c = g.transform('count')
    pr = np.where(c.values >= 2, r.values / np.maximum(c.values - 1.0, 1.0), 0.5)     # lens.feed_rk: fewer than 2 present rows -> 0 (+0.5)
    pr = np.where(view, pr, 0.5)
    return np.minimum(np.floor(8 * pr).astype(np.int64), 7)


def my_centre(x, key, view):
    s = pd.Series(np.where(view, x, np.nan)); m = s.groupby(key).transform('mean').values
    return np.where(view, x - m, 0.0)


def my_perm(x, key, rng):
    """values of x permuted among the rows that share key"""
    o1 = np.argsort(key, kind='stable'); o2 = np.lexsort((rng.random(len(key)), key))
    out = np.empty_like(x); out[o1] = x[o2]
    return out


ND = 300
for wit, sp, q in (('T', 'val', 'val_twin'), ('H', 'hold', 'hold')):
    D = lens.load_split(sp, 'TO16'); condlib.sp = sp; cn, cl = condlib.control_block(D); off, M = D['off'], D['M']; n = D['n']
    fid = np.repeat(np.arange(n), M)
    if wit == 'H':
        rows = D['rows']
        nz = np.nan_to_num(np.asarray(np.load(NX3 / 'out/files/train/nx_rp3_z.npy', mmap_mode='r')[rows]).astype(np.float32), nan=0.0)
        nr = np.nan_to_num(np.asarray(np.load(NX3 / 'out/files/train/nx_rp3_rk.npy', mmap_mode='r')[rows]).astype(np.float32), nan=0.0)
    else:
        nz = np.load(NX3 / 'staleval/nx_rp3_z.npy').astype(np.float32); nr = np.load(NX3 / 'staleval/nx_rp3_rk.npy').astype(np.float32)
    N = condlib.Null(D, cn + ['nx_rp3_z', 'nx_rp3_rk'], cl + [nz, nr])
    s0 = (N.X @ N.r['theta'].astype(np.float32)).astype(np.float64)
    nd0 = ndcg_per_feed(s0, D['y'], off, M)
    P(f'=== witness {wit} ({q}): feeds {n}, rows {len(fid)}; null model = pool + {len(cn)} control columns + nx3; NDCG pool {ndcg_per_feed(D["base"], D["y"], off, M).mean():.5f}, fitted null score {nd0.mean():.5f} (in-sample)')
    imp = np.load(ARR / f'{q}__imp3.npy').astype(np.float64); lag = np.load(ARR / f'{q}__lag_row.npy').astype(np.int64)
    view_f = np.minimum.reduceat(lag, off) > 0
    rng = np.random.default_rng(771100 + (1 if wit == 'T' else 2))
    for fn, k in (('F1 cs3a_c', 'cs3a'), ('F2 cs3x_c', 'cs3x')):
        raw = np.load(ARR / f'{q}__{k}_x.npy').astype(np.float64); view = ~np.isnan(raw)
        assert (view == (lag > 0)).all()
        stored = np.nan_to_num(np.load(ARR / f'{q}__{k}_c.npy').astype(np.float64))
        st = my_strata(imp, view, fid); key = fid * 8 + st
        mine = my_centre(np.nan_to_num(raw), key, view)
        dmax = float(np.abs(mine - stored).max()); sd = float(stored[view].std())
        sc = 1.0 / sd
        x = (stored * sc).astype(np.float32)
        r = N.score(x); read = float(r['cz'])
        r_mine = float(N.score((mine * sc).astype(np.float32))['cz'])
        # (c) pool-only null, Wald with the user-clustered sandwich
        f1 = lens.fit(D, [x]); zr, zm = lens.robust_z(D, f1)
        P(f'  {fn}: my recomputation of the entered value against the stored array: max abs difference {dmax:.2e} (sd of the value {sd:.4f}); cz stored {read:+.3f}, cz of my recomputation {r_mine:+.3f}; '
          f'nats per feed {float(r["dnats"]):.6f}; Wald z under the pool-only null (no control block, no nx3) {zr[1]:+.2f}')
        # (d) my placebo
        pz = []
        for s in range(ND):
            xp = my_perm(np.nan_to_num(raw), key, rng); xe = (my_centre(xp, key, view) * sc).astype(np.float32); pz.append(float(N.score(xe)['cz']))
        pz = np.array(pz)
        P(f'      my placebo (item statistic permuted inside feed x exposure bin, re-centred; {ND} draws): mean {pz.mean():+.2f}, sd {pz.std():.2f}, min {pz.min():+.2f}, max {pz.max():+.2f}; '
          f'(read - mean) / sd = {(read - pz.mean()) / pz.std():+.2f} (bar 2.5); draws at or above the read {int((pz >= read).sum())} of {ND}; draws with |cz| >= read {int((np.abs(pz) >= abs(read)).sum())} of {ND}')
    del N, D
P('DONE')
