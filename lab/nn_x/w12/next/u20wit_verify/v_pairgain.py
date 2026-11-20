"""u20wit_verify: is 'A gains +0.00159 from 2-pooling, B / C +0.00090' inside the same-recipe distribution of 2-pool gains?
Pair pooling gain = NDCG(mean z of 2 members) - mean member NDCG, cross-user holdout / cross-user val, all 28 pairs of kd05w192 s951-958 and of kd05t25w192 s951-958."""
import os, itertools
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS', 'POLARS_MAX_THREADS'): os.environ[k] = '6'
import sys, numpy as np, polars as pl
sys.path.insert(0, '<repo>/nn_x/w12/next/u20wit_verify')
from pathlib import Path
import importlib.util
ROOT = Path('<repo>'); W = ROOT / 'nn_x/w5/preds'; FLAT = ROOT / 'data/flat'
src = open(ROOT / 'nn_x/w12/next/u20wit_verify/v_cmp.py').read().split("f = pl.read_parquet")[0]   # my own njit helpers (my_ndcg, my_z, fold_of)
exec(src); FLAT = ROOT / 'data/flat'; W = ROOT / 'nn_x/w5/preds'
ft = pl.read_parquet(FLAT / 'train/feeds.parquet', columns=['user_id', 'M']); Mtr = ft['M'].to_numpy().astype(np.int64); uid = ft['user_id'].to_numpy().astype(np.int64)
pr = np.load(ROOT / 'eda/labels_position/out/train_prandom.npz'); pold = np.nan_to_num(pr['p_random'].astype(np.float32), nan=0.0); pold[Mtr < 50] = 0.0
hold = np.sort(np.random.default_rng(12345).choice(np.where((pold > 0.9) & (Mtr >= 50))[0], 30000, replace=False))
Mh, oh = np.load(W / 'holdout/holdout_meta.npy'); Mh = Mh.astype(np.int64); oh = oh.astype(np.int64); yh = np.load(W / 'holdout/holdout_y.npy').astype(np.float32)
cuh = fold_of(uid, np.arange(len(uid)))[hold] != 0
fv = pl.read_parquet(FLAT / 'val/feeds.parquet', columns=['user_id', 'M', 'cand_off']); Mv = fv['M'].to_numpy().astype(np.int64); ov = fv['cand_off'].to_numpy().astype(np.int64); uv = fv['user_id'].to_numpy().astype(np.int64)
yv = np.load(FLAT / 'val/cand_target.npy').astype(np.float32)
cuv = ~(np.isin(uv, np.unique(uid)) & (uv != 0) & (fold_of(uv, np.zeros(len(uv), np.int64) - 1) == 0))
SP = {'hold': (yh, oh, Mh, cuh, lambda v: W / f'holdout/nn_{v}_holdout.npy'), 'val': (yv, ov, Mv, cuv, lambda v: W / f'nn_{v}_trainonly_val.npy')}
def gains(variants, sp):
    y, off, M, mk, pth = SP[sp]; nd = []; zz = []
    for v in variants:
        x = np.load(pth(v)).astype(np.float32); nd.append(my_ndcg(x, y, off, M)[mk].mean()); zz.append(my_z(x, off, M))
    g = []
    for i, j in itertools.combinations(range(len(variants)), 2):
        g.append(my_ndcg(((zz[i] + zz[j]) / 2).astype(np.float32), y, off, M)[mk].mean() - (nd[i] + nd[j]) / 2)
    return np.array(g), np.array(nd)
for sp in ('hold', 'val'):
    for arm in ('kd05w192', 'kd05t25w192'):
        g, nd = gains([f'w5_e10_{arm}_s{s}' for s in range(951, 959)], sp)
        print(f'{sp} cross-user | {arm} x8: 2-pool gain over 28 pairs mean {g.mean():+.5f} sd {g.std(ddof=1):.5f} min {g.min():+.5f} max {g.max():+.5f} | member mean {nd.mean():.5f}', flush=True)
    for a in 'abc':
        g, nd = gains([f'w5_e10_u20{a}_s{s}' for s in (951, 952)], sp)
        print(f'{sp} cross-user | u20{a} x2: 2-pool gain {g[0]:+.5f} | members {nd[0]:.5f} {nd[1]:.5f}', flush=True)
