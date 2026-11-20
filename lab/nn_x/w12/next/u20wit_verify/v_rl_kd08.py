"""u20wit_verify: (a) short witness with the val / test domain token: my logits vs the study's rl cache on the sample, then B-A on all W0S from the cache with my NDCG;
(b) one-seed look at the 'more teacher weight' control named by the lead: kd08w192_s951 - kd05w192_s951, member, cross-user columns (stored GPU files)."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS', 'POLARS_MAX_THREADS'): os.environ[k] = '4'
import numpy as np, polars as pl
from pathlib import Path
ROOT = Path('<repo>')
src = open(ROOT / 'nn_x/w12/next/u20wit_verify/v_cmp.py').read().split("f = pl.read_parquet")[0]; exec(src)
W = ROOT / 'nn_x/w5/preds'; ST = ROOT / 'nn_x/w12/next/u20wit/cache'; C = ROOT / 'nn_x/w12/next/u20wit_verify/cache'
ft = pl.read_parquet(ROOT / 'data/flat/train/feeds.parquet', columns=['user_id', 'M', 'cand_off']); Mtr = ft['M'].to_numpy().astype(np.int64); otr = ft['cand_off'].to_numpy().astype(np.int64); uid = ft['user_id'].to_numpy().astype(np.int64)
ytr = np.load(ROOT / 'data/flat/train/cand_target.npy', mmap_mode='r')
sm = np.load(ST / 'sub_meta.npz'); subf = sm['src_fidx']; sub_rows = np.load(ST / 'sub/rows.npy'); Msub = Mtr[subf]; osub = np.concatenate([[0], np.cumsum(Msub)[:-1]]); sid = sm['set_id']
y_sub = np.asarray(ytr[sub_rows]).astype(np.float32)
S = np.load(C / 'sample.npz'); fids = np.concatenate([S[k] for k in ('W0S', 'W0L', 'H0', 'XH')]); Ms = Mtr[fids]
rows_s = np.concatenate([np.arange(o, o + k) for o, k in zip(otr[fids], Ms)]); n_w0s = int(Mtr[S['W0S']].sum())
pos = np.searchsorted(sub_rows, rows_s[:n_w0s]); assert (sub_rows[pos] == rows_s[:n_w0s]).all()
print('(a) val / test domain token on the short witness')
nd = {}
for a in 'ab':
    for s in (951, 952):
        v = f'w5_e10_u20{a}_s{s}'; c = np.load(ST / f'sc/{v}__rl.npy'); mine = np.load(C / f'my_{v}__rl.npy')[:n_w0s]; tr = np.load(C / f'my_{v}.npy')[:n_w0s]
        print(f'  {v}: W0S sample rows {n_w0s}: max |my rl - cache rl| {np.abs(mine - c[pos]).max():.2e}; mean |my rl - my tr| {np.abs(mine - tr).mean():.3f} (the token changes the logits)')
        ok = np.isfinite(c[np.repeat(sid == 0, Msub)]).all(); assert ok
        nd[a, s] = my_ndcg(np.nan_to_num(c).astype(np.float32), y_sub, osub, Msub)
mk = sid == 0; d = np.mean([nd['b', s][mk] - nd['a', s][mk] for s in (951, 952)], 0)
print(f'  W0S (n {int(mk.sum())}) B-A member with the val / test token: {ms(d)} (per seed ' + ', '.join(f'{np.mean(nd["b", s][mk] - nd["a", s][mk]):+.5f}' for s in (951, 952)) + ')')
print('(b) kd08w192_s951 - kd05w192_s951 (teacher share 0.8 vs 0.5 on every feed; ONE seed; member)')
pr = np.load(ROOT / 'eda/labels_position/out/train_prandom.npz'); pold = np.nan_to_num(pr['p_random'].astype(np.float32), nan=0.0); pold[Mtr < 50] = 0.0
hold = np.sort(np.random.default_rng(12345).choice(np.where((pold > 0.9) & (Mtr >= 50))[0], 30000, replace=False))
Mh, oh = np.load(W / 'holdout/holdout_meta.npy'); Mh = Mh.astype(np.int64); oh = oh.astype(np.int64); yh = np.load(W / 'holdout/holdout_y.npy').astype(np.float32)
cuh = fold_of(uid, np.arange(len(uid)))[hold] != 0
fv = pl.read_parquet(ROOT / 'data/flat/val/feeds.parquet', columns=['user_id', 'M', 'cand_off']); Mv = fv['M'].to_numpy().astype(np.int64); ov = fv['cand_off'].to_numpy().astype(np.int64)
yv = np.load(ROOT / 'data/flat/val/cand_target.npy').astype(np.float32)
for suf in ('', '@last'):
    try:
        h = [my_ndcg(np.load(W / f'holdout/nn_w5_e10_{a}_s951{suf}_holdout.npy').astype(np.float32), yh, oh, Mh) for a in ('kd08w192', 'kd05w192')]
        v = [my_ndcg(np.load(W / f'nn_w5_e10_{a}_s951{suf}_trainonly_val.npy').astype(np.float32), yv, ov, Mv) for a in ('kd08w192', 'kd05w192')]
        print(f'  {"selected" if suf == "" else "@last"}: val all (n {len(Mv)}) {ms(v[0] - v[1])} | holdout all (n 30000) {ms(h[0] - h[1])} | holdout cross-user (n {int(cuh.sum())}) {ms((h[0] - h[1])[cuh])}')
    except FileNotFoundError as e: print('  missing', e)
