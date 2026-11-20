"""u20wit_verify: sensitivity of the decision statistic to the tie convention (stored predictions come from bf16 forwards; the declared read uses pessimistic ties).
B - A member, cross-user holdout (n 24,026), under pessimistic (project metric), optimistic and file-order ties; share of positives that are tied."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS', 'POLARS_MAX_THREADS'): os.environ[k] = '4'
import numpy as np, polars as pl, numba as nb
from pathlib import Path
ROOT = Path('<repo>'); W = ROOT / 'nn_x/w5/preds'

@nb.njit(parallel=True, cache=False)
def ndcg_mode(s, y, off, M, mode):   # mode 0 pessimistic, 1 optimistic, 2 file order (earlier candidate first)
    n = len(M); out = np.ones(n, np.float64); tied = np.zeros(n, np.int64); npos_all = np.zeros(n, np.int64)
    for f in nb.prange(n):
        o = off[f]; m = M[f]; npos = 0
        for k in range(m):
            if y[o + k] > 0: npos += 1
        npos_all[f] = npos
        if npos == 0: continue
        dcg = 0.0
        for k in range(m):
            if y[o + k] > 0:
                sk = s[o + k]; r = 0; t = 0
                for j in range(m):
                    if j == k: continue
                    sj = s[o + j]
                    if sj > sk: r += 1
                    elif sj == sk:
                        t += 1
                        if mode == 0:
                            if y[o + j] <= 0 or j < k: r += 1
                        elif mode == 1:
                            if y[o + j] > 0 and j < k: r += 1
                        else:
                            if j < k: r += 1
                if t > 0: tied[f] += 1
                dcg += 1.0 / np.log2(r + 2.0)
        idcg = 0.0
        for k in range(npos): idcg += 1.0 / np.log2(k + 2.0)
        out[f] = dcg / idcg
    return out, tied, npos_all

ft = pl.read_parquet(ROOT / 'data/flat/train/feeds.parquet', columns=['user_id', 'M']); Mtr = ft['M'].to_numpy().astype(np.int64); uid = ft['user_id'].to_numpy().astype(np.int64)
pr = np.load(ROOT / 'eda/labels_position/out/train_prandom.npz'); pold = np.nan_to_num(pr['p_random'].astype(np.float32), nan=0.0); pold[Mtr < 50] = 0.0
hold = np.sort(np.random.default_rng(12345).choice(np.where((pold > 0.9) & (Mtr >= 50))[0], 30000, replace=False))
u = uid.astype(np.uint64); xf = np.where(uid == 0, np.arange(len(uid)) % 5, (((u * np.uint64(2654435761)) % np.uint64(4294967296)) % np.uint64(5)).astype(np.int64)); cu = xf[hold] != 0
Mh, oh = np.load(W / 'holdout/holdout_meta.npy'); Mh = Mh.astype(np.int64); oh = oh.astype(np.int64); yh = np.load(W / 'holdout/holdout_y.npy').astype(np.float32)
se = lambda d: d.std(ddof=1) / np.sqrt(len(d))
for suf in ('', '@last'):
    for mode, nm in ((0, 'pessimistic'), (1, 'optimistic'), (2, 'file order')):
        nd = {}
        for a in 'ab':
            for s in (951, 952):
                e, tied, npos = ndcg_mode(np.load(W / f'holdout/nn_w5_e10_u20{a}_s{s}{suf}_holdout.npy').astype(np.float32), yh, oh, Mh, mode); nd[a, s] = e[cu]
                if mode == 0 and suf == '': print(f'   u20{a}_s{s}: positives tied with another candidate {tied[cu].sum()} of {npos[cu].sum()} ({tied[cu].sum() / npos[cu].sum():.4%})')
        d = np.mean([nd['b', s] - nd['a', s] for s in (951, 952)], 0)
        print(f'{"selected" if suf == "" else "@last"} | ties {nm}: B-A member cross-user holdout (n {int(cu.sum())}) {d.mean():+.7f} +- {se(d):.7f}', flush=True)
