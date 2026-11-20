"""hz2: basic label-free size facts (no feature distribution of val/test is compared here)."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS','POLARS_MAX_THREADS'): os.environ[k] = '12'
import numpy as np, polars as pl, time
R = '<repo>'; MSK = 3*3600; DAY = 86400
t0 = time.time()
for sp in ('train','val','test'):
    f = pl.read_parquet(f'{R}/data/flat/{sp}/feeds.parquet')
    d = (f['timestamp'].to_numpy().astype(np.int64) + MSK)//DAY
    u, c = np.unique(d, return_counts=True)
    print(sp, len(f), dict(zip((u-20702).tolist(), c.tolist())), 'sumM', int(f['M'].sum()), 'sumH', int(f['H'].sum()), flush=True)
    if sp == 'train':
        fid = f['feed_id'].to_numpy()
        h = (fid.astype(np.uint64) * np.uint64(0xC2B2AE3D27D4EB4F)) >> np.uint64(40)
        sp_ = (h % np.uint64(37)) == 0
        u2, c2 = np.unique(d[sp_], return_counts=True); print('  sparse-sample feeds per day', dict(zip((u2-20702).tolist(), c2.tolist())))
        us = f['user_id'].to_numpy(); print('  user0 feeds', int((us==0).sum()), 'distinct users', len(np.unique(us)))
ci = np.load(f'{R}/data/flat/train/cand_item_id.npy', mmap_mode='r')
print('cand_item dtype', ci.dtype, 'max', int(ci.max()), f'{time.time()-t0:.0f}s', flush=True)
ut = np.unique(np.asarray(ci)); print('distinct train cand items', len(ut), f'{time.time()-t0:.0f}s', flush=True)
for sp in ('val','test'):
    x = np.load(f'{R}/data/flat/{sp}/cand_item_id.npy'); ux = np.unique(x); print(sp, 'distinct cand items', len(ux), 'not in train cands', int((~np.isin(ux, ut)).sum()), 'max', int(x.max()))
hi = np.load(f'{R}/data/flat/train/hist_item_id.npy', mmap_mode='r'); print('hist item max', int(hi.max()), hi.dtype, len(hi))
x = np.load(f'{R}/nn/feat_cache/train/seq_cv_pop.npy', mmap_mode='r'); s = np.asarray(x[::50]); print('as-built train seq_cv_pop (1/50 rows): mean', float(s.mean()), 'q50/q90/q99', np.quantile(s,[.5,.9,.99]).tolist(), 'share>=3000', float((s>=3000).mean()))
x = np.load(f'{R}/nn/feat_cache/train/it_hist_all_luo.npy', mmap_mode='r'); s = np.asarray(x[::50]); print('as-built train it_hist_all_luo (1/50 rows): mean', float(s.mean()), 'q50/q90/q99', np.quantile(s,[.5,.9,.99]).tolist(), 'max', float(s.max()))
print(f'{time.time()-t0:.0f}s')
