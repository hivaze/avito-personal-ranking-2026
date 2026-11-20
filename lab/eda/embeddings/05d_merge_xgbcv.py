"""Merge per-feed OOF NDCG arrays from 05c runs (XGBoost rank:ndcg, 5-fold val CV, 2 fold seeds) and report paired differences."""
import numpy as np, glob, os
from common import feeds, arr, ndcg_per_feed
E = '<repo>/eda/embeddings/cache'
R = {os.path.basename(p)[6:-4]: np.load(p).mean(0) for p in sorted(glob.glob(f'{E}/xgbcv_*.npy'))}
n = len(R['base'])
fv = feeds('val'); M = fv['M'].to_numpy(); off = fv['cand_off'].to_numpy(); y = np.asarray(arr('val', 'cand_target')).astype(float)
f1 = ndcg_per_feed(np.asarray(arr('val', 'cand_f1')) + np.random.default_rng(0).random(len(y)) * 1e-9, y, off, M)
print(f'f1 alone {f1.mean():.4f}')
for ref in ['base', 'base+compact8']:
    print(f'\npaired vs {ref}:')
    for k, v in R.items():
        d = v - R[ref]
        print(f'  {k:30s} {v.mean():.4f}  {d.mean():+.4f} (SE {d.std(ddof=1) / np.sqrt(n):.4f})')
