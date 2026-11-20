"""Compute embedding features for a random subsample of train feeds (for LightGBM lift experiments)."""
import sys
import time
import numpy as np
from common import feeds
from emb_features import load_split, compute_features

N = int(sys.argv[1]) if len(sys.argv) > 1 else 200_000
OUT = '<repo>/eda/embeddings/cache'
import os
os.makedirs(OUT, exist_ok=True)
t0 = time.time()
nf = len(feeds('train'))
fidx = np.sort(np.random.default_rng(42).choice(nf, N, replace=False))
D = load_split('train', fidx)
print('loaded', time.time() - t0, 'cands', len(D['c_target']), 'hist', len(D['h_emb_idx']))
F, names = compute_features(D)
print('features', time.time() - t0)
np.save(f'{OUT}/train_feats.npy', F)
np.save(f'{OUT}/train_fidx.npy', fidx)
base = np.stack([D['c_f1'], D['c_f2'], D['c_f3'], D['c_f4'], D['c_f5'], D['c_category_id'].astype(np.float32)], 1).astype(np.float32)
np.save(f'{OUT}/train_base.npy', base)
np.save(f'{OUT}/train_target.npy', D['c_target'])
np.save(f'{OUT}/train_pos.npy', D['c_pos'])
np.save(f'{OUT}/train_M.npy', D['M'])
open(f'{OUT}/feat_names.txt', 'w').write('\n'.join(names))
print('done', time.time() - t0)
