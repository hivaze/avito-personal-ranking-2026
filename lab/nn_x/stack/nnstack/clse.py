"""t-stats with per-feed SE and user-clustered SE (user0 feeds as singletons)."""
import sys; sys.path.insert(0, '<repo>/src')
import numpy as np
from pathlib import Path
from common import feed_meta
from metrics import ndcg_per_feed
D = Path('<repo>/nn_x/stack/nnstack/data'); O = Path('<repo>/nn_x/stack/nnstack/out')
f, M, o = feed_meta('val'); y = np.load(D / 'val_y.npy')
u = f['user_id'].to_numpy().astype(np.int64); u = np.where(u == 0, -1 - np.arange(len(u)), u)
_, inv = np.unique(u, return_inverse=True); print('feeds', len(u), 'clusters', inv.max() + 1)
for pool, runs in [('G3', sys.argv[1].split(',')), ('PF7x', sys.argv[2].split(','))]:
    nb = ndcg_per_feed(np.load(D / f'val_pool_{pool}_spec.npy'), y, o, M)
    for r in runs:
        d = ndcg_per_feed(np.load(O / f'{r}_val_oof.npy'), y, o, M) - nb
        se = d.std() / np.sqrt(len(d))
        cs = np.bincount(inv, d); n = len(d); G = len(cs)
        cse = np.sqrt(G / (G - 1) * np.sum((cs - np.bincount(inv, minlength=G) * d.mean()) ** 2)) / n
        print(f'{pool} {r:18s} delta {d.mean():+.6f} se {se:.6f} t {d.mean() / se:.2f} | clustered se {cse:.6f} t {d.mean() / cse:.2f}')
