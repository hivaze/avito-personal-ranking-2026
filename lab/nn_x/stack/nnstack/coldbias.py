"""Decomposition check: how much of the stacker gain is a per-candidate cold-item bias? score = z(pool) + beta*cold.
Nested on FT folds (beta picked on the 4 other folds), and holdout-fitted (to-half) -> val."""
import sys; sys.path.insert(0, '<repo>/src')
import numpy as np
from pathlib import Path
from common import feed_meta
from metrics import ndcg_per_feed
from stack_val import feed_rank_z
D = Path('<repo>/nn_x/stack/nnstack/data')
pool = sys.argv[1] if len(sys.argv) > 1 else 'PF7x'
_, M, o = feed_meta('val'); y = np.load(D / 'val_y.npy'); cold = np.load(D / 'val_cold.npy'); fold = np.load(D / 'val_fold.npy')
base = np.load(D / f'val_pool_{pool}_spec.npy')
B = np.round(np.arange(-0.2, 0.61, 0.05), 3)
nd = np.stack([ndcg_per_feed(base + b * cold, y, o, M) for b in B])  # [nb, nfeeds]
i0 = int(np.where(B == 0)[0][0])
print('in-sample val delta by beta:', ' '.join(f'{b:+.2f}:{(nd[i] - nd[i0]).mean():+.5f}' for i, b in enumerate(B)))
oof = np.zeros(len(M)); picks = []
for k in range(5):
    tr = fold != k; bi = int(np.argmax(nd[:, tr].mean(1))); picks.append(B[bi]); oof[fold == k] = nd[bi, fold == k]
d = oof - nd[i0]
tue = np.load(D / 'val_tue.npy'); fid = np.repeat(np.arange(len(M)), M); npos = np.bincount(fid, y, len(M)); nc = np.bincount(fid, y * cold, len(M))
sl = {'all': np.ones(len(M), bool), 'M<50': M < 50, 'M>=50': M >= 50, 'Tue': tue, 'Thu': ~tue, 'coldpos': (npos > 0) & (nc == npos), 'warmpos': (npos > 0) & (nc == 0)}
rep = lambda d: ' '.join(f'{k} {d[m].mean():+.5f}+-{d[m].std() / np.sqrt(m.sum()):.5f}' for k, m in sl.items())
print('nested picks', picks); print('NESTED cold-bias vs equal:', rep(d))
print('slice sizes', {k: int(m.sum()) for k, m in sl.items()})
# holdout-fitted
Mh, oh = np.load(D / 'hold_meta.npy'); yh = np.load(D / 'hold_y.npy'); ch = np.load(D / 'hold_cold.npy'); bh = np.load(D / f'hold_pool_{pool}_to.npy')
ndh = np.stack([ndcg_per_feed(bh + b * ch, yh, oh, Mh) for b in B]); i0h = i0
print('holdout delta by beta:', ' '.join(f'{b:+.2f}:{(ndh[i] - ndh[i0h]).mean():+.5f}' for i, b in enumerate(B)))
bh_best = B[int(np.argmax(ndh.mean(1)))]
dv = nd[int(np.where(B == bh_best)[0][0])] - nd[i0]
print(f'HOLDOUT-fitted beta {bh_best} -> val:', rep(dv))
print('cold share of candidates: val', cold.mean(), 'hold', ch.mean(), 'test', np.load(D / 'test_cold.npy').mean())
ft, Mt, ot = feed_meta('test'); ct = np.load(D / 'test_cold.npy'); pub = ft['timestamp'].to_numpy() < 1790497920
pubc = np.repeat(pub, Mt)
print('test cold share public', ct[pubc].mean(), 'private', ct[~pubc].mean())
print('pos cold share: val', (y * cold).sum() / y.sum(), 'hold', (yh * ch).sum() / yh.sum())
