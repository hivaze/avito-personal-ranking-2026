"""Compare single-signal NDCG on train subsample (production-ranked, position-biased) vs val (random-shuffled).
Shows whether embedding signals transfer and how much of train label signal is position."""
import numpy as np
from common import ndcg_per_feed, feeds, arr

C = '<repo>/eda/embeddings/cache'
names = open(f'{C}/feat_names.txt').read().split('\n')
Ftr = np.load(f'{C}/train_feats.npy', mmap_mode='r'); Btr = np.load(f'{C}/train_base.npy')
ytr = np.load(f'{C}/train_target.npy').astype(np.float64); Mtr = np.load(f'{C}/train_M.npy'); pos = np.load(f'{C}/train_pos.npy')
offtr = np.r_[0, np.cumsum(Mtr)[:-1]]
Fva = np.load('<repo>/eda/embeddings/val_feats.npy', mmap_mode='r')
fv = feeds('val'); Mva = fv['M'].to_numpy(); offva = fv['cand_off'].to_numpy()
yva = np.asarray(arr('val', 'cand_target')).astype(np.float64)
Bva = np.stack([np.asarray(arr('val', f'cand_f{k}')) for k in range(1, 6)], 1)


def nd(s, y, off, M, seed=0):
    s = np.asarray(s, np.float64).copy(); s[~np.isfinite(s)] = -1e18
    fid = np.repeat(np.arange(len(M)), M)
    tb = np.random.default_rng(seed).random(len(s))
    order = np.lexsort((tb, s, fid)); r = np.empty(len(s)); r[order] = np.arange(len(s))
    return ndcg_per_feed(r, y, off, M).mean()


print(f'train subsample: {len(Mtr)} feeds, {len(ytr)} cands, pos rate {ytr.mean():.4f}, mean M {Mtr.mean():.1f}')
print(f'{"signal":18s} {"train":>8s} {"val":>8s}')
print(f'{"random":18s} {nd(np.zeros(len(ytr)), ytr, offtr, Mtr):8.4f} {nd(np.zeros(len(yva)), yva, offva, Mva):8.4f}')
print(f'{"-pos (train only)":18s} {nd(-pos.astype(float), ytr, offtr, Mtr):8.4f}')
for k in range(5):
    print(f'{"f" + str(k + 1):18s} {nd(Btr[:, k], ytr, offtr, Mtr):8.4f} {nd(Bva[:, k], yva, offva, Mva):8.4f}')
for nm in ['user_dot', 'rec_hl5', 'last5_cos', 'max_sim', 'hmean_cos', 'centrality_loo', 'type1_max', 'item_norm']:
    j = names.index(nm)
    print(f'{nm:18s} {nd(Ftr[:, j], ytr, offtr, Mtr):8.4f} {nd(Fva[:, j], yva, offva, Mva):8.4f}')

# within-feed Spearman of signals with pos (train)
fid = np.repeat(np.arange(len(Mtr)), Mtr)


def wrank(s):
    s = np.asarray(s, np.float64)
    tb = np.random.default_rng(1).random(len(s))
    order = np.lexsort((tb, s, fid)); r = np.empty(len(s)); r[order] = np.arange(len(s))
    return r - np.repeat(offtr, Mtr)


def fsp(a, b):
    ra, rb = wrank(a), wrank(b)
    n = len(Mtr)
    ma = np.bincount(fid, ra, n) / Mtr; mb = np.bincount(fid, rb, n) / Mtr
    da = ra - ma[fid]; db = rb - mb[fid]
    num = np.bincount(fid, da * db, n); den = np.sqrt(np.bincount(fid, da * da, n) * np.bincount(fid, db * db, n))
    ok = den > 0
    return (num[ok] / den[ok]).mean()

print('\nwithin-feed Spearman with -pos on train (how much the production ranker followed each signal):')
for k in range(5):
    print(f'  f{k + 1}: {fsp(Btr[:, k], -pos):+.3f}')
for nm in ['user_dot', 'rec_hl5', 'max_sim', 'centrality_loo']:
    print(f'  {nm}: {fsp(np.nan_to_num(Ftr[:, names.index(nm)], nan=-9), -pos):+.3f}')

# NDCG of signals within train restricted to top-k positions vs lower positions? -> positive rate by pos bucket
print('\ntrain positive rate by original position bucket:')
for lo, hi in [(0, 5), (5, 10), (10, 20), (20, 50), (50, 100), (100, 200), (200, 301)]:
    m = (pos >= lo) & (pos < hi)
    print(f'  pos [{lo},{hi}): n={m.sum()}, pos rate {ytr[m].mean():.4f}')
