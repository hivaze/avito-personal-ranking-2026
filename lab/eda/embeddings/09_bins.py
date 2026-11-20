"""Positive rate vs embedding-similarity bins; signal lift by history length and feed size (val)."""
import numpy as np
from common import ndcg_per_feed, feeds, arr

names = open('<repo>/eda/embeddings/val_feat_names.txt').read().split('\n')
F = np.load('<repo>/eda/embeddings/val_feats.npy')
fv = feeds('val'); M = fv['M'].to_numpy(); off = fv['cand_off'].to_numpy(); H = fv['H'].to_numpy()
y = np.asarray(arr('val', 'cand_target')).astype(np.float64)
nf = len(M); fid = np.repeat(np.arange(nf), M)
tb = np.random.default_rng(0).random(len(y))


def rank_in_feed(s):
    s = np.asarray(s, np.float64).copy(); s[~np.isfinite(s)] = -1e18
    order = np.lexsort((tb, -s, fid)); r = np.empty(len(s)); r[order] = np.arange(len(s))
    return r - np.repeat(off, M)  # 0 = top


def nd(s):
    s = np.asarray(s, np.float64).copy(); s[~np.isfinite(s)] = -1e18
    order = np.lexsort((tb, s, fid)); r = np.empty(len(s)); r[order] = np.arange(len(s))
    return ndcg_per_feed(r, y, off, M)


print(f'base positive rate {y.mean():.4f}')
for nm, edges in [('max_sim', [-1, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.01]),
                  ('user_dot', None), ('rec_hl5', None)]:
    s = F[:, names.index(nm)]
    ok = np.isfinite(s)
    if edges is None:
        edges = np.quantile(s[ok], np.linspace(0, 1, 11)); edges[-1] += 1e-6
    print(f'\npositive rate by {nm} bin:')
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = ok & (s >= lo) & (s < hi)
        if m.sum():
            print(f'  [{lo:8.3f},{hi:8.3f}): n={m.sum():8d} ({m.mean():.3f})  pos rate {y[m].mean():.4f}  lift {y[m].mean() / y.mean():.2f}x')

print('\npositive rate by within-feed rank percentile (0=top) of user_dot / rec_hl5 / f1:')
for nm, s in [('user_dot', F[:, names.index('user_dot')]), ('rec_hl5', F[:, names.index('rec_hl5')]), ('f1', np.asarray(arr('val', 'cand_f1')))]:
    pct = rank_in_feed(s) / np.repeat(M, M)
    row = []
    for lo in np.arange(0, 1, 0.1):
        m = (pct >= lo) & (pct < lo + 0.1)
        row.append(f'{y[m].mean() / y.mean():.2f}')
    print(f'  {nm:9s} decile lift (top->bottom): ' + ' '.join(row))

# rand expected per feed
disc_cum = np.cumsum(1.0 / np.log2(np.arange(2, 400)))
npos = np.bincount(fid, weights=y, minlength=nf)
rand_pf = npos * disc_cum[M - 1] / M / disc_cum[np.maximum(npos.astype(int), 1) - 1]
pf = {nm: nd(F[:, names.index(nm)]) for nm in ['user_dot', 'rec_hl5', 'max_sim', 'centrality_loo']}
pf['f1'] = nd(np.asarray(arr('val', 'cand_f1')))
print('\nNDCG lift over expected-random by history length H:')
for lo, hi in [(0, 1), (1, 10), (10, 50), (50, 150), (150, 256), (256, 257)]:
    m = (H >= lo) & (H < hi)
    print(f'  H in [{lo},{hi}): feeds {m.sum():5d} | ' + ' '.join(f'{k}:{(v[m] - rand_pf[m]).mean():+.4f}' for k, v in pf.items()))
print('\nNDCG lift over expected-random by feed size M:')
for lo, hi in [(5, 20), (20, 50), (50, 100), (100, 200), (200, 301)]:
    m = (M >= lo) & (M < hi)
    print(f'  M in [{lo},{hi}): feeds {m.sum():5d} rand {rand_pf[m].mean():.3f} | ' + ' '.join(f'{k}:{(v[m] - rand_pf[m]).mean():+.4f}' for k, v in pf.items()))
# public/private proxy: first 30% vs last 70% of val by time
print('\nby time (val first 30% vs last 70% of feeds):')
cut = int(0.3 * nf)
for k, v in pf.items():
    print(f'  {k}: first30 {v[:cut].mean():.4f}  last70 {v[cut:].mean():.4f}')
