"""Training-free fusions (within-feed z-score sums) incl. centrality and max_sim, val."""
import numpy as np
from common import ndcg_per_feed, feeds, arr
E = '<repo>/eda/embeddings'
names = open(f'{E}/val_feat_names.txt').read().split('\n'); F = np.load(f'{E}/val_feats.npy')
fv = feeds('val'); M = fv['M'].to_numpy(); off = fv['cand_off'].to_numpy(); nf = len(M); fid = np.repeat(np.arange(nf), M)
y = np.asarray(arr('val', 'cand_target')).astype(np.float64); tb = np.random.default_rng(0).random(len(y))
def nd(s):
    s = np.asarray(s, np.float64).copy(); s[~np.isfinite(s)] = -1e18
    o = np.lexsort((tb, s, fid)); r = np.empty(len(s)); r[o] = np.arange(len(s)); return ndcg_per_feed(r, y, off, M)
def z(s):
    s = np.where(np.isfinite(s), s, np.nan).astype(np.float64); ok = np.isfinite(s)
    m = np.bincount(fid, np.nan_to_num(s), nf) / np.maximum(np.bincount(fid, ok, nf), 1)
    d = np.where(ok, s - m[fid], 0); sd = np.sqrt(np.bincount(fid, d * d, nf) / M); return d / np.maximum(sd[fid], 1e-9)
g = lambda n: z(F[:, names.index(n)])
zf1 = z(np.asarray(arr('val', 'cand_f1'))); zf3 = z(np.asarray(arr('val', 'cand_f3')))
zu, zr, zm, zc = g('user_dot'), g('rec_hl5'), g('max_sim'), g('centrality_loo')
ref = nd(zf1 + zu + zr)
for nm, s in [('f1+user+rec', zf1 + zu + zr), ('f1+user+rec+0.5max', zf1 + zu + zr + 0.5 * zm), ('f1+user+rec+0.5cent', zf1 + zu + zr + 0.5 * zc),
              ('f1+user+rec+0.5max+0.5cent', zf1 + zu + zr + 0.5 * zm + 0.5 * zc), ('f1+0.5f3+user+rec+0.5max+0.5cent', zf1 + 0.5 * zf3 + zu + zr + 0.5 * zm + 0.5 * zc)]:
    pf = nd(s); d = pf - ref
    print(f'{nm:36s} {pf.mean():.4f}  vs f1+user+rec {d.mean():+.4f} (SE {d.std(ddof=1) / np.sqrt(nf):.4f})')
