# Real-NN data point on loss diversity: round-1 train-only models (2 epochs, seed 0). base / nopos / noset use softmax CE,
# bce uses pointwise BCE (same architecture as base). Compare z-mean ensembles of base with each.
import sys, numpy as np
sys.path.insert(0, '<repo>/solvers/solvers')
from proxy_lib import *
fv, Mv, ov = feed_meta('val'); yv = np.load(FLAT / 'val/cand_target.npy').astype(np.float64)
fid = np.repeat(np.arange(len(Mv)), Mv)
def zf(s):
    mu = np.bincount(fid, weights=s) / Mv; d = s - mu[fid]; sd = np.sqrt(np.bincount(fid, weights=d * d) / Mv) + 1e-9
    return d / sd[fid]
P = {k: np.load(f'<repo>/preds/nn_{k}_trainonly_val.npy').astype(np.float64) for k in ['base', 'nopos', 'noset', 'bce', 'big_nopos']}
nd = {k: ndcg_per_feed(v, yv, ov, Mv) for k, v in P.items()}
for k in P: print('%-10s single %.5f' % (k, nd[k].mean()))
Z = {k: zf(v) for k, v in P.items()}
ref = ndcg_per_feed(Z['base'] + Z['nopos'], yv, ov, Mv)
for k in ['nopos', 'noset', 'bce', 'big_nopos']:
    e = ndcg_per_feed(Z['base'] + Z[k], yv, ov, Mv); d = e - ref
    corr = (np.bincount(fid, weights=Z['base'] * Z[k]) / Mv).mean()
    print('base+%-10s %.5f  gain over mean of singles %+.5f  vs base+nopos %+.5f (SE %.5f)  within-feed corr %.3f' % (
        k, e.mean(), e.mean() - (nd['base'].mean() + nd[k].mean()) / 2, d.mean(), d.std() / np.sqrt(len(d)), corr))
