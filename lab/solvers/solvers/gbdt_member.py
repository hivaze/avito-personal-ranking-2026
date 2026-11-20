# Does a train-fit LUO GBDT ranker (CatBoost YetiRank / XGB) add to the current best v11 at pre-declared weights?
# v11 = 0.5 z(nn_final5_val_oof) + 0.5 z(nn_final5_trainonly_val). Weights 0.05/0.1/0.2 pre-declared (no fitting).
import sys, numpy as np
sys.path.insert(0, '<repo>/solvers/solvers')
from proxy_lib import *
P = '<repo>/preds/'
fv, Mv, ov = feed_meta('val'); yv = np.load(FLAT / 'val/cand_target.npy').astype(np.float64)
fid = np.repeat(np.arange(len(Mv)), Mv)
def zf(s):
    s = s.astype(np.float64); mu = np.bincount(fid, weights=s) / Mv; d = s - mu[fid]
    sd = np.sqrt(np.bincount(fid, weights=d * d) / Mv) + 1e-9; return d / sd[fid]
v11 = 0.5 * zf(np.load(P + 'nn_final5_val_oof.npy')) + 0.5 * zf(np.load(P + 'nn_final5_trainonly_val.npy'))
b = ndcg_per_feed(v11, yv, ov, Mv); print('v11 %.5f' % b.mean())
for k in ['cb_tr300k_luo', 'xgb_x300_luo', 'cb_tr500k_pw', 'pre2_final', 'pre_final']:
    x = zf(np.load(P + k + '_val.npy')); s1 = ndcg_per_feed(x, yv, ov, Mv).mean()
    corr = (np.bincount(fid, weights=zf(v11) * x) / Mv).mean()
    out = []
    for w in [0.05, 0.1, 0.2]:
        d = ndcg_per_feed(v11 + w * x, yv, ov, Mv) - b; out.append('w%.2f %+.5f (%.5f)' % (w, d.mean(), d.std() / np.sqrt(len(d))))
    print('%-14s single %.5f corr %.3f | %s' % (k, s1, corr, ' | '.join(out)))
