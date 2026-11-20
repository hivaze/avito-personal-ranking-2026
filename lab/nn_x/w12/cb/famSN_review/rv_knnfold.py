"""Measured fold exclusion of famN's hand-made KNN column (hknn50_pair, n30k): for sampled fit rows, exact 50-NN mean of teacher_oof over (a) the fit rows of the OTHER user folds
(what hx.py declares), (b) all fit rows except the row itself (would include the own feed / own user), compared with the stored column."""
import sys
sys.path.insert(0, '<repo>/nn_x/w12/cb/famSN_review'); sys.path.insert(0, '<repo>/nn_x/w12/cb/famN/scripts')
import numpy as np, fn; from fn import cbx
tr = cbx.load('xm'); fit, sel = fn.subset(30000, 5000); b = fn.Block(tr, fit)
P = b.vec_pca('pair', 32).astype(np.float32); t = b.label('oof').astype(np.float64); fold = tr.xfold[b.frow].astype(np.int64); g = b.gid
stored = np.load(fn.CACHE / 'hx/n30k/hknn50_pair_xm.npy')[b.idx].astype(np.float64)
rg = np.random.default_rng(11); q = np.sort(rg.choice(b.n, 400, replace=False))
pn = (P * P).sum(1)
ex, inc, own_share = [], [], []
for i in q:
    d = pn - 2.0 * (P @ P[i]) + pn[i]
    d_out = d.copy(); d_out[fold == fold[i]] = np.inf; nn = np.argpartition(d_out, 50)[:50]; ex.append(t[nn].mean())
    d_in = d.copy(); d_in[i] = np.inf; nn2 = np.argpartition(d_in, 50)[:50]; inc.append(t[nn2].mean()); own_share.append(float((g[nn2] == g[i]).mean()))
ex, inc, s = np.array(ex), np.array(inc), stored[q]
print(f'hknn50_pair, {len(q)} sampled fit rows of {b.n}: corr(stored, exact other-fold 50-NN mean) {np.corrcoef(s, ex)[0, 1]:.4f}, mean abs diff {np.abs(s - ex).mean():.4f}; '
      f'corr(stored, exact 50-NN mean incl. own fold) {np.corrcoef(s, inc)[0, 1]:.4f}, mean abs diff {np.abs(s - inc).mean():.4f}; sd of the column {s.std():.4f}; '
      f'own-feed share among the 50 nearest when the own fold is NOT excluded {np.mean(own_share):.3f}; corr with own teacher value: stored {np.corrcoef(s, t[q])[0, 1]:.3f}, other-fold exact {np.corrcoef(ex, t[q])[0, 1]:.3f}, incl. own fold {np.corrcoef(inc, t[q])[0, 1]:.3f}')
