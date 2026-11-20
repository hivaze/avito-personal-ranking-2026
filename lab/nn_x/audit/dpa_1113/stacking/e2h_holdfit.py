"""E2h (reverse direction): family weights fitted on the HOLDOUT (user-grouped 5-fold inside the holdout for the holdout
OOF read; full-holdout fit applied to val at w_ft 0.5 for the val read). Uses the grid matrices of e23_grid.py."""
from lib import *
from sklearn.model_selection import GroupKFold
import pmix

V = S('val'); HO = S('hold')
NV = np.load(CACHE / 'grid_NV.npy'); NH = np.load(CACHE / 'grid_NH.npy'); A = np.load(CACHE / 'grid_A.npy')
iME = int(np.argmin(np.abs(A - np.array([.3, .35, .35])).sum(1))); iW5 = 2
wpriv = pmix.weights()['private']
g = np.where(HO.u0, -1 - np.arange(HO.n), HO.user)
hf = np.zeros(HO.n, int)
for k, (_, te) in enumerate(GroupKFold(5).split(np.arange(HO.n), groups=g)): hf[te] = k
np.save(CACHE / 'hold_fold.npy', hf)
log = open(OUT / 'e2h_holdfit.txt', 'w')
def pr(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); log.write(s + '\n'); log.flush()
COARSE = [ia for ia in range(len(A)) if np.allclose(np.round(A[ia] * 10), A[ia] * 10)]
for nm, allowed in [('step .05', list(range(len(A)))), ('step .1', COARSE)]:
    oof = np.zeros(HO.n); ch = []
    for k in range(5):
        tr = hf != k; ia = max(allowed, key=lambda j: NH[tr, j].mean()); oof[hf == k] = NH[hf == k, ia]; ch.append(tuple(np.round(A[ia], 2)))
    dh = oof - NH[:, iME]; hm, hse = se_cl(dh, HO.cl)
    ia = max(allowed, key=lambda j: NH[:, j].mean())
    dv = NV[:, ia, iW5].astype(np.float64) - NV[:, iME, iW5]; vm, vse = se_cl(dv, V.cl); pm, pse = wmean_se(dv, wpriv)
    pr(f'holdout-fit family weights ({nm}): HOLD-OOF {hm:+.5f} +- {hse:.5f} (z {hm / hse:+.2f}) fold choices {ch} | full-holdout a={tuple(np.round(A[ia], 2))} '
       f'-> VAL {vm:+.5f} +- {vse:.5f} (z {vm / vse:+.2f}) pmix {pm:+.5f} +- {pse:.5f}')
