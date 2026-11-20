"""15) Full-train verification: do candidates ever appear in their own feed's history?"""
import numpy as np
import numba as nb
from common import feeds, arr
nb.set_num_threads(16)

@nb.njit(parallel=True, cache=True)
def kern(coff, M, hoff, H, ci, hi, out):
    for f in nb.prange(len(M)):
        h = H[f]
        if h == 0:
            continue
        s = np.sort(hi[hoff[f]:hoff[f] + h])
        c = 0
        for i in range(M[f]):
            k = np.searchsorted(s, ci[coff[f] + i])
            if k < h and s[k] == ci[coff[f] + i]:
                c += 1
        out[f] = c

for split in ['train', 'val', 'test']:
    F = feeds(split).sort('fidx')
    coff = F['cand_off'].to_numpy(); M = F['M'].to_numpy(); hoff = F['hist_off'].to_numpy(); H = F['H'].to_numpy()
    ci = np.asarray(arr(split, 'cand_item_id')); hi = np.asarray(arr(split, 'hist_item_id'))
    out = np.zeros(len(M), np.int64)
    kern(coff, M, hoff, H, ci, hi, out)
    print(f'{split}: candidates found in own history: {out.sum()} of {M.sum()} ({out.sum()/M.sum():.2e}); feeds affected {(out>0).sum()}')
