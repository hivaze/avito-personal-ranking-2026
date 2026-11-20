"""LambdaRank gradients of NDCG at the model's score: positive lambda = moving the item up helps."""
import numba
import numpy as np

from lib import DISC


@numba.njit(cache=True)
def lambdas(s, y, off, sz, disc, sigma):
    lam = np.zeros(len(s))
    for k in range(len(sz)):
        o = off[k]
        m = sz[k]
        ss = s[o:o + m]
        yy = y[o:o + m]
        npos = 0
        for i in range(m):
            npos += yy[i]
        if npos == 0 or npos == m:
            continue
        ideal = 0.0
        for i in range(npos):
            ideal += disc[i]
        idx = np.argsort(-ss)
        r = np.empty(m, np.int64)
        for t in range(m):
            r[idx[t]] = t
        for i in range(m):
            if yy[i] == 0:
                continue
            for j in range(m):
                if yy[j] == 1:
                    continue
                dn = abs(disc[r[i]] - disc[r[j]]) / ideal
                rho = 1.0 / (1.0 + np.exp(sigma * (ss[i] - ss[j])))
                lam[o + i] += dn * rho
                lam[o + j] -= dn * rho
    return lam


def get(D, sigma=1.0, s=None):
    s = D.s if s is None else s
    return lambdas(np.asarray(s, np.float64), D.y, D.off, D.sz, DISC, sigma)
