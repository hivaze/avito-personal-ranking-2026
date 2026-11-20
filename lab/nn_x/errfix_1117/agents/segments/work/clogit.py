"""Conditional logit (softmax within feed) fitter: new = a*score + X @ w; used to find directions of residuals."""
import numpy as np
from scipy.optimize import minimize


def fit(score, X, y, feed, nf, rowmask=None, l2=1.0, a0=3.0):
    """Maximise sum over positives of log softmax_feed(a*score + X w). Returns (a, w)."""
    if rowmask is None:
        rowmask = np.ones(len(y), bool)
    s = score[rowmask]; Xm = X[rowmask]; ym = y[rowmask]; fm = feed[rowmask]
    # re-index feeds
    uf, fi = np.unique(fm, return_inverse=True)
    n = len(uf)
    npos = np.bincount(fi, weights=ym, minlength=n)
    k = Xm.shape[1]

    def f(theta):
        a, w = theta[0], theta[1:]
        z = a * s + Xm @ w
        zmax = np.full(n, -np.inf); np.maximum.at(zmax, fi, z)
        e = np.exp(z - zmax[fi])
        se = np.bincount(fi, weights=e, minlength=n)
        lse = np.log(se) + zmax
        ll = (ym * z).sum() - (npos * lse).sum()
        p = e / se[fi]
        r = ym - npos[fi] * p  # gradient wrt z
        ga = (r * s).sum()
        gw = Xm.T @ r
        obj = -ll + 0.5 * l2 * (w ** 2).sum()
        grad = -np.r_[ga, gw] + np.r_[0.0, l2 * w]
        return obj, grad

    th0 = np.r_[a0, np.zeros(k)]
    res = minimize(f, th0, jac=True, method="L-BFGS-B")
    return res.x[0], res.x[1:], -res.fun
