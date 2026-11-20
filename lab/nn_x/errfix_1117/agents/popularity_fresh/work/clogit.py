"""Conditional (within-feed softmax) logit with offset = model score, L2-regularised, fitted by L-BFGS."""
import numpy as np
from scipy.optimize import minimize


def fit(X, y, off, s, lam, feed_of_row, nfeeds, b0=None):
    npos = np.add.reduceat(y, off).astype(float)
    ypos = y.astype(float)

    def f(b):
        z = s + X @ b
        mx = np.maximum.reduceat(z, off)
        e = np.exp(z - mx[feed_of_row])
        se = np.add.reduceat(e, off)
        lse = np.log(se) + mx
        ll = (ypos * z).sum() - (npos * lse).sum()
        p = e / se[feed_of_row]
        w = npos[feed_of_row] * p
        g = X.T @ (ypos - w)
        return -ll + lam * (b @ b), -g + 2 * lam * b

    b0 = np.zeros(X.shape[1]) if b0 is None else b0
    r = minimize(f, b0, jac=True, method="L-BFGS-B", options=dict(maxiter=500))
    return r.x
