"""Conditional logit (within-feed choice model) on the head: P(item i is a chosen positive | feed head)
∝ exp(b0*score_i + b . x_i). Each positive in the head is one choice among itself + the head negatives."""
import numpy as np
from scipy.optimize import minimize


def build_sets(fid, y, inh):
    """Return list of (pos_index, neg_indices) per positive, over rows with inh True."""
    idx = np.nonzero(inh)[0]
    f = fid[idx]
    starts = np.r_[0, np.nonzero(f[1:] != f[:-1])[0] + 1, len(f)]
    P, S, Soff = [], [], [0]
    for a, b in zip(starts[:-1], starts[1:]):
        rows = idx[a:b]
        yy = y[rows]
        if yy.sum() == 0 or yy.sum() == len(rows):
            continue
        negs = rows[yy == 0]
        for p in rows[yy == 1]:
            P.append(p)
            S.extend([p] + negs.tolist())
            Soff.append(len(S))
    return np.array(P), np.array(S), np.array(Soff)


def fit(X, P, S, Soff, l2=1e-4, free0=False):
    """X: rows x k design (all rows). Returns beta, se."""
    Xs = X[S]
    seg = np.repeat(np.arange(len(Soff) - 1), np.diff(Soff))
    Xp = X[P]
    nset = len(P)

    pen = np.ones(X.shape[1])
    if free0:
        pen[0] = 0.0

    def f(b):
        u = Xs @ b
        mx = np.maximum.reduceat(u, Soff[:-1])
        e = np.exp(u - mx[seg])
        den = np.add.reduceat(e, Soff[:-1])
        ll = (Xp @ b).sum() - (np.log(den) + mx).sum()
        p = e / den[seg]
        g = Xp.sum(0) - np.add.reduceat(p[:, None] * Xs, Soff[:-1]).sum(0)
        return -ll / nset + 0.5 * l2 * (pen * b @ b), -g / nset + l2 * pen * b

    b0 = np.zeros(X.shape[1])
    r = minimize(f, b0, jac=True, method="L-BFGS-B")
    b = r.x
    # Hessian
    u = Xs @ b
    mx = np.maximum.reduceat(u, Soff[:-1])
    e = np.exp(u - mx[seg])
    den = np.add.reduceat(e, Soff[:-1])
    p = e / den[seg]
    mean = np.add.reduceat(p[:, None] * Xs, Soff[:-1])
    H = np.zeros((X.shape[1], X.shape[1]))
    Xc = Xs - mean[seg]
    H = (Xc * p[:, None]).T @ Xc
    cov = np.linalg.pinv(H)
    return b, np.sqrt(np.diag(cov)), -r.fun * nset
