import numpy as np
def fit(y, X, iters=30):
    """Plain logistic regression by Newton; returns params, tvalues."""
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ b)))
        W = p * (1 - p)
        H = X.T @ (X * W[:, None]) + 1e-9 * np.eye(X.shape[1])
        g = X.T @ (y - p)
        step = np.linalg.solve(H, g)
        b += step
        if np.abs(step).max() < 1e-8: break
    cov = np.linalg.inv(H)
    return b, b / np.sqrt(np.diag(cov))
def add_constant(X):
    return np.c_[np.ones(len(X)), X]
