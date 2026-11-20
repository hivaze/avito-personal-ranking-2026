import numpy as np, numba as nb

@nb.njit(cache=True)
def feed_z_nan(off, M, x):
    out = np.zeros(len(x))
    for g in range(len(M)):
        o, m = off[g], M[g]
        s = 0.0; s2 = 0.0; c = 0
        for k in range(m):
            v = x[o + k]
            if v == v:
                s += v; s2 += v * v; c += 1
        if c >= 2:
            mu = s / c; sd = np.sqrt(max(s2 / c - mu * mu, 0.0))
            if sd > 0:
                for k in range(m):
                    v = x[o + k]
                    if v == v:
                        out[o + k] = (v - mu) / sd
    return out
