"""Numba kernels for set-level re-ranking rules on per-feed z-scored base scores."""
import numpy as np, numba as nb


@nb.njit(parallel=True, cache=True)
def simprop_topk(X, miss, off, M, z, k, a):
    """z_i + a * max_{j in top-k(z), j != i} cos(i, j)."""
    out = z.copy()
    for f in nb.prange(len(M)):
        o = off[f]; m = M[f]
        if m < 2: continue
        order = np.argsort(-z[o:o + m])
        kk = min(k, m)
        for i in range(m):
            if miss[o + i]: continue
            best = -2.0
            for r in range(kk):
                j = order[r]
                if j == i: continue
                if miss[o + j]: continue
                c = 0.0
                for d in range(X.shape[1]): c += X[o + i, d] * X[o + j, d]
                if c > best: best = c
            if kk == 1 and order[0] == i:
                # top-1 itself: use its similarity to the 2nd item
                j = order[1]
                if not miss[o + j]:
                    c = 0.0
                    for d in range(X.shape[1]): c += X[o + i, d] * X[o + j, d]
                    best = c
            if best > -2.0: out[o + i] = z[o + i] + a * best
    return out


@nb.njit(parallel=True, cache=True)
def softprop(X, miss, off, M, z, tau, a, center):
    """z_i + a * sum_{j != i} w_j cos(i,j), w = softmax(z / tau) renormalised without i; optionally minus the feed mean of that term."""
    out = z.copy()
    for f in nb.prange(len(M)):
        o = off[f]; m = M[f]
        if m < 2: continue
        Xf = X[o:o + m]
        C = Xf @ Xf.T
        zm = z[o:o + m].max()
        w = np.exp((z[o:o + m] - zm) / tau)
        for j in range(m):
            if miss[o + j]: w[j] = 0.0
        term = np.zeros(m)
        for i in range(m):
            s = 0.0; ws = 0.0
            for j in range(m):
                if j == i: continue
                s += w[j] * C[i, j]; ws += w[j]
            term[i] = s / ws if ws > 0 else 0.0
        mu = term.mean() if center else 0.0
        for i in range(m):
            if not miss[o + i]:
                out[o + i] = z[o + i] + a * (term[i] - mu)
    return out


@nb.njit(parallel=True, cache=True)
def knn_smooth(X, miss, off, M, z, thr, b):
    """(1-b) z_i + b * mean_{j: cos(i,j) > thr (incl. i)} z_j  (duplicate grouping / score smoothing)."""
    out = z.copy()
    for f in nb.prange(len(M)):
        o = off[f]; m = M[f]
        Xf = X[o:o + m]
        C = Xf @ Xf.T
        for i in range(m):
            if miss[o + i]: continue
            s = z[o + i]; n = 1.0
            for j in range(m):
                if j != i and (not miss[o + j]) and C[i, j] > thr:
                    s += z[o + j]; n += 1.0
            out[o + i] = (1 - b) * z[o + i] + b * s / n
    return out


@nb.njit(parallel=True, cache=True)
def dup_max(X, miss, off, M, z, thr, b):
    """z_i + b * max(0, max_{j: cos>thr} z_j - z_i): pull near-duplicates up toward the best member of their group."""
    out = z.copy()
    for f in nb.prange(len(M)):
        o = off[f]; m = M[f]
        Xf = X[o:o + m]
        C = Xf @ Xf.T
        for i in range(m):
            if miss[o + i]: continue
            mx = z[o + i]
            for j in range(m):
                if j != i and (not miss[o + j]) and C[i, j] > thr and z[o + j] > mx:
                    mx = z[o + j]
            out[o + i] = z[o + i] + b * (mx - z[o + i])
    return out


@nb.njit(parallel=True, cache=True)
def mmr(X, miss, off, M, z, a):
    """Greedy MMR: pick argmax z_i - a * max_{j selected} cos(i,j); output = descending rank score."""
    out = np.empty_like(z)
    for f in nb.prange(len(M)):
        o = off[f]; m = M[f]
        Xf = X[o:o + m]
        C = Xf @ Xf.T
        sel = np.zeros(m, np.bool_)
        mxs = np.full(m, -1.0)
        for r in range(m):
            best = -1e18; bi = -1
            for i in range(m):
                if sel[i]: continue
                v = z[o + i] - (a * mxs[i] if mxs[i] > -1.0 else 0.0)
                if v > best: best = v; bi = i
            sel[bi] = True
            out[o + bi] = m - r
            if not miss[o + bi]:
                for i in range(m):
                    if not sel[i] and not miss[o + i] and C[i, bi] > mxs[i]:
                        mxs[i] = C[i, bi]
    return out


@nb.njit(parallel=True, cache=True)
def cat_mass(off, M, z, cat, tau, a, mode):
    """mode 0: z_i + a * log(softmax(z/tau) mass of cat(i)); mode 1: z_i + a * 1[cat(i) == cat(top1)] (top1 itself unchanged)."""
    out = z.copy()
    for f in nb.prange(len(M)):
        o = off[f]; m = M[f]
        if m < 2: continue
        zf = z[o:o + m]; cf = cat[o:o + m]
        if mode == 1:
            t1 = np.argmax(zf)
            for i in range(m):
                if i != t1 and cf[i] == cf[t1]:
                    out[o + i] = zf[i] + a
        else:
            zm = zf.max()
            w = np.exp((zf - zm) / tau)
            w /= w.sum()
            for i in range(m):
                s = 0.0
                for j in range(m):
                    if cf[j] == cf[i]: s += w[j]
                out[o + i] = zf[i] + a * np.log(s + 1e-12)
    return out
