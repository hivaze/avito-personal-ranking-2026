"""Shared helpers for the NONLINEAR lens (CPU only)."""
import json, sys
import numpy as np
import numba as nb
sys.path.insert(0, "<repo>/src")
from metrics import ndcg_per_feed  # noqa: E402

D = "/tmp/scratchpad/dpa/nonlinear"
ROOT = "<repo>"
VERT = json.load(open(f"{ROOT}/nn_x/codes/categories/out/code_vertical.json"))
VNAMES = sorted(set(VERT.values()))


def load(split):
    z = np.load(f"{D}/{split}.npz")
    d = {k: z[k] for k in z.files}
    d["fid"] = np.repeat(np.arange(len(d["M"])), d["M"])
    d["nf"] = len(d["M"])
    vmap = np.full(64, len(VNAMES), np.int64)
    for k, v in VERT.items():
        vmap[int(k)] = VNAMES.index(v)
    d["vert"] = vmap[d["cat"].astype(np.int64)]
    return d


def fsum(d, x):
    return np.bincount(d["fid"], x, minlength=d["nf"])


def fz(d, x):
    x = np.asarray(x, np.float64)
    M = d["M"]
    m = fsum(d, x) / M
    s = np.sqrt(np.maximum(fsum(d, x * x) / M - m ** 2, 0))
    return (x - m[d["fid"]]) / np.where(s[d["fid"]] > 1e-9, s[d["fid"]], 1.0)


@nb.njit(parallel=True)
def _pct(off, M, x, out):
    for f in nb.prange(len(M)):
        o, m = off[f], M[f]
        v = x[o:o + m]
        order = np.argsort(v, kind="mergesort")
        r = np.empty(m)
        i = 0
        while i < m:
            j = i
            while j + 1 < m and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = 0.5 * (i + j)
            i = j + 1
        for k in range(m):
            out[o + k] = r[k] / (m - 1) if m > 1 else 0.5


def fpct(d, x):
    out = np.empty(len(x))
    _pct(d["off"], d["M"], np.asarray(x, np.float64), out)
    return out


def group_pct(d, key, x):
    """mid-rank percentile of x within groups `key` (int64, contiguous not required). 0.5 for singletons."""
    x = np.asarray(x, np.float64)
    o = np.lexsort((x, key))
    ks = key[o]; xs = x[o]
    st = np.r_[True, ks[1:] != ks[:-1]]
    gid = np.cumsum(st) - 1
    start = np.flatnonzero(st)
    n = np.bincount(gid)
    # ties: mid-rank via (key, x) groups
    st2 = st | np.r_[True, xs[1:] != xs[:-1]]
    g2 = np.cumsum(st2) - 1
    s2 = np.flatnonzero(st2); n2 = np.bincount(g2)
    rk_first = s2[g2] - start[gid]
    rk = rk_first + 0.5 * (n2[g2] - 1)
    out = np.empty(len(x))
    out[o] = np.where(n[gid] > 1, rk / np.maximum(n[gid] - 1, 1), 0.5)
    return out, n[gid][np.argsort(o)]


def ndcg(d, s):
    return ndcg_per_feed(np.asarray(s, np.float32), d["y"], d["off"], d["M"])


def folds5(d, seed=0):
    """user-grouped 5 folds over feeds; user 0 feeds each their own group."""
    uid = d["uid"]
    rng = np.random.default_rng(seed)
    uu = np.unique(uid[uid != 0])
    fu = dict(zip(uu.tolist(), rng.integers(0, 5, len(uu)).tolist()))
    f = np.array([fu[u] if u != 0 else rng.integers(0, 5) for u in uid.tolist()])
    return f


def se(x):
    return x.std() / np.sqrt(len(x))


# private / public mix weights over val feeds (src/pmix.py)
def pmix_w():
    sys.path.insert(0, f"{ROOT}/eda/errors/segments")
    import pmix
    return pmix.weights()


def wmean_se(x, w):
    m = np.average(x, weights=w)
    neff = w.sum() ** 2 / (w ** 2).sum()
    return m, np.sqrt(np.average((x - m) ** 2, weights=w) / neff)


def fit_calib(d, g, knots=(-1.0, 0.0, 1.0, 1.5, 2.0, 2.5, 3.0)):
    """Within-feed conditional logit on a hinge basis of g: logit = B(g) @ beta. Returns beta, basis fn, p (softmax)."""
    from scipy.optimize import minimize

    def basis(gg):
        return np.stack([gg] + [np.maximum(gg - k, 0) for k in knots], 1)
    B = basis(np.asarray(g, np.float64))
    y = d["y"].astype(np.float64); fid = d["fid"]; nf = d["nf"]
    npos = np.bincount(fid, y, nf)

    def f(beta):
        l = B @ beta
        mx = np.maximum.reduceat(l, d["off"])
        e = np.exp(l - mx[fid]); Z = np.bincount(fid, e, nf)
        p = e / Z[fid]
        loss = -(y * l).sum() + (npos * (np.log(Z) + mx)).sum()
        grad = -(y[:, None] * B).sum(0) + ((npos[fid] * p)[:, None] * B).sum(0)
        return loss, grad
    r = minimize(f, np.r_[1.0, np.zeros(len(knots))], jac=True, method="L-BFGS-B")
    beta = r.x
    l = B @ beta
    mx = np.maximum.reduceat(l, d["off"]); e = np.exp(l - mx[fid]); p = e / np.bincount(fid, e, nf)[fid]
    return beta, basis, p, npos
