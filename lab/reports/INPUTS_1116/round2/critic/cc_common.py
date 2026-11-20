"""CRITIC of round 2: shared code. CPU only, 12 threads. Read-only on every existing file; writes only under reports/INPUTS_1116/round2/critic/.
The null model fit is the lens of record (lens.fit, condlib.fit_big, condlib.control_block). The SCORE of a candidate column is my own numpy path
(explicit residualisation of the column on the null model's columns in the information metric), not condlib.Null.score."""
import os, sys, time
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS', 'POLARS_MAX_THREADS'):
    os.environ[_k] = '12'
os.environ['RS_NT'] = '12'
CC = '<repo>/reports/INPUTS_1116/round2/critic'
os.environ['NUMBA_CACHE_DIR'] = CC + '/out/numba_cache'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True
import numba as nb          # numba reads NUMBA_CACHE_DIR here; later assignments by imported modules do not move the cache
import numpy as np
import polars as pl
from pathlib import Path
sys.path.insert(0, '<repo>/reports/INPUTS_1116/round2/rescue')
sys.path.insert(0, '<repo>/reports/INPUTS_1116/residual_rows')
sys.path.insert(0, '<repo>/src')
import rs_common as rs      # Views (row maps of the rescue arrays), window(); no file is written at import
_argv = sys.argv; sys.argv = [sys.argv[0], 'val', 'TO16']
import lens
import condlib
sys.argv = _argv
nb.set_num_threads(12)
ROOT = Path('<repo>'); FLAT = ROOT / 'data/flat'; OUT = Path(CC) / 'out'; NX3 = ROOT / 'nn_x/w12/nx3'
RARR = ROOT / 'reports/INPUTS_1116/round2/rescue/out/arr'
T0 = time.time()


class Log:
    def __init__(self, path):
        self.f = open(path, 'x')

    def __call__(self, *a):
        s = f'[{time.time() - T0:6.0f}s] ' + ' '.join(str(x) for x in a)
        print(s, flush=True); self.f.write(s + '\n'); self.f.flush()


def rget(q, name):
    return np.load(RARR / f'{q}__{name}.npy').astype(np.float64)


def form(x, off, M, t, fm):
    x = np.array(x, np.float64); x[~np.isfinite(x)] = np.nan
    if t == 'log1p':
        x = np.log1p(np.maximum(x, 0))
    return (lens.feed_z(x, off, M, 5.0) if fm == 'z' else lens.feed_rk(x, off, M)).astype(np.float32)


class MyNull:
    """null model = pool + cols (fit: condlib.fit_big). Score of one more column by explicit residualisation (numpy)."""

    def __init__(self, D, cols, theta0=None):
        self.D = D; n = D['n']; off = D['off']
        self.X = np.ascontiguousarray(np.stack([D['base']] + list(cols), 1).astype(np.float32)); K = self.X.shape[1]
        if theta0 is None:
            f0 = lens.fit(D, []); theta0 = [f0['theta'][0]] + [0.0] * len(cols)
        r = condlib.fit_big(D, self.X, theta0=theta0)
        self.theta = r['theta']; self.L = r['L']
        self.fid = np.repeat(np.arange(n), D['M'])
        # my own probabilities from theta (numpy), not the kernel's
        s = self.X @ self.theta.astype(np.float32); s = s.astype(np.float64)
        mx = np.maximum.reduceat(s, off); e = np.exp(s - mx[self.fid]); Z = np.add.reduceat(e, off); self.p = e / Z[self.fid]
        self.npos = np.bincount(self.fid, D['y'], n)
        self.w = self.npos[self.fid] * self.p
        self.res = D['y'].astype(np.float64) - self.w
        self.Lmine = float((D['y'] * np.log(np.maximum(self.p, 1e-300))).sum())
        self.grad = self.X.astype(np.float64).T @ self.res
        A = np.zeros((K, K)); self.MU = np.zeros((n, K))
        step = 400000; b = np.searchsorted(off, np.arange(0, len(self.p), step)); b = np.unique(np.concatenate([b, [n]]))
        for i in range(len(b) - 1):
            f0_, f1_ = b[i], b[i + 1]; r0 = off[f0_]; r1 = off[f1_] if f1_ < n else len(self.p)
            Xc = self.X[r0:r1].astype(np.float64)
            A += (Xc * self.w[r0:r1, None]).T @ Xc
            self.MU[f0_:f1_] = np.add.reduceat(Xc * self.p[r0:r1, None], off[f0_:f1_] - r0, axis=0)
        A -= (self.MU * self.npos[:, None]).T @ self.MU
        self.A = A
        self.Ainv = np.linalg.inv(A + 1e-6 * np.eye(K) * np.trace(A) / K)
        self.Gf = None

    def score_f(self, x):
        """per-feed efficient score, model variance V"""
        D = self.D; off = D['off']; x = np.asarray(x, np.float64)
        mx = np.add.reduceat(self.p * x, off)
        b = self.X.astype(np.float32).T @ (self.w * x).astype(np.float32)
        b = b.astype(np.float64) - self.MU.T @ (self.npos * mx)
        g = self.Ainv @ b
        xt = x - (self.X @ g.astype(np.float32)).astype(np.float64)
        Uf = np.add.reduceat(self.res * xt, off)
        Ixx = float((self.w * x * x).sum() - (self.npos * mx * mx).sum())
        V = Ixx - float(b @ g)
        return Uf, V, Ixx

    def cz(self, x, wf=None):
        Uf, V, Ixx = self.score_f(x)
        return czw(self.D, Uf, wf), V, Ixx, Uf


def czw(D, Uf, wf=None):
    u = Uf if wf is None else Uf * wf
    Uc = np.bincount(D['clus'], u); s = np.sqrt((Uc ** 2).sum())
    return float(u.sum() / s) if s > 0 else 0.0


def perm_order(fid, rng):
    """row order that shuffles rows inside every feed (rows of a feed are contiguous)"""
    return np.argsort(fid + rng.random(len(fid)), kind='stable')
