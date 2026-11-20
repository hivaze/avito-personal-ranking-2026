"""Lane A (repaired click strength): shared code. CPU only, <= 12 threads. Read-only on every existing file; writes only under
reports/NEXT_1116/clickstrength/ (exclusive create for arrays and logs). Definitions: RULE.txt in this directory (written before this file).

View "P3": window(e) = slots {e-2, e-1, e}; train rows e = slot - l with the lag tables of RULE.txt section 2; val / test natural e = 13;
val private-like twin e = 8 / 7 (slot 16), 8 (slots >= 17)."""
import os, sys, time, hashlib
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS', 'POLARS_MAX_THREADS'):
    os.environ[_k] = '12'
CS = '<repo>/reports/NEXT_1116/clickstrength'
os.environ['NUMBA_CACHE_DIR'] = CS + '/out/numba_cache'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True
import numba as nb          # numba reads NUMBA_CACHE_DIR here
import numpy as np
import polars as pl
from pathlib import Path
nb.set_num_threads(12)
sys.path.insert(0, '<repo>/nn_x/w12/stale/hz2')
import hz2_common as hc     # constants and hashes only; no file is written at import
nb.set_num_threads(12)

ROOT = Path('<repo>'); FLAT = ROOT / 'data/flat'; OUT = Path(CS) / 'out'; ARR = OUT / 'arr'
NX3 = ROOT / 'nn_x/w12/nx3'
SPLITS = ('train', 'val', 'test')
NROWS = {'train': 246745836, 'val': 1389971, 'test': 2904806}
DENSE = hc.DENSE_SLOTS.astype(np.int64)            # 0..8, 11, 12, 13
S12 = np.full(24, -1, np.int64); S12[DENSE] = np.arange(len(DENSE))   # slot -> column 0..11 of the per-slot tables (-1 = not a source day)
DEPTH = 3
NAT_E = 13
H_TR = np.uint64(0x9E3779B97F4A7C15)       # RULE.txt section 2: lag draw of a train feed
H_TW = np.uint64(0xBF58476D1CE4E5B9)       # RULE.txt section 2: lag draw of a val feed in the private-like twin
Q5, A11, B8, B9 = 0.668, 0.390, 0.161, 0.412
NB = 8                                     # exposure strata per feed (critic/c07_condperm.py)
T0 = time.time()
QS = ('hold', 'pool', 'val_nat', 'val_twin', 'test_nat')


def md5(p):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


class Log:
    def __init__(self, path):
        self.f = open(path, 'x')

    def __call__(self, *a):
        s = f'[{time.time() - T0:6.0f}s] ' + ' '.join(str(x) for x in a)
        print(s, flush=True); self.f.write(s + '\n'); self.f.flush()


def save_new(path, x):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'xb') as f:
        np.save(f, x)


def window(e):
    """the three consecutive slots that end at the newest source slot e; all must be dense"""
    w = np.array([e - 2, e - 1, e], np.int64)
    assert (w >= 0).all() and (S12[w] >= 0).all(), f'window of e = {e} is not three dense slots'
    return w


def train_lag(slot, feed_id):
    """RULE.txt section 2: lag per train feed (-1 = no view)"""
    u = hc.hash_uniform(feed_id, H_TR); lag = np.full(len(slot), -1, np.int64)
    lag[slot == 6] = 3
    m = (slot == 7) | (slot == 8); lag[m] = np.where(u[m] < Q5, 5, 3)
    m = slot == 11; lag[m] = np.where(u[m] < A11, 8, 9)
    m = (slot == 12) | (slot == 13); lag[m] = np.where(u[m] < B8, 8, np.where(u[m] < B9, 9, 10))
    return lag


def twin_e(slot, feed_id):
    """RULE.txt section 2: newest source slot of a val feed in the private-like twin"""
    u = hc.hash_uniform(feed_id, H_TW)
    return np.where(slot == 16, np.where(u < A11, 8, 7), 8).astype(np.int64)


class Views:
    """feeds tables, the query sets and their newest source slot e per feed (-1 = no view)"""

    def __init__(self, P=None):
        self.F = {s: pl.read_parquet(FLAT / s / 'feeds.parquet') for s in SPLITS}
        self.A = {}
        for s in SPLITS:
            f = self.F[s]; ts = f['timestamp'].to_numpy().astype(np.int64)
            assert (f['fidx'].to_numpy() == np.arange(len(f))).all()
            M = f['M'].to_numpy().astype(np.int64); co = f['cand_off'].to_numpy().astype(np.int64)
            assert int(M.sum()) == NROWS[s] and (co == np.concatenate([[0], np.cumsum(M)[:-1]])).all()
            self.A[s] = dict(M=M, co=co, ts=ts, slot=(hc.day_of(ts) - hc.DAY0).astype(np.int64), u=f['user_id'].to_numpy().astype(np.int64), fid=f['feed_id'].to_numpy(), n=len(f))
        a = self.A['train']
        hold, prand = hc.holdout_feeds(self.F['train']); self.hold = hold
        self.ish = np.zeros(a['n'], bool); self.ish[hold] = True
        self.lag = train_lag(a['slot'], a['fid']); self.has = self.lag > 0
        self.e_tr = np.where(self.has, a['slot'] - self.lag, -1)
        self.pool = (prand > 0.5) & (a['M'] >= 50) & ~self.ish & self.has
        self.poolf = np.nonzero(self.pool)[0]
        v = self.A['val']; t = self.A['test']
        self.Q = {'hold': ('train', hold, self.e_tr[hold]),
                  'pool': ('train', self.poolf, self.e_tr[self.poolf]),
                  'val_nat': ('val', np.arange(v['n']), np.full(v['n'], NAT_E, np.int64)),
                  'val_twin': ('val', np.arange(v['n']), twin_e(v['slot'], v['fid'])),
                  'test_nat': ('test', np.arange(t['n']), np.full(t['n'], NAT_E, np.int64))}
        self.es = sorted({int(x) for _, _, e in self.Q.values() for x in np.unique(e) if x >= 0})
        for e in self.es:
            window(e)
        if P is not None:
            P(f'views: holdout feeds {len(hold)} (with a view {int(self.has[hold].sum())}), pool feeds with a view {len(self.poolf)}; newest source slots in use {self.es}')
            for q, (sp, sel, e) in self.Q.items():
                sl = self.A[sp]['slot'][sel]; lg = np.where(e >= 0, sl - e, -1); u, n = np.unique(lg, return_counts=True)
                P(f'   {q}: {len(sel)} feeds, rows {int(self.A[sp]["M"][sel].sum())}; feeds by lag (-1 = no view) {dict(zip(u.tolist(), n.tolist()))}')

    def rows(self, q):
        """(flat row index into the split's candidate arrays, local feed index per row, local offsets, M) of query set q"""
        sp, sel, _ = self.Q[q]; a = self.A[sp]; M = a['M'][sel]; off = np.concatenate([[0], np.cumsum(M)[:-1]]).astype(np.int64)
        rows = np.repeat(a['co'][sel] - off, M) + np.arange(int(M.sum()), dtype=np.int64)
        return rows, np.repeat(np.arange(len(sel)), M), off, M

    def row_e(self, q):
        sp, sel, e = self.Q[q]; return np.repeat(e, self.A[sp]['M'][sel])

    def row_slot(self, q):
        sp, sel, e = self.Q[q]; return np.repeat(self.A[sp]['slot'][sel], self.A[sp]['M'][sel])


def arr_path(q, name):
    return ARR / f'{q}__{name}.npy'


def put(q, name, x, dtype=np.float32):
    save_new(arr_path(q, name), np.asarray(x, dtype))


def get(q, name):
    return np.load(arr_path(q, name)).astype(np.float64)


# ---------------------------------------------------------------- the statistic (RULE.txt section 3)
def m_table(a, emax):
    """m_a(e) = E[log((C + a) / (e + a))], C ~ Poisson(e), on a grid (exact pmf sums); returns (grid, values)"""
    from scipy.stats import poisson
    g = np.concatenate([[0.0], np.logspace(-6, np.log10(max(emax, 1.0) * 1.05 + 1.0), 6000)]); out = np.zeros(len(g))
    for i in range(1, len(g)):
        e = g[i]; K = int(np.ceil(e + 12 * np.sqrt(e) + 30)); k = np.arange(K + 1)
        out[i] = float((poisson.pmf(k, e) * np.log((k + a) / (e + a))).sum())
    return g, out


def m_exact(a, e):
    from scipy.stats import poisson
    K = int(np.ceil(e + 12 * np.sqrt(e) + 30)); k = np.arange(K + 1)
    return float((poisson.pmf(k, e) * np.log((k + a) / (e + a))).sum())


def x_stat(c, e, a, mt):
    """item statistic: log((c + a) / (e + a)) - m_a(e); 0 for an unseen item"""
    c = np.asarray(c, np.float64); e = np.asarray(e, np.float64)
    return np.log((c + a) / (e + a)) - np.interp(e, mt[0], mt[1])


def eb_a(c, e):
    """a = clip(1 / tau2, 0.5, 50), tau2 = sum[(c - e)^2 - e] / sum e^2 over items with e > 0; a = 50 if tau2 <= 0"""
    m = e > 0; c = c[m].astype(np.float64); e = e[m].astype(np.float64)
    tau2 = float((((c - e) ** 2) - e).sum() / (e ** 2).sum())
    return (50.0 if tau2 <= 0 else float(np.clip(1.0 / tau2, 0.5, 50.0))), tau2


def strata(expo, view, off, M):
    """exactly critic/c07_condperm.py: 8 within-feed rank bins of a label-free exposure (ties share a bin)"""
    import lens
    return np.minimum((NB * (lens.feed_rk(np.where(view, expo, np.nan), off, M) + 0.5)).astype(np.int64), NB - 1)


@nb.njit(parallel=True, cache=True)
def centre_in_strata(x, st, off, M, nbin):
    """entered value: x - mean of x over the rows of the same feed and the same stratum; NaN rows stay NaN"""
    out = np.full(len(x), np.nan, np.float32)
    for f in nb.prange(len(M)):
        o = off[f]; m = M[f]
        s = np.zeros(nbin); c = np.zeros(nbin)
        for k in range(m):
            v = x[o + k]
            if v == v:
                s[st[o + k]] += v; c[st[o + k]] += 1.0
        for k in range(m):
            v = x[o + k]
            if v == v:
                out[o + k] = v - s[st[o + k]] / c[st[o + k]]
    return out


def perm_in_strata(x, key, rng):
    """one draw: the values x are permuted among the rows that share key (= feed * NB + stratum); rows of a feed are contiguous"""
    ordk = np.argsort(key, kind='stable'); o = np.argsort(key + rng.random(len(key)), kind='stable')
    xp = np.empty_like(x); xp[ordk] = x[o]
    return xp


def load_lens():
    """the lens of record (reports/INPUTS_1116/residual_rows/lens.py, condlib.py), imported unchanged"""
    sys.path.insert(0, '<repo>/reports/INPUTS_1116/residual_rows'); sys.path.insert(0, '<repo>/src')
    _argv = sys.argv; sys.argv = [sys.argv[0], 'val', 'TO16']
    import lens, condlib
    sys.argv = _argv
    nb.set_num_threads(12)
    return lens, condlib
