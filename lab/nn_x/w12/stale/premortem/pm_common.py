"""premortem: shared loaders (CPU only; read-only on every file outside nn_x/w12/stale/premortem/).
Conventions = nn_x/w12/kd/build_r3.py: member score = per-feed z (stack_val.feed_rank_z[:, 1]); pool = member mean of z;
branch mix = (1 - w) z(base pool) + w z(pop pool); FINAL = w_ft z(FT branch) + (1 - w_ft) z(TO branch), w_ft = 0.5.
NDCG = per-feed full list, binary gain, pessimistic ties (numba copy of pc_common._ndcg; checked against src/metrics in t0_check.py)."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS', 'POLARS_MAX_THREADS'):
    os.environ[_k] = '12'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['NUMBA_CACHE_DIR'] = '<repo>/nn_x/w12/stale/premortem/cache/numba'
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, '<repo>/src')
sys.path.insert(0, '<repo>/eda/errors/segments')
from pathlib import Path
import numpy as np
import numba as nb
import polars as pl

ROOT = Path('<repo>'); P = ROOT / 'preds'; W5 = ROOT / 'nn_x/w5/preds'; AB = ROOT / 'nn_x/abl/preds'
HERE = ROOT / 'nn_x/w12/stale/premortem'; OUT = HERE / 'out'; CACHE = HERE / 'cache'; LOGS = HERE / 'logs'
WSV = ROOT / 'nn_x/w12/stale/witness/cache/staleval'; VSV = ROOT / 'nn_x/w12/stale/veto/preds'
FLAT = ROOT / 'data/flat'
BOUND = 1790497920
S4 = (951, 952, 953, 954); S4b = (955, 956, 957, 958); S8 = S4 + S4b
G = {
    'W8': [f'w5_e10_{a}_s{s}' for a in ('kd05w192', 'kd05t25w192') for s in S4],
    'W8b': [f'w5_e10_{a}_s{s}' for a in ('kd05w192', 'kd05t25w192') for s in S4b],
    'W16': [f'w5_e10_{a}_s{s}' for a in ('kd05w192', 'kd05t25w192') for s in S8],
    'N8': [f'w5_e10_{a}_s{s}' for a in ('kd05', 'kd05t25') for s in S4],
    'Pd': [f'abl_e10_pop1kd05t25_s{s}' for s in S4],
    'Pw': [f'abl_e10_pop1kd05w192_s{s}' for s in S4],
    'F10': [f'v23_gru_pre2s_s{s}' for s in range(501, 507)],
    'F7x': [f'r6_cd_m02_r{s}' for s in range(101, 108)],
    'F7x70': [f'r6_cd_m02_p70k_r{s}' for s in range(101, 108)],
    'pop1x8': [f'abl_e10_pop1_s{s}' for s in S8],
}
G['S16'] = G['W8'] + G['N8']
G['Px8'] = G['Pd'] + G['Pw']
G['G3'] = G['F10'] + G['F7x'] + G['F7x70']
G['C3'] = G['G3'] + G['pop1x8']


def path(v, kind):
    """kind: val_to, test_to, val_ft, test_ft, hold, sv (CPU stale-val, train-only), folds (per-fold test z, float16)."""
    if kind in ('val_to', 'test_to'):
        suf = 'trainonly_val' if kind == 'val_to' else 'trainonly_test'
        for d in (P, W5 if v.startswith('w5_') else AB):
            p = d / f'nn_{v}_{suf}.npy'
            if p.exists(): return p
        return P / f'nn_{v}_{suf}.npy'
    if kind in ('val_ft', 'test_ft', 'folds'):
        suf = {'val_ft': 'val_oof', 'test_ft': 'test', 'folds': 'test_folds'}[kind]
        for d in (P, W5 if v.startswith('w5_') else AB):
            p = d / f'nn_{v}_n2cpv3s3_{suf}.npy'
            if p.exists(): return p
        return P / f'nn_{v}_n2cpv3s3_{suf}.npy'
    if kind == 'hold':
        if v.startswith('w5_'): return W5 / 'holdout' / f'nn_{v}_holdout.npy'
        if v.startswith('abl_'): return AB / 'holdout' / f'nn_{v}_holdout.npy'
        return P / 'holdout' / f'nn_{v}_holdout.npy'
    if kind == 'sv':
        p = WSV / f'nn_{v}_staleval.npy'
        if p.exists(): return p
        return VSV / f'cpu_{v}_staleval.npy'
    raise KeyError(kind)


@nb.njit(parallel=True, cache=True)
def _ndcg(s, y, off, M):
    n = len(M); out = np.ones(n, np.float64)
    for f in nb.prange(n):
        o, m = off[f], M[f]
        sv = s[o:o + m]; yv = y[o:o + m]
        npos = 0
        for k in range(m):
            if yv[k] > 0: npos += 1
        if npos == 0: continue
        order = np.argsort(-sv.astype(np.float64) + 0.0, kind='mergesort')
        dcg = 0.0
        i = 0
        while i < m:
            j = i
            while j + 1 < m and sv[order[j + 1]] == sv[order[i]]: j += 1
            if j == i:
                if yv[order[i]] > 0: dcg += 1.0 / np.log2(i + 2.0)
            else:
                cp = 0
                for k in range(i, j + 1):
                    if yv[order[k]] > 0: cp += 1
                for k in range(j - cp + 1, j + 1): dcg += 1.0 / np.log2(k + 2.0)
            i = j + 1
        idcg = 0.0
        for k in range(npos): idcg += 1.0 / np.log2(k + 2.0)
        out[f] = dcg / idcg
    return out


@nb.njit(parallel=True, cache=True)
def _feed_z(off, M, x):
    out = np.empty(len(x), np.float32)
    for f in nb.prange(len(M)):
        o, m = off[f], M[f]
        v = x[o:o + m]
        mu = v.mean(); sd = v.std() + 1e-9
        for k in range(m):
            out[o + k] = (v[k] - mu) / sd
    return out


class Meta:
    def __init__(self, test=True, hold=True):
        from common import feed_meta
        fv, Mv, ov = feed_meta('val'); self.fv = fv
        self.M = {'val': np.asarray(Mv, np.int64)}; self.off = {'val': np.asarray(ov, np.int64)}
        self.y = {'val': np.load(FLAT / 'val/cand_target.npy').astype(np.float32)}
        ts = fv['timestamp'].to_numpy().astype(np.int64); self.vday = (ts + 3 * 3600) // 86400
        self.tue = self.vday == 20718; self.thu = self.vday == 20720
        u = fv['user_id'].to_numpy().astype(np.int64).copy()
        z0 = u == 0; u[z0] = -1 - np.arange(z0.sum())            # user 0 = pooled bucket: every such feed is its own group
        self.vuser = u
        d = pl.read_parquet(ROOT / 'eda/errors/segments/feed_table.parquet')
        assert (d['fidx'].to_numpy() == np.arange(len(Mv))).all() and (d['M'].to_numpy() == Mv).all()
        self.vcold = d['cold_share'].to_numpy().astype(np.float64)
        if test:
            ft, Mt, ot = feed_meta('test'); self.ft = ft
            self.M['test'] = np.asarray(Mt, np.int64); self.off['test'] = np.asarray(ot, np.int64)
            tts = ft['timestamp'].to_numpy().astype(np.int64); self.tts = tts; self.tday = (tts + 3 * 3600) // 86400; self.public = tts < BOUND
            te = pl.read_parquet(ROOT / 'eda/errors/segments/test_segments.parquet'); tf = te['fidx'].to_numpy()
            tc = np.full(len(Mt), np.nan); tc[tf] = te['cold_share'].to_numpy(); self.tcold = tc
        if hold:
            Mh, oh = np.load(W5 / 'holdout/holdout_meta.npy'); self.M['hold'] = Mh.astype(np.int64); self.off['hold'] = oh.astype(np.int64)
            self.y['hold'] = np.load(W5 / 'holdout/holdout_y.npy').astype(np.float32)
        self.n = {k: int(v.sum()) for k, v in self.M.items()}
        self._w = None

    def z(self, sp, x):
        return _feed_z(self.off[sp], self.M[sp], np.ascontiguousarray(x, np.float32))

    def ndcg(self, sp, s):
        return _ndcg(np.ascontiguousarray(s, np.float32), self.y[sp], self.off[sp], self.M[sp])

    def pmix(self):
        if self._w is None:
            import pmix
            self._w = pmix.weights()
        return self._w


SPLIT = {'val_to': 'val', 'val_ft': 'val', 'sv': 'val', 'test_to': 'test', 'test_ft': 'test', 'hold': 'hold'}
_ZC = {}


def mz(meta, v, kind):
    """z of one member (in-process cache)."""
    k = (v, kind)
    if k not in _ZC:
        x = np.load(path(v, kind)).astype(np.float32)
        assert len(x) == meta.n[SPLIT[kind]], (v, kind, len(x)); assert np.isfinite(x).all(), (v, kind)
        _ZC[k] = meta.z(SPLIT[kind], x)
    return _ZC[k]


def pool(meta, names, kind, keep=True):
    acc = np.zeros(meta.n[SPLIT[kind]], np.float64)
    for v in names:
        if keep: acc += mz(meta, v, kind)
        else:
            x = np.load(path(v, kind)).astype(np.float32); acc += meta.z(SPLIT[kind], x)
    return (acc / len(names)).astype(np.float32)


def se(d):
    d = np.asarray(d, np.float64); return float(d.std(ddof=1) / np.sqrt(len(d)))


def wms(x, w=None):
    """(mean, SE); weighted: pmix.wmean_se convention (n_eff)."""
    x = np.asarray(x, np.float64)
    if w is None: return float(x.mean()), se(x)
    m = np.average(x, weights=w); neff = w.sum() ** 2 / (w ** 2).sum()
    return float(m), float(np.sqrt(np.average((x - m) ** 2, weights=w) / neff))


def f5(t): return f'{t[0]:+.5f} ± {t[1]:.5f}'
