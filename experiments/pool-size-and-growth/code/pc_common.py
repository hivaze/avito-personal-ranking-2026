# Shared loader of the pool-composition study. It reads the stored predictions of every member,
# turns each into per-feed z-scores, defines a pool as the plain mean of member z-scores, and has
# a numba copy of the per-feed NDCG (pessimistic ties) for fast subset evaluation. The groups()
# function names the member sets used by the other scripts.
# How it was run: imported by the pc_*.py scripts of lab/nn_x/w12/recheck/pool_composition (CPU,
# 12 threads, read-only on every prediction file).
# What it produced: nothing by itself; pc_cache.py built the z-score cache it reads.
# Kept as it ran: paths and names refer to the lab tree (see lab/ and docs/glossary.md).
# Original: lab/nn_x/w12/recheck/pool_composition/pc_common.py
"""Shared loaders for the pool-composition recheck (CPU only, read-only on every prediction file).
Member score = per-feed z (stack_val.feed_rank_z[:,1]) of the train-only prediction; pool = member mean of z; fast NDCG = numba copy of
src/metrics.ndcg_per_feed (pessimistic ties), verified against it in pc_cache.py."""
import os, sys, json
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'POLARS_MAX_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[k] = '12'
sys.path.insert(0, '<repo>/src')
import numpy as np, numba as nb
from pathlib import Path
from common import feed_meta, FLAT
from metrics import ndcg_per_feed
from stack_val import feed_rank_z

ROOT = Path('<repo>'); W = ROOT/'nn_x/w5/preds'; AB = ROOT/'nn_x/abl/preds'; P = ROOT/'preds'
HERE = ROOT/'nn_x/w12/recheck/pool_composition'; CACHE = HERE/'cache'; OUT = HERE/'out'
S4 = (951, 952, 953, 954)


def groups():
    """Group -> member variant names. Snapshot of completed runs at 2026-11-14T19:04Z (out/snapshot_w5_e10_complete.txt)."""
    g = {}
    e = lambda arm, seeds: [f'w5_e10_{arm}_s{s}' for s in seeds]
    # students, d128
    g['kd05'] = e('kd05', S4); g['kd08'] = e('kd08', S4); g['kd10'] = e('kd10', S4); g['kd03'] = e('kd03', S4)
    g['kd05t25'] = e('kd05t25', S4); g['kd08t25'] = e('kd08t25', (951, 952, 953))
    # students, d192
    g['w192'] = e('kd05w192', S4); g['t25w192'] = e('kd05t25w192', S4)
    g['w192new'] = e('kd05w192', (955,)); g['t25w192new'] = e('kd05t25w192', (955, 957))
    g['kd08w192'] = e('kd08w192', (951,))
    g['g2k05'] = e('kd05g2w192', S4); g['g2k03'] = e('kd03g2w192', S4)
    g['kx05'] = e('kx05', S4); g['kx08'] = e('kx08', S4)
    for a in ('kw_drop0', 'kw_e20', 'kw_id', 'kw_hl2', 'kw_ff4'): g[a] = e(a, (951, 952))
    g['t25w256'] = e('kd05t25w256', (951, 952)); g['t25w192b3'] = e('kd05t25w192b3', (951, 952))
    g['pop1kd'] = [f'abl_e10_pop1kd05t25_s{s}' for s in S4]
    # hard-label members
    g['base'] = e('base', (951, 952, 953, 954, 955, 956))
    for k in range(5): g[f'xf{k}'] = e(f'xf{k}', (951, 952, 953))
    g['F10'] = [f'v23_gru_pre2s_s{s}' for s in range(501, 507)]
    g['F7x'] = [f'r6_cd_m02_r{s}' for s in range(101, 108)]
    g['F7x70'] = [f'r6_cd_m02_p70k_r{s}' for s in range(101, 108)]
    g['pop1'] = [f'abl_e10_pop1_s{s}' for s in range(951, 959)]
    return g


def paths(v, suf=''):
    """(val TO, holdout, test TO) paths of a member; suf '' or '@last'."""
    if v.startswith('w5_'):
        return W/f'nn_{v}{suf}_trainonly_val.npy', W/f'holdout/nn_{v}{suf}_holdout.npy', W/f'nn_{v}{suf}_trainonly_test.npy'
    if v.startswith('abl_e10_pop1kd'):
        return AB/f'nn_{v}{suf}_trainonly_val.npy', AB/f'holdout/nn_{v}{suf}_holdout.npy', AB/f'nn_{v}{suf}_trainonly_test.npy'
    if v.startswith('abl_'):
        vp = AB/f'nn_{v}{suf}_trainonly_val.npy'
        if not vp.exists(): vp = P/f'nn_{v}_trainonly_val.npy'
        tp = AB/f'nn_{v}{suf}_trainonly_test.npy'
        if not tp.exists(): tp = P/f'nn_{v}_trainonly_test.npy'
        return vp, AB/f'holdout/nn_{v}{suf}_holdout.npy', tp
    return P/f'nn_{v}_trainonly_val.npy', P/f'holdout/nn_{v}_holdout.npy', P/f'nn_{v}_trainonly_test.npy'


class Meta:
    def __init__(self):
        fv, self.Mv, self.ov = feed_meta('val'); ft, self.Mt, self.ot = feed_meta('test')
        self.fv, self.ft = fv, ft
        self.ov = np.asarray(self.ov, np.int64); self.ot = np.asarray(self.ot, np.int64)
        self.yv = np.load(FLAT/'val/cand_target.npy').astype(np.float32)
        Mh, oh = np.load(W/'holdout/holdout_meta.npy'); self.Mh = Mh.astype(np.int64); self.oh = oh.astype(np.int64)
        self.yh = np.load(W/'holdout/holdout_y.npy').astype(np.float32)
        self.n = dict(val=int(self.Mv.sum()), hold=int(self.Mh.sum()), test=int(self.Mt.sum()))
        self.M = dict(val=self.Mv, hold=self.Mh, test=self.Mt); self.off = dict(val=self.ov, hold=self.oh, test=self.ot)
        self.y = dict(val=self.yv, hold=self.yh)

    def z(self, sp, x):
        return feed_rank_z(self.off[sp], self.M[sp], np.asarray(x, np.float32))[:, 1]


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
        # pessimistic ties: among equal scores negatives first -> sort key (-score, target)
        order = np.argsort(-sv.astype(np.float64) + 0.0, kind='mergesort')
        dcg = 0.0
        # tie handling: walk runs of equal score, put negatives first inside a run
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


def ndcg_fast(s, y, off, M):
    return _ndcg(np.ascontiguousarray(s, np.float32), y, off, M)


class Store:
    """Memmap cache of member z arrays: cache/z_<split>.f32 (n_members x n_rows) + cache/index.json."""
    def __init__(self, meta, mode='r'):
        self.meta = meta; self.idx = json.load(open(CACHE/'index.json')) if (CACHE/'index.json').exists() else {}
        self.mm = {}
        for sp in ('val', 'hold', 'test'):
            f = CACHE/f'z_{sp}.f32'
            if f.exists(): self.mm[sp] = np.memmap(f, np.float32, mode, shape=(self.idx['cap'], meta.n[sp]))

    def get(self, sp, name):
        return self.mm[sp][self.idx['members'][name]]

    def has(self, name): return name in self.idx.get('members', {})

    def pool(self, sp, names, w=None):
        acc = np.zeros(self.meta.n[sp], np.float64)
        if w is None:
            for n_ in names: acc += self.get(sp, n_)
            return (acc / len(names)).astype(np.float32)
        for n_, w_ in zip(names, w): acc += w_ * np.asarray(self.get(sp, n_), np.float64)
        return (acc / np.sum(w)).astype(np.float32)


se = lambda d: float(d.std() / np.sqrt(len(d)))
fmt = lambda d: f'{d.mean():+.5f}+-{se(d):.5f}'
