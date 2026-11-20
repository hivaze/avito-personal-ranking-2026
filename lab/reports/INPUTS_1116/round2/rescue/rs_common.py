"""RESCUE (round 2): shared code. CPU only; threads = RS_NT (default 12). Read-only on every existing file; writes only under
reports/INPUTS_1116/round2/rescue/ (exclusive create for arrays and logs). Definitions: PREDECLARED.txt in this directory.

The view "D5": window(cut) = the last 5 dense slots below cut; train rows cut' = max(slot - L, 5) (nx3's cut file, re-derived here);
val / test natural cut = 14; stale-val twin cut = slot - L (nx3's twin cut file, re-derived here)."""
import os, sys, time, hashlib
NT = int(os.environ.get('RS_NT', '12'))
assert 1 <= NT <= 12
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'POLARS_MAX_THREADS'):
    os.environ[_k] = str(NT)
os.environ['NUMBA_NUM_THREADS'] = '12'     # pool size (the modules imported later set the same value); the threads in USE are nb.set_num_threads(NT)
RS = '<repo>/reports/INPUTS_1116/round2/rescue'
os.environ['NUMBA_CACHE_DIR'] = RS + '/out/numba_cache'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True
import numba as nb
import numpy as np
import polars as pl
from pathlib import Path
nb.set_num_threads(NT)
sys.path.insert(0, '<repo>/nn_x/w12/stale/hz2')
import hz2_common as hc
for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'POLARS_MAX_THREADS'):
    os.environ[_k] = str(NT)
nb.set_num_threads(NT)

ROOT = Path('<repo>'); FLAT = ROOT / 'data/flat'; OUT = Path(RS) / 'out'; ARR = OUT / 'arr'
NX3 = ROOT / 'nn_x/w12/nx3'
SPLITS = ('train', 'val', 'test')
NROWS = {'train': 246745836, 'val': 1389971, 'test': 2904806}
DENSE = hc.DENSE_SLOTS.astype(np.int64)            # 0..8, 11, 12, 13
S12 = np.full(24, -1, np.int64); S12[DENSE] = np.arange(len(DENSE))   # slot -> column 0..11 of the per-slot tables (-1 = not a source day)
NAT_CUT = 14
DEPTH = 5
T0 = time.time()


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


def window(cut):
    """the last DEPTH dense slots strictly below cut (sorted)"""
    d = DENSE[DENSE < cut]
    return d[-DEPTH:]


def wkey(cut):
    return int(sum(1 << int(S12[s]) for s in window(cut)))     # bit mask over the 12 table columns


class Views:
    """feeds tables, the query sets and their cuts / windows"""

    def __init__(self, P):
        self.F = {s: pl.read_parquet(FLAT / s / 'feeds.parquet') for s in SPLITS}
        self.A = {}
        for s in SPLITS:
            f = self.F[s]; ts = f['timestamp'].to_numpy().astype(np.int64)
            assert (f['fidx'].to_numpy() == np.arange(len(f))).all()
            M = f['M'].to_numpy().astype(np.int64); co = f['cand_off'].to_numpy().astype(np.int64)
            assert int(M.sum()) == NROWS[s] and (co == np.concatenate([[0], np.cumsum(M)[:-1]])).all()
            self.A[s] = dict(M=M, co=co, ts=ts, slot=(hc.day_of(ts) - hc.DAY0).astype(np.int64), u=f['user_id'].to_numpy().astype(np.int64),
                             fid=f['feed_id'].to_numpy(), H=f['H'].to_numpy().astype(np.int64), ho=f['hist_off'].to_numpy().astype(np.int64), n=len(f))
        a = self.A['train']
        hold, prand = hc.holdout_feeds(self.F['train']); self.hold = hold
        self.ish = np.zeros(a['n'], bool); self.ish[hold] = True
        tcut = np.load(NX3 / 'out/raw/train_cut_feed.npy').astype(np.int64); has = np.load(NX3 / 'out/raw/train_has_feed.npy').astype(bool)
        L = np.load(hc.HERE / 'feats_L/train/hz2_lag.npy').astype(np.int64)
        assert (has == (a['slot'] >= 6)).all() and (tcut[has] == np.maximum(a['slot'] - L, 5)[has]).all()       # the nx3 cut, re-derived
        self.tcut, self.has = tcut, has
        self.pool = (prand > 0.5) & (a['M'] >= 50) & ~self.ish & has
        self.poolf = np.nonzero(self.pool)[0]
        v = self.A['val']
        vcut = np.load(NX3 / 'out/raw/val_twin_cut_feed.npy').astype(np.int64)
        assert (vcut == v['slot'] - np.asarray(hc.assign_lags(v['fid'])[1]).astype(np.int64)).all()             # the twin cut, re-derived
        self.vcut = vcut
        # query sets: name -> (split, feed idx, cut per feed (-1 = no view))
        self.Q = {'hold': ('train', hold, np.where(has[hold], tcut[hold], -1)),
                  'pool': ('train', self.poolf, tcut[self.poolf]),
                  'val_nat': ('val', np.arange(v['n']), np.full(v['n'], NAT_CUT, np.int64)),
                  'val_twin': ('val', np.arange(v['n']), vcut),
                  'test_nat': ('test', np.arange(self.A['test']['n']), np.full(self.A['test']['n'], NAT_CUT, np.int64))}
        self.cuts = sorted({int(c) for _, _, c in self.Q.values() for c in np.unique(c) if c >= 0})
        self.wkeys = sorted({wkey(c) for c in self.cuts})
        if P is not None:
            P(f'views: holdout feeds {len(hold)} (has = 1 on {int(has[hold].sum())}), pool feeds with has = 1 {len(self.poolf)}; cuts in use {self.cuts}')
            for c in self.cuts:
                P(f'   cut {c:2d}: window slots {window(c).tolist()} (key {wkey(c)})')
            for q, (sp, sel, c) in self.Q.items():
                u, n = np.unique(c, return_counts=True); P(f'   {q}: {len(sel)} feeds, rows {int(self.A[sp]["M"][sel].sum())}; feeds by cut {dict(zip(u.tolist(), n.tolist()))}')

    def rows(self, q):
        """(flat row index into the split's candidate arrays, local feed index per row, local offsets, M) of query set q"""
        sp, sel, _ = self.Q[q]; a = self.A[sp]; M = a['M'][sel]; off = np.concatenate([[0], np.cumsum(M)[:-1]]).astype(np.int64)
        rows = np.repeat(a['co'][sel] - off, M) + np.arange(int(M.sum()), dtype=np.int64)
        return rows, np.repeat(np.arange(len(sel)), M), off, M

    def row_wkey(self, q):
        """window key per row (0 = no view)"""
        sp, sel, c = self.Q[q]; M = self.A[sp]['M'][sel]
        k = np.array([wkey(int(x)) if x >= 0 else 0 for x in range(-1, 16)], np.int64)       # index cut + 1
        return np.repeat(k[c + 1], M)


def arr_path(q, name):
    return ARR / f'{q}__{name}.npy'


def put(q, name, x):
    save_new(arr_path(q, name), np.asarray(x, np.float32))


def get(q, name):
    return np.load(arr_path(q, name)).astype(np.float64)
