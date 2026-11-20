"""Shared loaders for the G3 error anatomy (CPU only). Import after setting thread env vars."""
import os
for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "POLARS_MAX_THREADS", "NUMBA_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
    os.environ.setdefault(k, "12")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import sys, re, glob, time
from pathlib import Path
import numpy as np, polars as pl
sys.path.insert(0, "<repo>/src")
from metrics import ndcg_per_feed
from stack_val import feed_rank_z

ROOT = Path("<repo>")
FLAT = ROOT / "data/flat"
FEATS = ROOT / "data/feats"
FC = ROOT / "nn_x/r6/feat_cache"
P = ROOT / "preds"
PH = P / "holdout"
OUT = ROOT / "nn_x/insights/errors"
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:6.1f}s]", *a, flush=True)


FAM = {
    'F5cd': ['v15_cd_all_e10', 'v15_cd_all_s81', 'v15_cd_all_s82', 'v15_cd_allprew_s83', 'v15_cd_allprew_s84', 'v15_cd_allprew_s85'],
    'F6': ['v13_lr1_mix1', 'v14_s92', 'v14_s93', 'v14_s94', 'v14_s95', 'v14_s96'],
    'F7': [f'v17_pre2s_s{s}' for s in range(101, 108)],
    'F9': [f'v22_pre2s_lr1_s{s}' for s in range(401, 407)],
    'F10': [f'v23_gru_pre2s_s{s}' for s in range(501, 507)],
    'F12': [f'v26_d192_pre2s_s{s}' for s in range(701, 708)],
    'F7x': [f'r6_cd_m02_r{s}' for s in range(101, 108)],
    'F7x70': [f'r6_cd_m02_p70k_r{s}' for s in range(101, 108)],
    'F7b': [f'r6_cd_c0_r{s}' for s in range(101, 108)],
}


def have(m):
    return all(p.exists() for p in [P / f'nn_{m}_n2cpv3_val_oof.npy', P / f'nn_{m}_n2cpv3_test.npy',
                                    P / f'nn_{m}_trainonly_val.npy', P / f'nn_{m}_trainonly_test.npy', PH / f'nn_{m}_holdout.npy'])


def new_candidates():
    pats = ['v27_gru_m02_s*', 'v28_lr1_m02_s*', 'v29_d192_m02_s*', 'v30_f7_m02_s*']
    c = sorted({re.sub(r'^nn_|_trainonly_val\.npy$', '', Path(f).name) for pat in pats
                for f in glob.glob(str(P / f'nn_{pat}_trainonly_val.npy'))})
    return [m for m in c if '@' not in m and have(m)]


class Val:
    """Val split meta + per-candidate basics."""

    def __init__(self, split='val'):
        self.split = split
        f = pl.read_parquet(FLAT / split / 'feeds.parquet')
        self.f = f
        self.M = f['M'].to_numpy().astype(np.int64)
        self.off = f['cand_off'].to_numpy().astype(np.int64)
        self.n = len(self.M)
        self.fid = np.repeat(np.arange(self.n), self.M)
        self.y = np.load(FLAT / split / 'cand_target.npy').astype(np.float64) if (FLAT / split / 'cand_target.npy').exists() else None
        self.cat = np.load(FLAT / split / 'cand_category_id.npy').astype(np.int64)
        self.npos = np.bincount(self.fid, self.y, self.n) if self.y is not None else None

    def z(self, x):
        return feed_rank_z(self.off, self.M, np.asarray(x, np.float32))[:, 1].astype(np.float64)

    def rk(self, x):
        return feed_rank_z(self.off, self.M, np.asarray(x, np.float32))[:, 0].astype(np.float64)

    def ndcg(self, s):
        return ndcg_per_feed(s, self.y, self.off, self.M)

    def prank(self, s):
        """0-based pessimistic rank of each candidate within its feed (ties: positives placed after negatives)."""
        order = np.lexsort((self.y, -np.asarray(s, np.float64), self.fid))
        r = np.empty(len(order), np.int64)
        r[order] = np.arange(len(order)) - np.repeat(self.off, self.M)
        return r


def load_member(V, m, kind):
    if kind == 'ft':
        return V.z(np.load(P / f'nn_{m}_n2cpv3_val_oof.npy'))
    return V.z(np.load(P / f'nn_{m}_trainonly_val.npy'))


def pool_scores(V, ms, cache):
    """decide5 pool: final = 0.5 z(mean z(ft)) + 0.5 z(mean z(to))."""
    for m in ms:
        if (m, 'ft') not in cache:
            cache[(m, 'ft')] = load_member(V, m, 'ft')
            cache[(m, 'to')] = load_member(V, m, 'to')
    ft = V.z(np.mean([cache[(m, 'ft')] for m in ms], 0))
    to = V.z(np.mean([cache[(m, 'to')] for m in ms], 0))
    return 0.5 * ft + 0.5 * to, ft, to


def se(d):
    d = np.asarray(d, np.float64)
    return d.std(ddof=1) / np.sqrt(len(d))


def fmt_d(d):
    return f"{d.mean():+.5f} ± {se(d):.5f}"
