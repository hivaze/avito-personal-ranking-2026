"""Independent helpers of the merge_verify review (CPU only). Nothing here imports the study's code (nn_x/w12/next/merge/).
Only project primitives are used: src/common.feed_meta, src/metrics.ndcg_per_feed (the official metric)."""
import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS', 'POLARS_MAX_THREADS'):
    os.environ[_k] = '12'
sys.path.insert(0, '<repo>/src')
from pathlib import Path
import numpy as np, polars as pl
from common import feed_meta, FLAT, FEATS
from metrics import ndcg_per_feed

ROOT = Path('<repo>'); P = ROOT / 'preds'
BOUND = 1790497920


class Sp:
    def __init__(self, name):
        f, M, off = feed_meta(name); self.name = name; self.f = f
        self.M = np.asarray(M, np.int64); self.off = np.asarray(off, np.int64); self.n = len(self.M); self.N = int(self.M.sum())
        self.fid = np.repeat(np.arange(self.n), self.M)
        assert (self.off == np.concatenate([[0], np.cumsum(self.M)[:-1]])).all()
        self.ts = f['timestamp'].to_numpy().astype(np.int64); self.day = (self.ts + 3 * 3600) // 86400
        self.Mrow = self.M[self.fid]

    def z(self, x):
        """per-feed z, population sd, float64 (no epsilon; constant feeds -> 0)."""
        x = np.asarray(x, np.float64); mu = np.add.reduceat(x, self.off) / self.M; d = x - mu[self.fid]
        sd = np.sqrt(np.add.reduceat(d * d, self.off) / self.M); sd = np.where(sd > 0, sd, 1.0)
        return d / sd[self.fid]

    def rank(self, x, key=None):
        """within-feed ascending rank 0..M-1; ties broken by key (or by position when key is None)."""
        x = np.asarray(x, np.float64)
        o = np.lexsort((x, self.fid)) if key is None else np.lexsort((key, x, self.fid))
        r = np.empty(self.N, np.float64); r[o] = np.arange(self.N) - self.off[self.fid[o]]
        return r

    def rf(self, za, zb):
        """per-feed Pearson of two per-feed z vectors."""
        return np.add.reduceat(za * zb, self.off) / self.M

    def keyframe(self):
        return pl.DataFrame({'feed_id': self.f['feed_id'].to_numpy().astype(np.int64)[self.fid],
                             'item_id': np.load(FLAT / self.name / 'cand_item_id.npy').astype(np.int64), 'row': np.arange(self.N, dtype=np.int64)})

    def read_file(self, path):
        df = pl.read_parquet(path).select(pl.col('feed_id').cast(pl.Int64), pl.col('item_id').cast(pl.Int64), pl.col('score').cast(pl.Float64))
        j = self.keyframe().join(df, on=['feed_id', 'item_id'], how='left').sort('row'); s = j['score'].to_numpy()
        assert len(df) == self.N and np.isfinite(s).all(), path
        return s


se = lambda d: float(np.std(d) / np.sqrt(len(d)))
pm = lambda d, k=5: f'{np.mean(d):+.{k}f} ± {se(d):.{k}f}'


def fin(S, tag):
    """0.5 z(FT pool) + 0.5 z(TO pool), as the candidate files are built."""
    return 0.5 * S.z(np.load(P / f'ens_{tag}_ft_{S.name}.npy')) + 0.5 * S.z(np.load(P / f'ens_{tag}_to_{S.name}.npy'))


def partners(V, T):
    """the partner list of long_bets/p4_partner.py (the ones with val and test files), as (val z, test z)."""
    S4 = (951, 952, 953, 954); c = {}
    for tag, nm in (('Q2a8s3', 'C3'), ('G3', 'G3'), ('Pall', 'Pall'), ('Cd', 'Cd'), ('S16kd', 'S16kd')):
        c[nm] = (V.z(fin(V, tag)), T.z(fin(T, tag)))
    fam = lambda ms, d=P, pre='nn_': (V.z(np.mean([V.z(np.load(d / f'{pre}{m}_trainonly_val.npy')) for m in ms], 0)), T.z(np.mean([T.z(np.load(d / f'{pre}{m}_trainonly_test.npy')) for m in ms], 0)))
    c['GRUx6'] = fam([f'v23_gru_pre2s_s{i}' for i in range(501, 507)])
    c['basex4'] = fam([f'w5_e10_base_s{s}' for s in S4])
    cb = ROOT / 'nn_x/cb26/data'; c['CatBoost'] = (V.z(np.load(cb / 'e3_xm_yeti_val.npy')), T.z(np.load(cb / 'e3_xm_yeti_test.npy')))
    A = ROOT / 'archive/preds'; c['XGBoost'] = (V.z(np.load(A / 'xgb_tr1m_ndcg_val.npy')), T.z(np.load(A / 'xgb_tr1m_ndcg_test.npy')))
    c['LLMx2'] = fam([f'llm06_L1_e4_s{s}' for s in (23, 24)], ROOT / 'nn_x/llm/preds', '')
    return c
