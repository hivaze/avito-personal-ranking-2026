"""merge_verify r2: independent helpers (CPU only). Imports nothing from nn_x/w12/next/merge/ and nothing from the first review (v_common.py).
Project primitives used: src/common.feed_meta (feed table), src/metrics.ndcg_per_feed (only as a cross-check of my own NDCG)."""
import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS', 'POLARS_MAX_THREADS'):
    os.environ[_k] = '12'
sys.path.insert(0, '<repo>/src')
from pathlib import Path
import numpy as np, polars as pl
from common import feed_meta, FLAT, FEATS

ROOT = Path('<repo>'); PR = ROOT / 'preds'; HERE = ROOT / 'nn_x/w12/next/merge_verify/r2'
BOUND = 1790497920          # public = test feeds with timestamp < BOUND (src/pmix.py)


class Split:
    def __init__(self, name):
        f, M, off = feed_meta(name); self.name = name
        self.M = np.asarray(M, np.int64); self.off = np.asarray(off, np.int64); self.n = len(self.M); self.N = int(self.M.sum())
        assert (self.off == np.concatenate([[0], np.cumsum(self.M)[:-1]])).all()
        self.fid = np.repeat(np.arange(self.n), self.M); self.Mrow = self.M[self.fid]
        self.feed_id = f['feed_id'].to_numpy().astype(np.int64); self.ts = f['timestamp'].to_numpy().astype(np.int64)
        self.day = (self.ts + 3 * 3600) // 86400
        self.public = self.ts < BOUND if name == 'test' else np.zeros(self.n, bool)

    def fsum(self, x): return np.add.reduceat(np.asarray(x, np.float64), self.off)

    def z(self, x):
        """per-feed z, population sd, float64, no epsilon (constant feed -> 0)."""
        x = np.asarray(x, np.float64); d = x - (self.fsum(x) / self.M)[self.fid]
        sd = np.sqrt(self.fsum(d * d) / self.M); sd = np.where(sd > 0, sd, 1.0)
        return d / sd[self.fid]

    def rank(self, x, key=None):
        """within-feed ascending ordinal rank 0..M-1 (ties by key, else by row position)."""
        x = np.asarray(x, np.float64)
        o = np.lexsort((x, self.fid)) if key is None else np.lexsort((key, x, self.fid))
        r = np.empty(self.N, np.float64); r[o] = np.arange(self.N) - self.off[self.fid[o]]
        return r

    def rf(self, za, zb):
        """per-feed Pearson of two per-feed-z vectors."""
        return self.fsum(za * zb) / self.M

    def read_parquet(self, path):
        key = pl.DataFrame({'feed_id': self.feed_id[self.fid], 'item_id': np.load(FLAT / self.name / 'cand_item_id.npy').astype(np.int64),
                            'row': np.arange(self.N, dtype=np.int64)})
        df = pl.read_parquet(path).select(pl.col('feed_id').cast(pl.Int64), pl.col('item_id').cast(pl.Int64), pl.col('score').cast(pl.Float64))
        j = key.join(df, on=['feed_id', 'item_id'], how='left').sort('row'); s = j['score'].to_numpy()
        assert len(df) == self.N and np.isfinite(s).all(), path
        return s

    def cold(self):
        return np.nan_to_num(np.asarray(np.load(FEATS / self.name / 'it_hist_all_luo.npy'), np.float64), nan=0.0) == 0


def my_ndcg(S, s, y):
    """own NDCG per feed: binary gain, log2 discount, pessimistic ties (negatives first inside a tie), feeds without positives -> 1."""
    s = np.asarray(s, np.float64); o = np.lexsort((y, -s, S.fid)); ys = y[o]
    pos = np.arange(S.N) - S.off[S.fid]; disc = 1.0 / np.log2(pos + 2.0)
    dcg = S.fsum(ys * disc); npos = S.fsum(y).astype(np.int64)
    cd = np.concatenate([[0.0], np.cumsum(1.0 / np.log2(np.arange(int(S.M.max())) + 2.0))]); idcg = cd[npos]
    return np.where(npos > 0, dcg / np.where(idcg > 0, idcg, 1.0), 1.0)


se = lambda d: float(np.std(d) / np.sqrt(len(d)))
pm = lambda d, k=5: f'{np.mean(d):+.{k}f} ± {se(d):.{k}f}'


def fin(S, tag):
    """candidate as built: 0.5 z(FT pool) + 0.5 z(TO pool)."""
    return 0.5 * S.z(np.load(PR / f'ens_{tag}_ft_{S.name}.npy')) + 0.5 * S.z(np.load(PR / f'ens_{tag}_to_{S.name}.npy'))


def partner_table(V, T):
    """the 10 partners of long_bets/p4_partner.py that have val and test files -> name: (val z, test z)."""
    S4 = (951, 952, 953, 954); c = {}
    for tag, nm in (('Q2a8s3', 'C3'), ('G3', 'G3'), ('Pall', 'Pall'), ('Cd', 'Cd'), ('S16kd', 'S16kd')):
        c[nm] = (V.z(fin(V, tag)), T.z(fin(T, tag)))
    def fam(ms, d, pre='nn_'):
        return (V.z(np.mean([V.z(np.load(d / f'{pre}{m}_trainonly_val.npy')) for m in ms], 0)), T.z(np.mean([T.z(np.load(d / f'{pre}{m}_trainonly_test.npy')) for m in ms], 0)))
    c['GRU x6'] = fam([f'v23_gru_pre2s_s{i}' for i in range(501, 507)], PR)
    c['base x4'] = fam([f'w5_e10_base_s{s}' for s in S4], PR)
    cb = ROOT / 'nn_x/cb26/data'; c['CatBoost'] = (V.z(np.load(cb / 'e3_xm_yeti_val.npy')), T.z(np.load(cb / 'e3_xm_yeti_test.npy')))
    A = ROOT / 'archive/preds'; c['XGBoost'] = (V.z(np.load(A / 'xgb_tr1m_ndcg_val.npy')), T.z(np.load(A / 'xgb_tr1m_ndcg_test.npy')))
    c['LLM x2'] = fam([f'llm06_L1_e4_s{s}' for s in (23, 24)], ROOT / 'nn_x/llm/preds', '')
    return c
