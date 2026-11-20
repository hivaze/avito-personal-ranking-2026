"""CRITIC c06 (label-free): JOINT distribution of the new walk column with the collaborative inputs the ranker already reads (seq_cv_invd20, seq_cv_pop:
stored train / val / test arrays), on lag-view train rows against val and test rows. The designer's gates (coverage by age, PSI) are marginal.
On val / test rows the walk and seq_cv_* come from the SAME source; on lag-view train rows seq_cv_* is the adopted fresh build (source = all days
before the row) and the walk is 1-8 days older and 5-7 days deep. CPU only, read-only."""
import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS', 'POLARS_MAX_THREADS'): os.environ[k] = '12'
sys.dont_write_bytecode = True
import numpy as np, polars as pl
from pathlib import Path
CR = '<repo>/reports/INPUTS_1116/critic'; ROOT = Path('<repo>'); FC = ROOT / 'nn/feat_cache'; FLAT = ROOT / 'data/flat'
sys.path.insert(0, str(ROOT / 'nn_x/w12/stale/hz2'))
import hz2_common as hc
out = open(CR + '/out/c06_joint.txt', 'w')
def Pr(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); out.write(s + '\n')
W = np.load(CR + '/out/c01_walk.npz')
ft = hc.feeds('train'); hold, _ = hc.holdout_feeds(ft); Mh = ft['M'].to_numpy().astype(np.int64)[hold]; co = ft['cand_off'].to_numpy().astype(np.int64)[hold]
offh = np.concatenate([[0], np.cumsum(Mh)[:-1]]); rows = np.repeat(co - offh, Mh) + np.arange(int(Mh.sum())); slot = W['hold_slot'].astype(np.int64)
fe = hc.feeds('test'); Me = fe['M'].to_numpy().astype(np.int64); epub = np.repeat(fe['timestamp'].to_numpy().astype(np.int64) < hc.PUB_TS, Me)
fv = hc.feeds('val'); Mv = fv['M'].to_numpy().astype(np.int64)
def col(split, name, r=None):
    a = np.load(FC / split / f'{name}.npy', mmap_mode='r'); return np.asarray(a[r]) if r is not None else np.asarray(a)
S = {'train rows, lag view, slots 6-8 (holdout)': (W['hold_lag'][:, 1] > 0, col('train', 'seq_cv_invd20', rows) > 0, col('train', 'seq_cv_pop', rows) > 0, (slot >= 6) & (slot <= 8)),
     'train rows, lag view, slots 11-13 (holdout)': (W['hold_lag'][:, 1] > 0, col('train', 'seq_cv_invd20', rows) > 0, col('train', 'seq_cv_pop', rows) > 0, slot >= 11),
     'val rows (natural)': (W['val_nat'][:, 1] > 0, col('val', 'seq_cv_invd20') > 0, col('val', 'seq_cv_pop') > 0, np.ones(int(Mv.sum()), bool)),
     'val rows (stale-val twin of the walk; seq_cv as stored)': (W['val_twin'][:, 1] > 0, col('val', 'seq_cv_invd20') > 0, col('val', 'seq_cv_pop') > 0, np.ones(int(Mv.sum()), bool)),
     'test rows, public (natural)': (W['test_nat'][:, 1] > 0, col('test', 'seq_cv_invd20') > 0, col('test', 'seq_cv_pop') > 0, epub),
     'test rows, private (natural)': (W['test_nat'][:, 1] > 0, col('test', 'seq_cv_invd20') > 0, col('test', 'seq_cv_pop') > 0, ~epub)}
Pr('rows | walk > 0 | seq_cv_invd20 > 0 | seq_cv_pop > 0 | P(walk > 0 | invd20 > 0) | P(walk = 0 | invd20 > 0) | P(walk > 0 | invd20 = 0, pop > 0) | P(walk > 0 | pop = 0) | P(pop > 0 | walk = 0)')
for k, (w, i, p, m) in S.items():
    w, i, p = w[m], i[m], p[m]
    Pr(f'{k} | {w.mean():.4f} | {i.mean():.4f} | {p.mean():.4f} | {w[i].mean():.4f} | {1 - w[i].mean():.4f} | {w[~i & p].mean():.4f} | {w[~p].mean():.4f} | {p[~w].mean():.4f}')
out.close()
