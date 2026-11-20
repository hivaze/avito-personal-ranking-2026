"""DESIGN p03: label-free coverage of the stored RP3beta walk (C09) by source depth: non-zero share on holdout rows by train day
(as-built causal source = train days before the row's day, so depth = number of dense days before it), and on val rows by val day.
Also the share of feeds with at least one non-zero candidate. Output out/p03_walk_coverage.txt. CPU only, read-only."""
import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'POLARS_MAX_THREADS', 'NUMBA_NUM_THREADS'): os.environ[k] = '12'
sys.dont_write_bytecode = True
sys.path.insert(0, '<repo>/nn_x/w12/stale/hz2')
import numpy as np, polars as pl
import hz2_common as hc
from pathlib import Path
R = Path('<repo>'); C = R/'nn_x/insights/ccb_1113/scratch/tester3/cache'
out = open(R/'reports/INPUTS_1116/design/out/p03_walk_coverage.txt', 'w')
def P(*a):
    s = ' '.join(str(x) for x in a); print(s); out.write(s + '\n')
h = np.load(C/'c09_hold.npz'); v = np.load(C/'c09_val.npz')
P('keys hold', list(h.keys()), 'val', list(v.keys()))
ftr = hc.feeds("train"); hf, _pr = hc.holdout_feeds(ftr)
P('holdout_feeds type', type(hf), getattr(hf, 'shape', None))
Mh, oh = np.load(R/'nn_x/w5/preds/holdout/holdout_meta.npy'); Mh = Mh.astype(np.int64); oh = oh.astype(np.int64)
ts = ftr['timestamp'].to_numpy(); fid = np.asarray(hf, np.int64)
assert len(fid) == len(Mh), (len(fid), len(Mh))
slot = (hc.day_of(ts[fid]) - hc.DAY0).astype(np.int64)
kh = [k for k in h.keys() if 'rp3' in k][0]; x = np.asarray(h[kh], np.float64)
assert len(x) == Mh.sum()
nzrow = (x > 0).astype(np.float64); cs = np.concatenate([[0], np.cumsum(nzrow)])
feed_nz = (cs[oh + Mh] - cs[oh]); dense_before = np.array([int((hc.DENSE_SLOTS < s).sum()) for s in range(15)])
P('HOLDOUT (as built: train days before the row day; lag <= 1 d). slot | date | dense days before | feeds | rows | non-zero row share | feeds with >=1 non-zero')
for s in sorted(set(slot.tolist())):
    mk = slot == s
    P(f'{s:2d} | Sep {6 + s:2d} | {dense_before[s]:2d} | {mk.sum():5d} | {Mh[mk].sum():8d} | {feed_nz[mk].sum() / Mh[mk].sum():.4f} | {(feed_nz[mk] > 0).mean():.4f}')
P(f'all holdout rows: non-zero share {nzrow.mean():.4f}')
fv = hc.feeds('val'); Mv = fv['M'].to_numpy().astype(np.int64) if 'M' in fv.columns else None
kv = [k for k in v.keys() if 'rp3' in k][0]; xv = np.asarray(v[kv], np.float64)
if Mv is None or Mv.sum() != len(xv):
    Mv = np.load(R/'data/flat/val/M.npy').astype(np.int64) if (R/'data/flat/val/M.npy').exists() else Mv
ov = np.concatenate([[0], np.cumsum(Mv)[:-1]]); csv = np.concatenate([[0], np.cumsum((xv > 0).astype(np.float64))])
fnz = csv[ov + Mv] - csv[ov]; sv = (hc.day_of(fv['timestamp'].to_numpy()) - hc.DAY0).astype(np.int64)
P('VAL (all-train source + earlier val; 12 dense days; lag = slot - 14). slot | date | feeds | non-zero row share | feeds with >=1 non-zero')
for s in sorted(set(sv.tolist())):
    mk = sv == s
    P(f'{s:2d} | Sep {6 + s:2d} | {mk.sum():5d} | {fnz[mk].sum() / Mv[mk].sum():.4f} | {(fnz[mk] > 0).mean():.4f}')
P(f'all val rows: non-zero share {(xv > 0).mean():.4f}')
out.close()
