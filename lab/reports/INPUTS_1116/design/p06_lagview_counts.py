"""DESIGN p06: label-free counts of the lag view cut' = max(slot - L, 5) (BUILD_SPEC.txt section 1) from the stored per-feed lag draws.
Feeds per train day, has share, resulting lag and dense depth; the same for the 30,000 holdout feeds; stale-val twin depth. Output out/p06_lagview_counts.txt."""
import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'POLARS_MAX_THREADS', 'NUMBA_NUM_THREADS'): os.environ[k] = '4'
sys.dont_write_bytecode = True
sys.path.insert(0, '<repo>/nn_x/w12/stale/hz2')
import numpy as np
import hz2_common as hc
out = open('<repo>/reports/INPUTS_1116/design/out/p06_lagview_counts.txt', 'w')
def P(*a):
    s = ' '.join(str(x) for x in a); print(s); out.write(s + '\n')
f = hc.feeds('train'); slot = (hc.day_of(f['timestamp'].to_numpy()) - hc.DAY0).astype(np.int64)
L = np.load(hc.HERE / 'feats_L/train/hz2_lag.npy').astype(np.int64); M = f['M'].to_numpy().astype(np.int64)
cut = np.maximum(slot - L, 5); has = slot >= 6; lag = slot - cut
depth = np.array([(hc.DENSE_SLOTS < c).sum() for c in range(0, 16)])[cut]
hold, _ = hc.holdout_feeds(f); ish = np.zeros(len(f), bool); ish[hold] = True
P(f'train feeds {len(f)}, rows {M.sum()}; has = 1: feeds {has.sum()} ({has.mean():.4f}), rows {M[has].sum()} ({M[has].sum() / M.sum():.4f}); holdout feeds with has = 1: {(has & ish).sum()} of {ish.sum()}')
P('lag draws L (6/7/8/9):', np.bincount(L)[6:10].tolist(), 'shares', np.round(np.bincount(L)[6:10] / len(L), 4).tolist())
P('slot | date | feeds | holdout feeds | has | lag (days): feeds | dense depth: feeds')
for s in sorted(set(slot.tolist())):
    mk = slot == s
    if s < 6: P(f'{s:2d} | Sep {6 + s:2d} | {mk.sum():7d} | {(mk & ish).sum():5d} | 0 | - | -'); continue
    lg = dict(zip(*[x.tolist() for x in np.unique(lag[mk], return_counts=True)])); dp = dict(zip(*[x.tolist() for x in np.unique(depth[mk], return_counts=True)]))
    P(f'{s:2d} | Sep {6 + s:2d} | {mk.sum():7d} | {(mk & ish).sum():5d} | 1 | {lg} | {dp}')
tl = has & (slot >= 11); vl = has & (slot <= 9)
P(f'val-like regime (slots 6-9): {vl.sum()} feeds ({vl.mean():.4f}); test-like regime (slots 11-14): {tl.sum()} feeds ({tl.mean():.4f})')
u, c = np.unique(lag[tl], return_counts=True); P('test-like regime, lag distribution:', dict(zip(u.tolist(), np.round(c / c.sum(), 3).tolist())), f'mean {lag[tl].mean():.2f} d; test feeds (L draws): mean {L.mean():.2f} d')
fv = hc.feeds('val'); sv = (hc.day_of(fv['timestamp'].to_numpy()) - hc.DAY0).astype(np.int64); Lv = hc.assign_lags(fv['feed_id'].to_numpy())
Lv = np.asarray(Lv[1]).astype(np.int64); cv = sv - Lv   # assign_lags returns (part_pub, L, dist, ppub)
dv = np.array([(hc.DENSE_SLOTS < c).sum() for c in range(0, 20)])[cv]
P('stale-val twin: val slot | feeds | cut slot: feeds | dense depth: feeds')
for s in sorted(set(sv.tolist())):
    mk = sv == s
    P(f'{s} | {mk.sum()} | {dict(zip(*[x.tolist() for x in np.unique(cv[mk], return_counts=True)]))} | {dict(zip(*[x.tolist() for x in np.unique(dv[mk], return_counts=True)]))}')
ft = hc.feeds('test'); st = (hc.day_of(ft['timestamp'].to_numpy()) - hc.DAY0).astype(np.int64)
P('test feeds by slot (lag to the dense cutoff slot 14 = slot - 14):', dict(zip(*[x.tolist() for x in np.unique(st, return_counts=True)])))
out.close()
