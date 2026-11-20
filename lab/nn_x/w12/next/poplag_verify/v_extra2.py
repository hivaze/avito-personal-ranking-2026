"""poplag_verify step N (LABELLED, POST-HOC, NO DECISION): does the subset result depend on which half of W8 forms the base pool?
  cd <repo>/nn_x/w12/next/poplag_verify && PV_THREADS=4 CUDA_VISIBLE_DEVICES="" ../../../../.venv/bin/python v_extra2.py"""
import os
os.environ.setdefault('PV_THREADS', '4')
from v_common import *

Mh, oh = np.load(ROOT / 'nn_x/w5/preds/holdout/holdout_meta.npy'); Mh = Mh.astype(np.int64); oh = oh.astype(np.int64)
yh = np.load(ROOT / 'nn_x/w5/preds/holdout/holdout_y.npy').astype(np.float32)
ftr, hold, _ = my_holdout()
hd = (ftr['timestamp'].to_numpy().astype(np.int64)[hold] + TZ) // DAY; late = hd >= SEP6 + 7
ev = lambda s: ndcg_feed(s, yh, oh, Mh)
z = lambda x: zfeed(x, oh, Mh)
Z = {(view, v): z(np.load(STUDY / f'pred_{view}_{v}.npy')) for view in ('lag0', 'priv_v3') for v in W8 + P4}
pool = lambda view, mem: np.mean([Z[(view, v)] for v in mem], 0)
stat = lambda pw, pp, w=0.2: ev((1 - w) * z(pw).astype(np.float64) + w * z(pp).astype(np.float64)) - ev(pw)
f = lambda d, m: f'{np.asarray(d)[m].mean():+.5f} +- {np.asarray(d)[m].std(ddof=1) / np.sqrt(m.sum()):.5f}'
OUT = []
p0, p1 = pool('lag0', P4), pool('priv_v3', P4)
for nm, mem in (('W8 (all 8)', W8), ('kd05w192 x4', W8[:4]), ('kd05t25w192 x4', W8[4:])):
    w0, w1 = pool('lag0', mem), pool('priv_v3', mem); s0, s1 = stat(w0, p0), stat(w1, p1)
    OUT.append(f' base pool {nm:15s} + pop1kd x4, subset {int(late.sum())} feeds: S_0 {f(s0, late)} | S_lag {f(s1, late)} | D {f(s1 - s0, late)} | all 30,000: S_0 {f(s0, np.ones(len(late), bool))} | S_lag {f(s1, np.ones(len(late), bool))} | D {f(s1 - s0, np.ones(len(late), bool))}')
for nm, mem in (('pop1kd s951,952', P4[:2]), ('pop1kd s953,954', P4[2:])):
    w0, w1 = pool('lag0', W8), pool('priv_v3', W8); q0, q1 = pool('lag0', mem), pool('priv_v3', mem); s0, s1 = stat(w0, q0), stat(w1, q1)
    OUT.append(f' W8 + {nm:15s}, subset: S_0 {f(s0, late)} | S_lag {f(s1, late)} | D {f(s1 - s0, late)}')
print('\n'.join(OUT)); (HERE / 'logs' / 'v_extra2.txt').write_text('\n'.join(OUT) + '\n')
