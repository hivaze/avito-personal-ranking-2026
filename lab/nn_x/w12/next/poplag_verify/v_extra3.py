"""poplag_verify step O (LABELLED, POST-HOC, NO DECISION): is the change under lag specific to the pop-light view?
Base = one half of W8 (4 members). Admixture at w 0.2 of (a) the pop1kd pool, (b) the OTHER half of W8 (same full view, same inputs).
Specific effect = [S_pop - S_same] lagged minus [S_pop - S_same] at lag 0, paired per feed.
  cd <repo>/nn_x/w12/next/poplag_verify && PV_THREADS=4 CUDA_VISIBLE_DEVICES="" ../../../../.venv/bin/python v_extra3.py"""
import os
os.environ.setdefault('PV_THREADS', '4')
from v_common import *

Mh, oh = np.load(ROOT / 'nn_x/w5/preds/holdout/holdout_meta.npy'); Mh = Mh.astype(np.int64); oh = oh.astype(np.int64)
yh = np.load(ROOT / 'nn_x/w5/preds/holdout/holdout_y.npy').astype(np.float32)
ftr, hold, _ = my_holdout()
hd = (ftr['timestamp'].to_numpy().astype(np.int64)[hold] + TZ) // DAY; late = hd >= SEP6 + 7; ALL = np.ones(len(hd), np.bool_)
ev = lambda s: ndcg_feed(s, yh, oh, Mh)
z = lambda x: zfeed(x, oh, Mh)
Z = {(view, v): z(np.load(STUDY / f'pred_{view}_{v}.npy')) for view in ('lag0', 'priv_v3') for v in W8 + P4}
pool = lambda view, mem: np.mean([Z[(view, v)] for v in mem], 0)
mix = lambda a, b, w=0.2: ev((1 - w) * z(a).astype(np.float64) + w * z(b).astype(np.float64))
f = lambda d, m: f'{np.asarray(d)[m].mean():+.5f} +- {np.asarray(d)[m].std(ddof=1) / np.sqrt(m.sum()):.5f}'
OUT = []
tot = {}
for nm, base, other in (('base kd05w192 x4, same-view admixture kd05t25w192 x4', W8[:4], W8[4:]), ('base kd05t25w192 x4, same-view admixture kd05w192 x4', W8[4:], W8[:4])):
    r = {}
    for view in ('lag0', 'priv_v3'):
        b = pool(view, base); eb = ev(b)
        r[view] = (mix(b, pool(view, P4)) - eb, mix(b, pool(view, other)) - eb)
    for sc, m in (('subset', late), ('all 30,000', ALL)):
        d0 = r['lag0'][0] - r['lag0'][1]; d1 = r['priv_v3'][0] - r['priv_v3'][1]
        OUT.append(f' {nm} | {sc} ({int(m.sum())} feeds): lag 0: S_pop {f(r["lag0"][0], m)}, S_same {f(r["lag0"][1], m)}, S_pop - S_same {f(d0, m)} | lagged: S_pop {f(r["priv_v3"][0], m)}, S_same {f(r["priv_v3"][1], m)}, S_pop - S_same {f(d1, m)} | change of (S_pop - S_same) under lag {f(d1 - d0, m)} | change of S_same under lag {f(r["priv_v3"][1] - r["lag0"][1], m)}')
        tot.setdefault(sc, []).append(d1 - d0)
for sc, m in (('subset', late), ('all 30,000', ALL)):
    OUT.append(f' mean of the two base halves, {sc}: change of (S_pop - S_same) under lag {f(np.mean(tot[sc], 0), m)}')
print('\n'.join(OUT)); (HERE / 'logs' / 'v_extra3.txt').write_text('\n'.join(OUT) + '\n')
