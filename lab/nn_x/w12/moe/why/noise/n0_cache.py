"""why/noise step 0: build the z / member-NDCG cache for every member used (references, screen arms, historical w192 arms)."""
import time
from nz_common import *

HIST8 = ['kd05t25w192', 'kd08w192', 'kwwd0', 'kwwd01', 'kwwd05', 'kwaux0', 'kd05w192ps']
HIST4 = ['kd05w224', 'kd05w192gru', 'kdmixw192', 'kwaux1', 'kwd2', 'kwh30', 'kwh6', 'kwr32', 'kwsb2', 'kwwd10', 'kw256d2',
         'kd05g2w192', 'kd03g2w192']
names = [nm('kd05w192', s) for s in S8] + [nm(a, s) for a in ARMS.values() for s in S4A]
for a in HIST8:
    names += [nm(a, s) for s in S8]
for a in HIST4:
    names += [nm(a, s) for s in S8 if exists(nm(a, s))]
miss = [v for v in names if not exists(v)]
print('missing:', miss)
names = [v for v in names if exists(v)]
m = meta(); print('val feeds', len(m.Mv), 'rows', m.n['val'], '| holdout feeds', len(m.Mh), 'rows', m.n['hold'], flush=True)
t = time.time()
for i, v in enumerate(names):
    member_ndcg(v)
    if i % 10 == 0:
        print(i, v, f'{time.time() - t:.0f}s', flush=True)
for v in [nm('kd05w192', s) for s in S8] + [nm(a, s) for a in ARMS.values() for s in S4A]:
    member_ndcg(v, '@last')
print('done', len(names), f'{time.time() - t:.0f}s')
for a in HIST4:
    print(a, [s for s in S8 if exists(nm(a, s))])
