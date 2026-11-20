"""verify_where V7: W2b stale-val at @last (out/stale_last). Arm 4-pool - refN on fresh val and stale val, DiD, member DiD, and the
35-split seed null of the DiD. CPU only, read-only. Output out/v7_stale_last.txt"""
import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, '<repo>/nn_x/w12/recheck/pool_composition')
from pc_common import *  # noqa
import itertools
HERE = ROOT / 'lab/nn_x/w12/moe/why/verify_where'; ST = HERE / 'out/stale_last'; MO = ROOT / 'lab/nn_x/w12/moe/out/preds'
m = Meta(); ev = lambda s: ndcg_fast(s, m.yv, m.ov, m.Mv)
S8 = tuple(range(951, 959)); S4 = S8[:4]; ARMS = {'A': 'kd05w192moeffn', 'B': 'kd05w192moevo', 'C': 'kd05w192b3ff1'}
nm = lambda a, s: f'w5_e10_{a}_s{s}@last'
def fresh(v):
    p = MO / f'nn_{v}_trainonly_val.npy'
    return p if p.exists() else W / f'nn_{v}_trainonly_val.npy'
Z = {}
for v in [nm('kd05w192', s) for s in S8] + [nm(a, s) for a in ARMS.values() for s in S4]:
    Z[('fresh', v)] = m.z('val', np.load(fresh(v)).astype(np.float32)); Z[('stale', v)] = m.z('val', np.load(ST / f'nn_{v}_staleval.npy'))
pc = {}
def pool(view, vs):
    k = (view, tuple(vs))
    if k not in pc: pc[k] = ev(np.mean([Z[(view, v)] for v in vs], 0))
    return pc[k]
subs = list(itertools.combinations(S8, 4))
rN = {vw: np.mean([pool(vw, [nm('kd05w192', x) for x in s]) for s in subs], 0) for vw in ('fresh', 'stale')}
r4 = {vw: pool(vw, [nm('kd05w192', x) for x in S4]) for vw in ('fresh', 'stale')}
L = ['== W2b stale-val at @last: arm 4-pool - refN; fresh / stale / DiD; member DiD vs same-seed ref ==']
L.append(f'  levels: refN fresh {rN["fresh"].mean():.5f} stale {rN["stale"].mean():.5f}; ref4 fresh {r4["fresh"].mean():.5f} stale {r4["stale"].mean():.5f}; ref4 - refN fresh {(r4["fresh"] - rN["fresh"]).mean():+.5f} stale {(r4["stale"] - rN["stale"]).mean():+.5f}')
dd = []; st = []
for k, a in ARMS.items():
    f0 = pool('fresh', [nm(a, s) for s in S4]) - rN['fresh']; f1 = pool('stale', [nm(a, s) for s in S4]) - rN['stale']; dd.append(f1 - f0); st.append(f1)
    md = np.mean([(ev(Z[('stale', nm(a, s))]) - ev(Z[('stale', nm('kd05w192', s))])) - (ev(Z[('fresh', nm(a, s))]) - ev(Z[('fresh', nm('kd05w192', s))])) for s in S4], 0)
    L.append(f'  {k}: fresh {fmt(f0)} stale {fmt(f1)} DiD {fmt(f1 - f0)} | member DiD {fmt(md)} | stale vs ref4 {fmt(pool("stale", [nm(a, s) for s in S4]) - r4["stale"])}')
L.append(f'  mean arms: stale {fmt(np.mean(st, 0))} DiD {fmt(np.mean(dd, 0))}')
did, zd = [], []
for s in subs:
    if 951 not in s: continue
    o = tuple(x for x in S8 if x not in s)
    q = (pool('stale', [nm('kd05w192', x) for x in s]) - pool('stale', [nm('kd05w192', x) for x in o])) - (pool('fresh', [nm('kd05w192', x) for x in s]) - pool('fresh', [nm('kd05w192', x) for x in o]))
    did.append(q.mean()); zd.append(q.mean() / se(q))
L.append(f'  seed null of DiD (35 disjoint 4v4 @last): SD {np.std(did, ddof=1):.5f}, z sd {np.std(zd, ddof=1):.2f}')
txt = '\n'.join(L); print(txt); open(HERE / 'out/v7_stale_last.txt', 'w').write(txt + '\n')
