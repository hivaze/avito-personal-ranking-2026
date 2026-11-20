"""verify_where V6: W2a lagged holdout redone with @last checkpoints (GPU replay in out/lag_last, lag0 + priv_v3). Pool and member DiD,
and the 35-split seed null of the DiD. CPU only, read-only. Output out/v6_lag_last.txt"""
import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, '<repo>/nn_x/w12/recheck/pool_composition')
from pc_common import *  # noqa
import itertools
HERE = ROOT / 'lab/nn_x/w12/moe/why/verify_where'; LAG = HERE / 'out/lag_last'
m = Meta(); eh = lambda s: ndcg_fast(s, m.yh, m.oh, m.Mh)
S8 = tuple(range(951, 959)); S4 = S8[:4]; ARMS = {'A': 'kd05w192moeffn', 'B': 'kd05w192moevo', 'C': 'kd05w192b3ff1'}
nm = lambda a, s: f'w5_e10_{a}_s{s}@last'
Z = {}
for view in ('lag0', 'priv_v3'):
    for v in [nm('kd05w192', s) for s in S8] + [nm(a, s) for a in ARMS.values() for s in S4]:
        Z[(view, v)] = m.z('hold', np.load(LAG / f'pred_{view}_{v}.npy'))
pc = {}
def pool(view, vs):
    k = (view, tuple(vs))
    if k not in pc: pc[k] = eh(np.mean([Z[(view, v)] for v in vs], 0))
    return pc[k]
subs = list(itertools.combinations(S8, 4)); L = ['== W2a at @last: arm 4-pool - refN (70-subset mean), lag0 / priv_v3 / DiD; +- per-feed SE ==']
rN = {view: np.mean([pool(view, [nm('kd05w192', x) for x in s]) for s in subs], 0) for view in ('lag0', 'priv_v3')}
dids = []
for k, a in ARMS.items():
    d0 = pool('lag0', [nm(a, s) for s in S4]) - rN['lag0']; d1 = pool('priv_v3', [nm(a, s) for s in S4]) - rN['priv_v3']; dids.append(d1 - d0)
    md = np.mean([(eh(Z[('priv_v3', nm(a, s))]) - eh(Z[('priv_v3', nm('kd05w192', s))])) - (eh(Z[('lag0', nm(a, s))]) - eh(Z[('lag0', nm('kd05w192', s))])) for s in S4], 0)
    L.append(f'  {k}: lag0 {fmt(d0)} priv {fmt(d1)} DiD {fmt(d1 - d0)} | member DiD (same seed) {fmt(md)}')
d = np.mean(dids, 0); L.append(f'  mean of arms: DiD {fmt(d)}')
did, zd = [], []
for s in subs:
    if 951 not in s: continue
    o = tuple(x for x in S8 if x not in s)
    q = (pool('priv_v3', [nm('kd05w192', x) for x in s]) - pool('priv_v3', [nm('kd05w192', x) for x in o])) - (pool('lag0', [nm('kd05w192', x) for x in s]) - pool('lag0', [nm('kd05w192', x) for x in o]))
    did.append(q.mean()); zd.append(q.mean() / se(q))
L.append(f'  seed null of DiD (35 disjoint 4v4 @last): SD {np.std(did, ddof=1):.5f}, z sd {np.std(zd, ddof=1):.2f}')
txt = '\n'.join(L); print(txt); open(HERE / 'out/v6_lag_last.txt', 'w').write(txt + '\n')
