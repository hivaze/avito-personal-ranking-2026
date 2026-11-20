"""verify_where V9: at @last, arm 4-pools placed among the 70 kd05w192 4-subsets and vs the kd05t25w192 70-subset mean / 16-student mid
reference. CPU only, read-only. Output out/v9_last_rank.txt"""
import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, '<repo>/nn_x/w12/recheck/pool_composition')
from pc_common import *  # noqa
import itertools
HERE = ROOT / 'lab/nn_x/w12/moe/why/verify_where'; MO = ROOT / 'lab/nn_x/w12/moe/out/preds'
m = Meta(); ev = lambda s: ndcg_fast(s, m.yv, m.ov, m.Mv); eh = lambda s: ndcg_fast(s, m.yh, m.oh, m.Mh)
S8 = tuple(range(951, 959)); S4 = S8[:4]; ARMS = {'A': 'kd05w192moeffn', 'B': 'kd05w192moevo', 'C': 'kd05w192b3ff1'}
nm = lambda a, s: f'w5_e10_{a}_s{s}@last'
def pth(v):
    if (MO / f'nn_{v}_trainonly_val.npy').exists(): return MO / f'nn_{v}_trainonly_val.npy', MO / f'holdout/nn_{v}_holdout.npy'
    return W / f'nn_{v}_trainonly_val.npy', W / f'holdout/nn_{v}_holdout.npy'
Z = {}
for v in [nm(a, s) for a in ('kd05w192', 'kd05t25w192') for s in S8] + [nm(a, s) for a in ARMS.values() for s in S4]:
    pv, ph = pth(v); Z[v] = (m.z('val', np.load(pv).astype(np.float32)), m.z('hold', np.load(ph).astype(np.float32)))
pl_ = lambda vs: (ev(np.mean([Z[v][0] for v in vs], 0)), eh(np.mean([Z[v][1] for v in vs], 0)))
subs = list(itertools.combinations(S8, 4)); L = []
PR = {s: pl_([nm('kd05w192', x) for x in s]) for s in subs}; PT = {s: pl_([nm('kd05t25w192', x) for x in s]) for s in subs}
rN = [np.mean([PR[s][i] for s in subs], 0) for i in (0, 1)]; tN = [np.mean([PT[s][i] for s in subs], 0) for i in (0, 1)]
vals = np.array([PR[s][0].mean() for s in subs]); hols = np.array([PR[s][1].mean() for s in subs])
L.append(f'@last kd05w192 70 subsets: val mean {vals.mean():.5f} sd {vals.std(ddof=1):.5f}; holdout mean {hols.mean():.5f} sd {hols.std(ddof=1):.5f}; t25N val {tN[0].mean():.5f} hold {tN[1].mean():.5f}')
for k, a in ARMS.items():
    x = pl_([nm(a, s) for s in S4]); mid = [(rN[i] + tN[i]) / 2 for i in (0, 1)]
    L.append(f'  {k}@last: val {x[0].mean():.5f} rank {int((vals > x[0].mean()).sum()) + 1}/71 | hold {x[1].mean():.5f} rank {int((hols > x[1].mean()).sum()) + 1}/71 | '
             f'vs t25N val {fmt(x[0] - tN[0])} hold {fmt(x[1] - tN[1])} | vs mid16 val {fmt(x[0] - mid[0])} hold {fmt(x[1] - mid[1])}')
txt = '\n'.join(L); print(txt); open(HERE / 'out/v9_last_rank.txt', 'w').write(txt + '\n')
