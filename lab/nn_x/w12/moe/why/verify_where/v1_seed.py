"""verify_where V1: re-derive R0 from raw prediction files and test whether the reference's val 'luck' is seed-borne
(shared by any model trained with seeds 951-954 = same data order), which would bias arm - refN in the arms' favour.
CPU only, read-only. Output out/v1_seed.txt, cache out/v1_perfeed.npz"""
import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, '<repo>/nn_x/w12/recheck/pool_composition')
from pc_common import *  # noqa
import itertools
HERE = ROOT / 'lab/nn_x/w12/moe/why/verify_where'; (HERE / 'out').mkdir(parents=True, exist_ok=True)
MO = ROOT / 'lab/nn_x/w12/moe/out/preds'
ARMS = {'A': 'kd05w192moeffn', 'B': 'kd05w192moevo', 'C': 'kd05w192b3ff1'}
S4 = (951, 952, 953, 954); S8 = tuple(range(951, 959))
nm = lambda a, s: f'w5_e10_{a}_s{s}'
def pth(v):
    if (MO / f'nn_{v}_trainonly_val.npy').exists(): return MO / f'nn_{v}_trainonly_val.npy', MO / f'holdout/nn_{v}_holdout.npy'
    return W / f'nn_{v}_trainonly_val.npy', W / f'holdout/nn_{v}_holdout.npy'
m = Meta()
ev = lambda s: ndcg_fast(s, m.yv, m.ov, m.Mv); eh = lambda s: ndcg_fast(s, m.yh, m.oh, m.Mh)
names = [nm('kd05w192', s) for s in S8] + [nm('kd05t25w192', s) for s in S8] + [nm(a, s) for a in ARMS.values() for s in S4]
Z = {}
for v in names:
    pv, ph = pth(v); Z[v] = (m.z('val', np.load(pv).astype(np.float32)), m.z('hold', np.load(ph).astype(np.float32)))
print('loaded', len(Z), flush=True)
cache = {}
def pool(vs):
    k = tuple(vs)
    if k not in cache: cache[k] = (ev(np.mean([Z[v][0] for v in vs], 0)), eh(np.mean([Z[v][1] for v in vs], 0)))
    return cache[k]
L = []
def subsets(arm):
    subs = list(itertools.combinations(S8, 4)); P = {s: pool([nm(arm, x) for x in s]) for s in subs}
    rN = (np.mean([P[s][0] for s in subs], 0), np.mean([P[s][1] for s in subs], 0)); return subs, P, rN
for arm in ('kd05w192', 'kd05t25w192'):
    subs, P, rN = subsets(arm)
    vals = np.array([P[s][0].mean() for s in subs]); hols = np.array([P[s][1].mean() for s in subs]); i0 = subs.index(S4)
    mem = {s: (ev(Z[nm(arm, s)][0]).mean(), eh(Z[nm(arm, s)][1]).mean()) for s in S8}
    d4b = (P[S8[4:] if False else tuple(S8[4:])][0] - P[S4][0], P[tuple(S8[4:])][1] - P[S4][1])
    L.append(f'== {arm}: 70 4-subsets val mean {vals.mean():.5f} sd {vals.std(ddof=1):.5f}; holdout mean {hols.mean():.5f} sd {hols.std(ddof=1):.5f}')
    L.append(f'   s951-954 val {vals[i0]:.5f} rank {int((vals > vals[i0]).sum()) + 1}/70 (vs mean {vals[i0] - vals.mean():+.5f}); holdout {hols[i0]:.5f} rank {int((hols > hols[i0]).sum()) + 1}/70 ({hols[i0] - hols.mean():+.5f})')
    L.append(f'   swap s955-958 minus s951-954: val {fmt(d4b[0])} holdout {fmt(d4b[1])}')
    L.append(f'   members val: 951-954 mean {np.mean([mem[s][0] for s in S4]):.5f}, 955-958 mean {np.mean([mem[s][0] for s in S8[4:]]):.5f}; holdout 951-954 {np.mean([mem[s][1] for s in S4]):.5f}, 955-958 {np.mean([mem[s][1] for s in S8[4:]]):.5f}')
    if arm == 'kd05w192': R = dict(subs=subs, P=P, rN=rN, vals=vals, hols=hols)
    else: T = dict(subs=subs, P=P, rN=rN)
# reproduce R0 arm vs refN / ref4
rN = R['rN']; r4 = R['P'][S4]; r4b = R['P'][tuple(S8[4:])]
AP = {k: pool([nm(a, s) for s in S4]) for k, a in ARMS.items()}
L.append('== reproduction of R0 (arm 4-pool selected ckpt):')
for k in ARMS:
    L.append(f'   {k}: vs ref4 val {fmt(AP[k][0] - r4[0])} hold {fmt(AP[k][1] - r4[1])} | vs refN val {fmt(AP[k][0] - rN[0])} hold {fmt(AP[k][1] - rN[1])} | vs ref4b val {fmt(AP[k][0] - r4b[0])} hold {fmt(AP[k][1] - r4b[1])}')
am = [np.mean([AP[k][i] for k in ARMS], 0) for i in (0, 1)]
L.append(f'   mean arms vs refN val {fmt(am[0] - rN[0])} hold {fmt(am[1] - rN[1])} | vs ref4 val {fmt(am[0] - r4[0])} | vs ref4b val {fmt(am[0] - r4b[0])} hold {fmt(am[1] - r4b[1])}')
# seed-sharing test 1: per-feed regression of (X - refN) on (ref4 - refN); under no seed sharing the slope is ~0 by symmetry
L.append('== seed-sharing test 1: per-feed slope/corr of (X4(951-954) - refN) on (ref4 - refN); ~0 if X shares nothing seed-specific with ref4')
def slope(x, y):
    x = x - x.mean(); y = y - y.mean(); b = (x * y).sum() / (x * x).sum(); r = np.corrcoef(x, y)[0, 1]
    # SE of slope via per-feed residual (feeds independent)
    res = y - b * x; sb = np.sqrt(((x * res) ** 2).sum()) / (x * x).sum(); return b, sb, r
for sp, i in (('val', 0), ('hold', 1)):
    e4 = r4[i] - rN[i]
    for k in ARMS:
        b, sb, r = slope(e4, AP[k][i] - rN[i]); L.append(f'   {sp} arm {k}: slope {b:+.3f} +- {sb:.3f} corr {r:+.3f}')
    t4 = T['P'][S4]; tN = T['rN']
    b, sb, r = slope(e4, t4[i] - tN[i]); L.append(f'   {sp} t25 4-pool(951-954) - t25N: slope {b:+.3f} +- {sb:.3f} corr {r:+.3f}   (positive = seed/data-order effects shared across recipes)')
    b, sb, r = slope(e4, t4[i] - rN[i]); L.append(f'   {sp} t25 4-pool(951-954) - refN: slope {b:+.3f} +- {sb:.3f} corr {r:+.3f}')
# seed-sharing test 2: row-level z correlation of members, same seed vs other seeds
L.append('== seed-sharing test 2: corr of member rank-z (val rows) after removing the 8-ref mean z; same seed vs different seed')
zr = np.mean([Z[nm('kd05w192', s)][0] for s in S8], 0)
def rc(a, b): x = a - zr; y = b - zr; return float(np.corrcoef(x, y)[0, 1])
for lab, arm in list(ARMS.items()) + [('t25', 'kd05t25w192')]:
    same = [rc(Z[nm(arm, s)][0], Z[nm('kd05w192', s)][0]) for s in S4]
    diff = [rc(Z[nm(arm, s)][0], Z[nm('kd05w192', t)][0]) for s in S4 for t in S8 if t != s]
    L.append(f'   {lab}: same-seed corr {np.mean(same):+.4f} (each {", ".join(f"{x:+.3f}" for x in same)}) | different-seed corr {np.mean(diff):+.4f} (min {min(diff):+.3f} max {max(diff):+.3f})')
same = []; diff = []
for s in S8:
    for t in S8:
        if s < t: diff.append(rc(Z[nm('kd05w192', s)][0], Z[nm('kd05w192', t)][0]))
L.append(f'   ref-ref different seeds corr {np.mean(diff):+.4f} (residualised on the 8-ref mean, so slightly negative by construction: expect ~-1/7 = -0.143 for iid)')
# test 3: the 4 seed-matched member pairs - does the ref member's val deviation predict the arm member's?
L.append('== seed-sharing test 3: member val NDCG by seed (arm member vs same-seed ref member)')
for lab, arm in list(ARMS.items()) + [('t25', 'kd05t25w192')]:
    ss = S4 if lab != 't25' else S8
    x = np.array([ev(Z[nm('kd05w192', s)][0]).mean() for s in ss]); y = np.array([ev(Z[nm(arm, s)][0]).mean() for s in ss])
    L.append(f'   {lab}: corr over {len(ss)} seeds {np.corrcoef(x, y)[0, 1]:+.2f}')
txt = '\n'.join(L); print(txt); open(HERE / 'out/v1_seed.txt', 'w').write(txt + '\n')
np.savez(HERE / 'out/v1_perfeed.npz', r4v=r4[0], r4h=r4[1], r4bv=r4b[0], r4bh=r4b[1], rNv=rN[0], rNh=rN[1],
         t4v=T['P'][S4][0], t4h=T['P'][S4][1], tNv=T['rN'][0], tNh=T['rN'][1], t4bv=T['P'][tuple(S8[4:])][0], t4bh=T['P'][tuple(S8[4:])][1],
         **{f'{k}v': AP[k][0] for k in ARMS}, **{f'{k}h': AP[k][1] for k in ARMS},
         **{f'mem_{v}_v': ev(Z[v][0]) for v in names}, **{f'mem_{v}_h': eh(Z[v][1]) for v in names})
