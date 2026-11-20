"""why/where W2a: lagged-holdout read. Holdout (30,000 feeds) re-scored on GPU with popularity inputs at lag 0 (replay of the stored
files) and at the test lag (poplag views priv_v3 = private-like, pub_v3 = public-like; the 8 inputs these members use are at
label-free parity with test rows on all 30,000 feeds, gate_v3.txt; the declared gate (which also covers xp_imp_c50, not an input
here) holds on days >= Sep 13, reported as a robustness column).
Pools = member-equal mean of per-feed rank-z (4 members). Contrasts paired per feed:
  D_view = NDCG(arm 4-pool) - NDCG(reference), reference = ref4 (kd05w192 s951-954, the screen's), refN (mean over the 70 4-subsets of
  kd05w192 s951-958); DiD = D_priv - D_lag0 (does the arm's holdout edge vanish under test-lag inputs?).
Also each pool's own level change (lagged - lag 0). CPU, read-only. Output out/lag_read.txt"""
from common_where import *
import itertools
m = Meta()
LAG = HERE / 'out/lag'
ts = np.load(ROOT / 'lab/nn_x/w12/next/poplag/cache/mini/ts.npy').astype(np.int64)
late = ((ts + 3 * 3600) // 86400) >= 20709
assert len(late) == len(m.Mh)
eh = lambda s: ndcg_fast(s, m.yh, m.oh, m.Mh)
VIEWS = ('lag0', 'priv_v3', 'pub_v3')
Z = {}
def z(view, v):
    if (view, v) not in Z: Z[(view, v)] = m.z('hold', np.load(LAG / f'pred_{view}_{v}.npy'))
    return Z[(view, v)]
pool = lambda view, vs: eh(np.mean([z(view, v) for v in vs], 0))
P = {}
for view in VIEWS:
    P[(view, 'ref4')] = pool(view, [nm('kd05w192', s) for s in S4]); P[(view, 'ref4b')] = pool(view, [nm('kd05w192', s) for s in S8[4:]])
    acc = np.zeros(len(m.Mh)); subs = list(itertools.combinations(S8, 4))
    for sub in subs: acc += pool(view, [nm('kd05w192', s) for s in sub]) / len(subs)
    P[(view, 'refN')] = acc
    for k, a in ARMS.items(): P[(view, k)] = pool(view, [nm(a, s) for s in S4])
    for s in S8: P[(view, f'r{s}')] = eh(z(view, nm('kd05w192', s)))
    for k, a in ARMS.items():
        for s in S4: P[(view, f'{k}{s}')] = eh(z(view, nm(a, s)))
f = lambda d, sel=None: fmt(d if sel is None else d[sel])
L = [f'=== W2a lagged holdout: 30,000 holdout feeds (late = days >= Sep 13: {int(late.sum())} feeds); +- = 1 SE paired per feed ===']
L.append('pool levels (lag0 GPU replay | priv_v3 | pub_v3) and level change priv - lag0:')
for k in ('ref4', 'ref4b', 'refN', 'A', 'B', 'C'):
    L.append(f'  {k:5s}: {P[("lag0", k)].mean():.5f} | {P[("priv_v3", k)].mean():.5f} | {P[("pub_v3", k)].mean():.5f} | priv-lag0 {f(P[("priv_v3", k)] - P[("lag0", k)])} (late {f(P[("priv_v3", k)] - P[("lag0", k)], late)})')
L.append('member level change priv - lag0 (mean over the 4 members of each arm / the 8 refs):')
for k in ('r', 'A', 'B', 'C'):
    ss = S8 if k == 'r' else S4
    d = np.mean([P[('priv_v3', f'{k}{s}')] - P[('lag0', f'{k}{s}')] for s in ss], 0)
    L.append(f'  {k}: {f(d)} (late {f(d, late)})')
for ref in ('ref4', 'refN'):
    L.append(f'--- arm 4-pool minus {ref} ---')
    for k in ARMS:
        d0 = P[('lag0', k)] - P[('lag0', ref)]; d1 = P[('priv_v3', k)] - P[('priv_v3', ref)]; d2 = P[('pub_v3', k)] - P[('pub_v3', ref)]
        L.append(f'  {k}: lag0 {f(d0)} | priv_v3 {f(d1)} | pub_v3 {f(d2)} | DiD priv-lag0 {f(d1 - d0)} | late: lag0 {f(d0, late)} priv {f(d1, late)} DiD {f(d1 - d0, late)}')
    d0 = np.mean([P[('lag0', k)] for k in ARMS], 0) - P[('lag0', ref)]; d1 = np.mean([P[('priv_v3', k)] for k in ARMS], 0) - P[('priv_v3', ref)]
    L.append(f'  mean of A,B,C: lag0 {f(d0)} | priv_v3 {f(d1)} | DiD {f(d1 - d0)} | late DiD {f(d1 - d0, late)}')
d0 = P[('lag0', 'ref4b')] - P[('lag0', 'ref4')]; d1 = P[('priv_v3', 'ref4b')] - P[('priv_v3', 'ref4')]
L.append(f'--- null: ref4b (s955-958) minus ref4: lag0 {f(d0)} | priv_v3 {f(d1)} | DiD {f(d1 - d0)}')
L.append('member-level DiD (arm member minus same-seed ref member, priv - lag0), mean over 4 seeds:')
for k in ARMS:
    d = np.mean([(P[('priv_v3', f'{k}{s}')] - P[('priv_v3', f'r{s}')]) - (P[('lag0', f'{k}{s}')] - P[('lag0', f'r{s}')]) for s in S4], 0)
    L.append(f'  {k}: {f(d)} (late {f(d, late)})')
txt = '\n'.join(L); print(txt); open(HERE / 'out/lag_read.txt', 'w').write(txt + '\n')
np.savez(HERE / 'out/lag_perfeed.npz', **{f'{a}|{b}': v for (a, b), v in P.items()})
