"""why/where W2b: stale-val read. Val re-predicted (GPU, stored path) with private-like stale inputs:
  staleval  = 8 popularity / co-visitation inputs at private-like coverage + pre2_fe* no-ID score on newly cold rows (project's stale-val)
  stalepop  = only the 8 popularity inputs;  stalepre2 = only the pre2_fe* part.
Contrasts paired per feed: arm 4-pool minus reference (refN = mean over the 70 4-subsets of kd05w192 s951-958; ref4 = s951-954),
fresh val vs stale views; DiD = (arm - ref | stale) - (arm - ref | fresh). Also by positive-coldness segment. CPU, read-only.
Output out/stale_read.txt"""
from common_where import *
import itertools
m = Meta()
ST = HERE / 'out/stale'
sv = dict(np.load(HERE / 'out/seg_val.npz'))
ev = lambda s: ndcg_fast(s, m.yv, m.ov, m.Mv)
Z = {}
def z(view, v):
    if (view, v) not in Z:
        p = paths2(v)[0] if view == 'fresh' else ST / f'nn_{v}_{view}.npy'
        Z[(view, v)] = m.z('val', np.load(p).astype(np.float32))
    return Z[(view, v)]
pool = lambda view, vs: ev(np.mean([z(view, v) for v in vs], 0))
VIEWS = [v for v in ('fresh', 'staleval', 'stalepop', 'stalepre2') if v == 'fresh' or all((ST / f'nn_{nm(a, s)}_{v}.npy').exists() for a in ['kd05w192'] + list(ARMS.values()) for s in S4)]
P = {}
subs = list(itertools.combinations(S8, 4))
for view in VIEWS:
    P[(view, 'ref4')] = pool(view, [nm('kd05w192', s) for s in S4])
    P[(view, 'refN')] = np.mean([pool(view, [nm('kd05w192', s) for s in sub]) for sub in subs], 0)
    for k, a in ARMS.items(): P[(view, k)] = pool(view, [nm(a, s) for s in S4])
    for s in S8: P[(view, f'r{s}')] = ev(z(view, nm('kd05w192', s)))
    for k, a in ARMS.items():
        for s in S4: P[(view, f'{k}{s}')] = ev(z(view, nm(a, s)))
L = [f'=== W2b stale-val (14,423 val feeds; +- = 1 SE paired per feed); views {VIEWS} ===']
L.append('pool levels: ' + ' | '.join(VIEWS))
for k in ('ref4', 'refN', 'A', 'B', 'C'):
    L.append(f'  {k:5s}: ' + ' | '.join(f'{P[(v, k)].mean():.5f}' for v in VIEWS) + ' || change vs fresh: ' + ' '.join(f'{v} {fmt(P[(v, k)] - P[("fresh", k)])}' for v in VIEWS[1:]))
L.append('member mean change vs fresh:')
for k in ('r', 'A', 'B', 'C'):
    ss = S8 if k == 'r' else S4
    L.append(f'  {k}: ' + ' '.join(f'{v} {fmt(np.mean([P[(v, f"{k}{s}")] - P[("fresh", f"{k}{s}")] for s in ss], 0))}' for v in VIEWS[1:]))
cold = sv['pos_seen'] <= 0; warm = sv['pos_seen'] >= 1
for ref in ('refN', 'ref4'):
    L.append(f'--- arm 4-pool minus {ref}: fresh | stale views | DiD (stale - fresh) ---')
    for k in list(ARMS) + ['mean']:
        g = (lambda v: np.mean([P[(v, a)] for a in ARMS], 0)) if k == 'mean' else (lambda v, k=k: P[(v, k)])
        d0 = g('fresh') - P[('fresh', ref)]
        s = f'  {k:4s}: fresh {fmt(d0)}'
        for v in VIEWS[1:]:
            d1 = g(v) - P[(v, ref)]; s += f' | {v} {fmt(d1)} DiD {fmt(d1 - d0)}'
        L.append(s)
        if k == 'mean' and 'staleval' in VIEWS:
            d1 = g('staleval') - P[('staleval', ref)]
            L.append(f'        by segment: positives all seen (n {int(warm.sum())}): fresh {fmt(d0[warm])} stale {fmt(d1[warm])} DiD {fmt((d1 - d0)[warm])} | '
                     f'no positive seen (n {int(cold.sum())}): fresh {fmt(d0[cold])} stale {fmt(d1[cold])} DiD {fmt((d1 - d0)[cold])} | M<50: DiD {fmt((d1 - d0)[m.Mv < 50])} M>=50: DiD {fmt((d1 - d0)[m.Mv >= 50])}')
L.append('member-level DiD (arm member minus same-seed ref member, stale - fresh), mean over 4 seeds:')
for k in ARMS:
    L.append(f'  {k}: ' + ' '.join(f'{v} {fmt(np.mean([(P[(v, f"{k}{s}")] - P[(v, f"r{s}")]) - (P[("fresh", f"{k}{s}")] - P[("fresh", f"r{s}")]) for s in S4], 0))}' for v in VIEWS[1:]))
txt = '\n'.join(L); print(txt); open(HERE / 'out/stale_read.txt', 'w').write(txt + '\n')
