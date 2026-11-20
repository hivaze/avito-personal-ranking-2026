"""why/where: how lucky is the reference 4-pool kd05w192 s951-954? Distribution of the 70 4-subsets of the 8 same-recipe
reference seeds (s951-958) on val and holdout; the 35 disjoint 4-vs-4 splits as a same-recipe null (z under the per-feed SE);
the arms' 4-pools placed in that distribution. CPU, read-only. Output: out/ref_subsets.txt"""
from common_where import *
import itertools
m = Meta()
Z = {}
for v in [nm('kd05w192', s) for s in S8] + [nm(a, s) for a in ARMS.values() for s in S4]:
    pv, ph, _ = paths2(v); Z[v] = (m.z('val', np.load(pv).astype(np.float32)), m.z('hold', np.load(ph).astype(np.float32)))
ev = lambda s: ndcg_fast(s, m.yv, m.ov, m.Mv); eh = lambda s: ndcg_fast(s, m.yh, m.oh, m.Mh)
pf = {}
def pool(sub):
    if sub not in pf:
        vs = [nm('kd05w192', s) for s in sub]
        pf[sub] = (ev(np.mean([Z[v][0] for v in vs], 0)), eh(np.mean([Z[v][1] for v in vs], 0)))
    return pf[sub]
mem = {s: (ev(Z[nm('kd05w192', s)][0]).mean(), eh(Z[nm('kd05w192', s)][1]).mean()) for s in S8}
subs = list(itertools.combinations(S8, 4))
L = []
L.append('=== reference members kd05w192 (val / holdout NDCG) ===')
for s in S8: L.append(f'  s{s}: {mem[s][0]:.5f} / {mem[s][1]:.5f}')
for k, a in ARMS.items():
    L.append(f'  arm {k} members: ' + ' '.join(f's{s} {ev(Z[nm(a, s)][0]).mean():.5f}/{eh(Z[nm(a, s)][1]).mean():.5f}' for s in S4))
vals = np.array([pool(s)[0].mean() for s in subs]); hols = np.array([pool(s)[1].mean() for s in subs])
gain_v = np.array([pool(s)[0].mean() - np.mean([mem[x][0] for x in s]) for s in subs])
gain_h = np.array([pool(s)[1].mean() - np.mean([mem[x][1] for x in s]) for s in subs])
i0 = subs.index(S4)
L.append(f'=== 70 reference 4-subsets: val mean {vals.mean():.5f} sd {vals.std(ddof=1):.5f} min {vals.min():.5f} max {vals.max():.5f}; '
         f'holdout mean {hols.mean():.5f} sd {hols.std(ddof=1):.5f} ===')
L.append(f'  s951-954 (the screen reference): val {vals[i0]:.5f} rank {int((vals > vals[i0]).sum()) + 1}/70 (z_sub {(vals[i0] - vals.mean()) / vals.std(ddof=1):+.2f}); '
         f'holdout {hols[i0]:.5f} rank {int((hols > hols[i0]).sum()) + 1}/70 (z_sub {(hols[i0] - hols.mean()) / hols.std(ddof=1):+.2f})')
L.append(f'  pool gain over mean member: s951-954 val {gain_v[i0]:+.5f} (70-subset mean {gain_v.mean():+.5f} sd {gain_v.std(ddof=1):.5f}); holdout {gain_h[i0]:+.5f} (mean {gain_h.mean():+.5f} sd {gain_h.std(ddof=1):.5f})')
# arms in the distribution
for k, a in ARMS.items():
    av = ev(np.mean([Z[nm(a, s)][0] for s in S4], 0)); ah = eh(np.mean([Z[nm(a, s)][1] for s in S4], 0))
    am_v = np.mean([ev(Z[nm(a, s)][0]).mean() for s in S4]); am_h = np.mean([eh(Z[nm(a, s)][1]).mean() for s in S4])
    L.append(f'  arm {k} 4-pool: val {av.mean():.5f} (rank among the 70 reference subsets {int((vals > av.mean()).sum()) + 1}/71, z_sub {(av.mean() - vals.mean()) / vals.std(ddof=1):+.2f}; pool gain {av.mean() - am_v:+.5f}) | '
             f'holdout {ah.mean():.5f} (rank {int((hols > ah.mean()).sum()) + 1}/71, z_sub {(ah.mean() - hols.mean()) / hols.std(ddof=1):+.2f}; pool gain {ah.mean() - am_h:+.5f})')
# 35 disjoint splits: same-recipe null of a 4-vs-4 contrast under the per-feed SE
zv, zh, dvs, dhs = [], [], [], []
for s in subs:
    if 951 not in s: continue
    o = tuple(x for x in S8 if x not in s)
    dv = pool(s)[0] - pool(o)[0]; dh = pool(s)[1] - pool(o)[1]
    zv.append(dv.mean() / se(dv)); zh.append(dh.mean() / se(dh)); dvs.append(dv.mean()); dhs.append(dh.mean())
zv, zh = np.array(zv), np.array(zh)
L.append(f'=== 35 disjoint same-recipe 4-vs-4 splits (null): val delta sd {np.std(dvs, ddof=1):.5f}, z sd {zv.std(ddof=1):.2f} (|z|>=2: {int((np.abs(zv) >= 2).sum())}/35); '
         f'holdout delta sd {np.std(dhs, ddof=1):.5f}, z sd {zh.std(ddof=1):.2f} (|z|>=2: {int((np.abs(zh) >= 2).sum())}/35); typical per-feed SE val {np.mean([abs(d / z) for d, z in zip(dvs, zv)]):.5f} holdout {np.mean([abs(d / z) for d, z in zip(dhs, zh)]):.5f}')
L.append('    (if the per-feed SE captured all seed noise, z sd would be ~1; the splits are not independent, so this sd is a rough estimate)')
L.append(f'    corr(val delta, holdout delta) over the 35 splits: {np.corrcoef(dvs, dhs)[0, 1]:+.2f}')
txt = '\n'.join(L); print(txt); open(HERE / 'out/ref_subsets.txt', 'w').write(txt + '\n')
