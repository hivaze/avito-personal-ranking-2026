"""why/where W1 robustness: is the arms' cold-positive deficit (and warm-positive gain) beyond same-recipe seed noise?
Statistic T = mean(delta | no positive seen in training) - mean(delta | all positives seen), per split.
Null: the 35 disjoint 4-vs-4 splits of the 8 same-recipe references (kd05w192 s951-958), T of (pool A - pool B); its SD scaled by
sqrt(0.75) approximates the SD of an arm-4-pool-minus-refN contrast (var 4-pool + var refN ~ 1.5 var 4-pool vs 2 var for 4-vs-4).
Also member level: mean over the arm's 4 members minus mean over the 8 reference members (no pooling luck). CPU, read-only.
Output out/robust_cold.txt"""
from common_where import *
import itertools
m = Meta(); P = perfeed(m)
sv = dict(np.load(HERE / 'out/seg_val.npz')); sh = dict(np.load(HERE / 'out/seg_hold.npz'))
SEG = {'v': (sv['pos_seen'] <= 0, sv['pos_seen'] >= 1, sv['M'] < 50), 'h': (sh['pos_seen'] <= 0, sh['pos_seen'] >= 1, None)}
Z = {}
for v in [nm('kd05w192', s) for s in S8]:
    pv, ph, _ = paths2(v); Z[v] = (m.z('val', np.load(pv).astype(np.float32)), m.z('hold', np.load(ph).astype(np.float32)))
ev = lambda s: ndcg_fast(s, m.yv, m.ov, m.Mv); eh = lambda s: ndcg_fast(s, m.yh, m.oh, m.Mh)
pf = {}
def pool(sub):
    if sub not in pf:
        vs = [nm('kd05w192', s) for s in sub]; pf[sub] = (ev(np.mean([Z[v][0] for v in vs], 0)), eh(np.mean([Z[v][1] for v in vs], 0)))
    return pf[sub]
T = lambda d, sp: d[SEG[sp][0]].mean() - d[SEG[sp][1]].mean()
Tse = lambda d, sp: np.sqrt(se(d[SEG[sp][0]]) ** 2 + se(d[SEG[sp][1]]) ** 2)
nullT = {'v': [], 'h': []}; nullAll = {'v': [], 'h': []}; nullCold = {'v': [], 'h': []}; nullWarm = {'v': [], 'h': []}; nullS = {'v': []}
for s in itertools.combinations(S8, 4):
    if 951 not in s: continue
    o = tuple(x for x in S8 if x not in s)
    for i, sp in enumerate('vh'):
        d = pool(s)[i] - pool(o)[i]
        nullT[sp].append(T(d, sp)); nullAll[sp].append(d.mean()); nullCold[sp].append(d[SEG[sp][0]].mean()); nullWarm[sp].append(d[SEG[sp][1]].mean())
        if sp == 'v': nullS['v'].append(d[SEG['v'][2]].mean() - d[~SEG['v'][2]].mean())
sc = np.sqrt(0.75)
L = ['=== cold-positive statistic T = mean(delta | no positive seen) - mean(delta | all positives seen) ===']
for sp, nmx in (('v', 'VAL'), ('h', 'HOLD')):
    L.append(f'{nmx}: 35 disjoint same-recipe 4-vs-4 splits: SD of T {np.std(nullT[sp], ddof=1):.5f} (mean per-feed SE of T {np.mean([Tse(pool(s)[0 if sp == "v" else 1] - pool(tuple(x for x in S8 if x not in s))[0 if sp == "v" else 1], sp) for s in itertools.combinations(S8, 4) if 951 in s]):.5f}); '
             f'SD of cold-segment delta {np.std(nullCold[sp], ddof=1):.5f}, warm-segment {np.std(nullWarm[sp], ddof=1):.5f}, all {np.std(nullAll[sp], ddof=1):.5f}')
    for k in ARMS:
        d = P[f'{k}_{sp}'] - P[f'refN_{sp}']
        t = T(d, sp); z_emp = t / (sc * np.std(nullT[sp], ddof=1))
        L.append(f'   arm {k} - refN: T {t:+.5f} +- {Tse(d, sp):.5f} (per-feed z {t / Tse(d, sp):+.2f}; vs seed-null z {z_emp:+.2f}) | cold {fmt(d[SEG[sp][0]])} warm {fmt(d[SEG[sp][1]])}')
    d = np.mean([P[f'{k}_{sp}'] for k in ARMS], 0) - P[f'refN_{sp}']; t = T(d, sp)
    L.append(f'   mean(A,B,C) - refN: T {t:+.5f} +- {Tse(d, sp):.5f} | cold {fmt(d[SEG[sp][0]])} warm {fmt(d[SEG[sp][1]])}  (null SD for a 3-arm mean is between {sc * np.std(nullT[sp], ddof=1) / np.sqrt(3):.5f} and {sc * np.std(nullT[sp], ddof=1):.5f}: the arms share refN)')
    # member level
    rm = np.mean([P[f'm:{nm("kd05w192", s)}_{sp}'] for s in S8], 0)
    for k, a in ARMS.items():
        am = np.mean([P[f'm:{nm(a, s)}_{sp}'] for s in S4], 0); d = am - rm; t = T(d, sp)
        L.append(f'   MEMBER level arm {k} (mean of 4) - ref (mean of 8): all {fmt(d)} | cold {fmt(d[SEG[sp][0]])} warm {fmt(d[SEG[sp][1]])} | T {t:+.5f} +- {Tse(d, sp):.5f}')
L.append(f'VAL M<50 minus M>=50 statistic: seed-null SD {np.std(nullS["v"], ddof=1):.5f}')
txt = '\n'.join(L); print(txt); open(HERE / 'out/robust_cold.txt', 'w').write(txt + '\n')
