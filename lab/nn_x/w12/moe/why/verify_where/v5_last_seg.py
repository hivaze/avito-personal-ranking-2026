"""verify_where V5: with @last checkpoints for every member (arms and references), redo the key W1 numbers: cold/warm T, the val-holdout gap,
the mix prediction; the per-feed SE of the selected-minus-@last swing; and the 35-split seed null at @last. CPU only, read-only. Output out/v5_last_seg.txt"""
import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, '<repo>/nn_x/w12/recheck/pool_composition')
from pc_common import *  # noqa
import itertools
HERE = ROOT / 'lab/nn_x/w12/moe/why/verify_where'; WH = ROOT / 'lab/nn_x/w12/moe/why/where'
sv = dict(np.load(WH / 'out/seg_val.npz')); sh = dict(np.load(WH / 'out/seg_hold.npz'))
m = Meta(); ev = lambda s: ndcg_fast(s, m.yv, m.ov, m.Mv); eh = lambda s: ndcg_fast(s, m.yh, m.oh, m.Mh)
S8 = tuple(range(951, 959)); S4 = S8[:4]; ARMS = {'A': 'kd05w192moeffn', 'B': 'kd05w192moevo', 'C': 'kd05w192b3ff1'}
nm = lambda a, s: f'w5_e10_{a}_s{s}'
MO = ROOT / 'lab/nn_x/w12/moe/out/preds'
def pth(v, suf):
    if (MO / f'nn_{v}{suf}_trainonly_val.npy').exists(): return MO / f'nn_{v}{suf}_trainonly_val.npy', MO / f'holdout/nn_{v}{suf}_holdout.npy'
    return W / f'nn_{v}{suf}_trainonly_val.npy', W / f'holdout/nn_{v}{suf}_holdout.npy'
Z = {}
for suf in ('', '@last'):
    for v in [nm('kd05w192', s) for s in S8] + [nm(a, s) for a in ARMS.values() for s in S4]:
        pv, ph = pth(v, suf); Z[(v, suf)] = (m.z('val', np.load(pv).astype(np.float32)), m.z('hold', np.load(ph).astype(np.float32)))
SEG = {'v': (sv['pos_seen'] <= 0, sv['pos_seen'] >= 1), 'h': (sh['pos_seen'] <= 0, sh['pos_seen'] >= 1)}
T = lambda d, sp: d[SEG[sp][0]].mean() - d[SEG[sp][1]].mean()
Tse = lambda d, sp: np.sqrt(se(d[SEG[sp][0]]) ** 2 + se(d[SEG[sp][1]]) ** 2)
L = []
R = {}
for suf in ('', '@last'):
    pl_ = lambda vs: (ev(np.mean([Z[(v, suf)][0] for v in vs], 0)), eh(np.mean([Z[(v, suf)][1] for v in vs], 0)))
    subs = list(itertools.combinations(S8, 4)); PS = {s: pl_([nm('kd05w192', x) for x in s]) for s in subs}
    rN = (np.mean([PS[s][0] for s in subs], 0), np.mean([PS[s][1] for s in subs], 0))
    AP = {k: pl_([nm(a, s) for s in S4]) for k, a in ARMS.items()}
    R[suf] = (PS, rN, AP)
    tag = 'selected' if suf == '' else '@last'
    L.append(f'== {tag}: arm - refN, cold (no positive seen) / warm (all seen) / T ==')
    for k in ARMS:
        row = []
        for i, sp in enumerate('vh'):
            d = AP[k][i] - rN[i]; row.append(f'{sp}: all {fmt(d)} cold {fmt(d[SEG[sp][0]])} warm {fmt(d[SEG[sp][1]])} T {T(d, sp):+.5f}+-{Tse(d, sp):.5f}')
        L.append(f'  {k}: ' + ' || '.join(row))
    dv = np.mean([AP[k][0] for k in ARMS], 0) - rN[0]; dh = np.mean([AP[k][1] for k in ARMS], 0) - rN[1]
    L.append(f'  mean: val {fmt(dv)} T {T(dv, "v"):+.5f}+-{Tse(dv, "v"):.5f} | hold {fmt(dh)} T {T(dh, "h"):+.5f}+-{Tse(dh, "h"):.5f}')
    for sub, sel in (('all val', np.ones(len(dv), bool)), ('val M>=50', sv['M'] >= 50)):
        g = dv[sel].mean() - dh.mean(); gs = np.sqrt(se(dv[sel]) ** 2 + se(dh) ** 2)
        L.append(f'  {sub}: val - holdout gap {g:+.5f}+-{gs:.5f} (z {g / gs:+.2f})')
    # 35-split null z sd at this checkpoint choice
    zv, zh = [], []
    for s in subs:
        if 951 not in s: continue
        o = tuple(x for x in S8 if x not in s); a, b = PS[s][0] - PS[o][0], PS[s][1] - PS[o][1]; zv.append(a.mean() / se(a)); zh.append(b.mean() / se(b))
    L.append(f'  seed null (35 disjoint 4v4): z sd val {np.std(zv, ddof=1):.2f} hold {np.std(zh, ddof=1):.2f}')
L.append('== swing selected -> @last, per arm pool and per reference (paired per feed: pool@last - pool@selected) ==')
for k in ARMS:
    L.append(f'  {k}: val {fmt(R["@last"][2][k][0] - R[""][2][k][0])} hold {fmt(R["@last"][2][k][1] - R[""][2][k][1])}')
L.append(f'  refN: val {fmt(R["@last"][1][0] - R[""][1][0])} hold {fmt(R["@last"][1][1] - R[""][1][1])}')
L.append(f'  ref4: val {fmt(R["@last"][0][S4][0] - R[""][0][S4][0])} hold {fmt(R["@last"][0][S4][1] - R[""][0][S4][1])}')
for k in ARMS:
    d = (R['@last'][2][k][0] - R['@last'][1][0]) - (R[''][2][k][0] - R[''][1][0])
    L.append(f'  {k}: change of (arm - refN) on val from selected to @last: {fmt(d)} (z {d.mean() / se(d):+.2f})')
txt = '\n'.join(L); print(txt); open(HERE / 'out/v5_last_seg.txt', 'w').write(txt + '\n')
