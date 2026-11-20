"""verify_where V4: selected (holdout-picked) vs @last checkpoint, member by member, val and holdout; are the stored selected/@last files
identical when best_step == last step? Then the screen read and R0 recomputed with @last for every member, and with a per-member
'mixed' choice. CPU only, read-only. Output out/v4_ckpt.txt"""
import os, sys, json
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, '<repo>/nn_x/w12/recheck/pool_composition')
from pc_common import *  # noqa
import itertools
HERE = ROOT / 'lab/nn_x/w12/moe/why/verify_where'
m = Meta(); ev = lambda s: ndcg_fast(s, m.yv, m.ov, m.Mv); eh = lambda s: ndcg_fast(s, m.yh, m.oh, m.Mh)
S8 = tuple(range(951, 959)); S4 = S8[:4]; ARMS = {'A': 'kd05w192moeffn', 'B': 'kd05w192moevo', 'C': 'kd05w192b3ff1'}
nm = lambda a, s: f'w5_e10_{a}_s{s}'
MO = ROOT / 'lab/nn_x/w12/moe/out'
def pth(v, suf):
    if (MO / 'preds' / f'nn_{v}{suf}_trainonly_val.npy').exists(): return MO / 'preds' / f'nn_{v}{suf}_trainonly_val.npy', MO / 'preds' / f'holdout/nn_{v}{suf}_holdout.npy', MO / 'logs' / f'{v}.json'
    return W / f'nn_{v}{suf}_trainonly_val.npy', W / f'holdout/nn_{v}{suf}_holdout.npy', ROOT / f'nn_x/w5/logs/{v}.json'
L = ['== member: best_step | val sel / @last / diff | holdout sel / @last / diff | identical files? ==']
Z = {}
for v in [nm('kd05w192', s) for s in S8] + [nm(a, s) for a in ARMS.values() for s in S4]:
    out = []
    for suf in ('', '@last'):
        pv, ph, js = pth(v, suf); a, b = np.load(pv).astype(np.float32), np.load(ph).astype(np.float32); out.append((a, b))
        Z[(v, suf)] = (m.z('val', a), m.z('hold', b))
    bs = json.load(open(pth(v, '')[2])).get('best_step')
    same = np.array_equal(out[0][0], out[1][0]) and np.array_equal(out[0][1], out[1][1])
    vs, vl = ev(Z[(v, '')][0]).mean(), ev(Z[(v, '@last')][0]).mean(); hs, hl = eh(Z[(v, '')][1]).mean(), eh(Z[(v, '@last')][1]).mean()
    L.append(f'  {v:32s} best {bs} | val {vs:.5f} {vl:.5f} {vs - vl:+.5f} | hold {hs:.5f} {hl:.5f} {hs - hl:+.5f} | identical {same}')
def pools(suf_of):
    pl_ = lambda vs: (ev(np.mean([Z[(v, suf_of(v))][0] for v in vs], 0)), eh(np.mean([Z[(v, suf_of(v))][1] for v in vs], 0)))
    subs = list(itertools.combinations(S8, 4)); PS = {s: pl_([nm('kd05w192', x) for x in s]) for s in subs}
    rN = (np.mean([PS[s][0] for s in subs], 0), np.mean([PS[s][1] for s in subs], 0))
    AP = {k: pl_([nm(a, s) for s in S4]) for k, a in ARMS.items()}
    vals = np.array([PS[s][0].mean() for s in subs])
    return PS[S4], rN, AP, vals, subs
for lab, f in (('selected (screen)', lambda v: ''), ('@last', lambda v: '@last')):
    r4, rN, AP, vals, subs = pools(f)
    L.append(f'== {lab}: ref4 val rank {int((vals > vals[subs.index(S4)]).sum()) + 1}/70, ref4 - mean {vals[subs.index(S4)] - vals.mean():+.5f}')
    for k in ARMS:
        L.append(f'  {k}: vs ref4 val {fmt(AP[k][0] - r4[0])} hold {fmt(AP[k][1] - r4[1])} | vs refN val {fmt(AP[k][0] - rN[0])} hold {fmt(AP[k][1] - rN[1])}')
    a0 = np.mean([AP[k][0] for k in ARMS], 0); a1 = np.mean([AP[k][1] for k in ARMS], 0)
    L.append(f'  mean(A,B,C): vs ref4 val {fmt(a0 - r4[0])} hold {fmt(a1 - r4[1])} | vs refN val {fmt(a0 - rN[0])} hold {fmt(a1 - rN[1])}')
    # member level vs 8 refs
    for k, a in ARMS.items():
        dv = np.mean([ev(Z[(nm(a, s), f(nm(a, s)))][0]) for s in S4], 0) - np.mean([ev(Z[(nm('kd05w192', s), f(nm('kd05w192', s)))][0]) for s in S8], 0)
        dh = np.mean([eh(Z[(nm(a, s), f(nm(a, s)))][1]) for s in S4], 0) - np.mean([eh(Z[(nm('kd05w192', s), f(nm('kd05w192', s)))][1]) for s in S8], 0)
        L.append(f'    member mean {k} - 8 refs: val {fmt(dv)} hold {fmt(dh)}')
txt = '\n'.join(L); print(txt); open(HERE / 'out/v4_ckpt.txt', 'w').write(txt + '\n')
