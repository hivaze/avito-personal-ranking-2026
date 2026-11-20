"""verify_noise V2 (CPU, read-only).
 (1) @last instead of the holdout-selected checkpoint: 4 arm members were selected 8k-13k steps before the end (A s954 87500, B s954 87500,
     C s951 90000, C s952 85000) while all 8 references were selected at 97500/98350. Re-read the arms vs Ra / Rb / Rbar70 with @last for
     every member (val and holdout), and selected-vs-@last per member on val.
 (2) Pool-level exchangeability of seed halves: for each 8-seed recipe, a-half (951-954) pool minus b-half (955-958) pool, val / holdout.
 (3) The 16-pool mix check replaced t25 s955-958; how strong is that 4-group, and what is the mix delta when the arm replaces each of the
     4 batch groups or a random 4-subset of the 16 wide students (60 random subsets, same for every arm)?
"""
import os, sys, json, itertools
for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[k] = '12'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, '<repo>/nn_x/w12/recheck/pool_composition')
from pc_common import Meta, ndcg_fast, ROOT  # noqa
import numpy as np
from pathlib import Path

OUT = ROOT / 'lab/nn_x/w12/moe/why/verify_noise'
SCR = Path('/tmp/scratchpad/why/verify_noise'); SCR.mkdir(parents=True, exist_ok=True)
MOE = ROOT / 'lab/nn_x/w12/moe/out'; W5 = ROOT / 'nn_x/w5'
nm = lambda a, s: f'w5_e10_{a}_s{s}'


def P_(v, suf=''):
    if (MOE / 'preds' / f'nn_{v}{suf}_trainonly_val.npy').exists():
        return MOE / 'preds' / f'nn_{v}{suf}_trainonly_val.npy', MOE / 'preds/holdout' / f'nn_{v}{suf}_holdout.npy'
    return W5 / 'preds' / f'nn_{v}{suf}_trainonly_val.npy', W5 / 'preds/holdout' / f'nn_{v}{suf}_holdout.npy'


m = Meta()
out = []
P = lambda *a: (print(*a, flush=True), out.append(' '.join(str(x) for x in a)))
se = lambda d: float(np.std(d, ddof=1) / np.sqrt(len(d)))
fmt = lambda d: f'{np.mean(d):+.5f}+-{se(d):.5f}'
Zc = {}


def Z(v, suf=''):
    key = v + suf
    if key not in Zc:
        f = SCR / f'z_{key}.npz'
        if f.exists():
            d = np.load(f); Zc[key] = (d['v'], d['h'])
        else:
            pv, ph = P_(v, suf)
            Zc[key] = (m.z('val', np.load(pv).astype(np.float32)).astype(np.float32), m.z('hold', np.load(ph).astype(np.float32)).astype(np.float32))
            np.savez(f, v=Zc[key][0], h=Zc[key][1])
    return Zc[key]


ev = lambda s: ndcg_fast(s, m.yv, m.ov, m.Mv)
eh = lambda s: ndcg_fast(s, m.yh, m.oh, m.Mh)
pool = lambda vs, suf='': (ev(np.mean([Z(v, suf)[0] for v in vs], 0)), eh(np.mean([Z(v, suf)[1] for v in vs], 0)))
REF = [nm('kd05w192', s) for s in range(951, 959)]
ARMS = {'A': 'kd05w192moeffn', 'B': 'kd05w192moevo', 'C': 'kd05w192b3ff1'}
AM = {k: [nm(a, s) for s in range(951, 955)] for k, a in ARMS.items()}
subs = list(itertools.combinations(range(8), 4))

# ---- (1) @last
P('== (1) member val NDCG: selected vs @last (selected = best holdout eval after step 12000)')
for v in REF + sum(AM.values(), []):
    j = json.load(open((MOE / 'logs' / f'{v}.json') if (MOE / 'logs' / f'{v}.json').exists() else (W5 / 'logs' / f'{v}.json')))
    sv, lv = ev(Z(v)[0]).mean(), ev(Z(v, '@last')[0]).mean(); sh, lh = eh(Z(v)[1]).mean(), eh(Z(v, '@last')[1]).mean()
    P(f'  {v:34s} best {j["best_step"]:6d} | val sel {sv:.5f} last {lv:.5f} (last-sel {lv - sv:+.5f}) | hold sel {sh:.5f} last {lh:.5f} (last-sel {lh - sh:+.5f})')
P('')
P('== (1b) pools with @last for EVERY member (arms and references), paired per feed')
pvL, phL = {}, {}
for c in subs:
    pvL[c], phL[c] = pool([REF[i] for i in c], '@last')
cA, cB = (0, 1, 2, 3), (4, 5, 6, 7)
mvL = np.array([pvL[c].mean() for c in subs])
RbarvL = np.mean([pvL[c] for c in subs], 0); RbarhL = np.mean([phL[c] for c in subs], 0)
P(f'  @last ref 4-pools: val mean {mvL.mean():.5f} sd {mvL.std(ddof=1):.5f}; Ra {pvL[cA].mean():.5f} rank {int((mvL < pvL[cA].mean()).sum()) + 1}/70; Ra-Rb val {fmt(pvL[cA] - pvL[cB])} hold {fmt(phL[cA] - phL[cB])}')
res = {}
for k in 'ABC':
    av, ah = pool(AM[k], '@last'); sv_, sh_ = pool(AM[k])
    P(f'  {k} @last: vs Ra val {fmt(av - pvL[cA])} hold {fmt(ah - phL[cA])} | vs Rb val {fmt(av - pvL[cB])} hold {fmt(ah - phL[cB])} | vs Rbar70 val {fmt(av - RbarvL)} hold {fmt(ah - RbarhL)}')
    P(f'     arm pool @last minus arm pool selected: val {fmt(av - sv_)} hold {fmt(ah - sh_)}')
    res[k] = dict(vRa=float((av - pvL[cA]).mean()), vRbar=float((av - RbarvL).mean()), hRbar=float((ah - RbarhL).mean()))

# ---- (2) seed halves per recipe, pool level
P('')
P('== (2) pool-level a-half (951-954) minus b-half (955-958), per 8-seed recipe (paired per feed); exchangeability of seed halves')
REC = ['kd05w192', 'kd05t25w192', 'kd08w192', 'kwwd0', 'kwwd01', 'kwwd05', 'kwaux0', 'kd05w192ps']
H = []
for r in REC:
    a = pool([nm(r, s) for s in range(951, 955)]); b = pool([nm(r, s) for s in range(955, 959)])
    H.append(((a[0] - b[0]).mean(), (a[1] - b[1]).mean()))
    P(f'  {r:12s}: val {fmt(a[0] - b[0])} hold {fmt(a[1] - b[1])}')
H = np.array(H)
P(f'  non-reference recipes (n={len(REC) - 1}): mean a-b val {H[1:, 0].mean():+.5f} (sd {H[1:, 0].std(ddof=1):.5f}) hold {H[1:, 1].mean():+.5f} (sd {H[1:, 1].std(ddof=1):.5f});'
  f' reference a-b val {H[0, 0]:+.5f} = {(H[0, 0] - H[1:, 0].mean()) / H[1:, 0].std(ddof=1):+.2f} sd of the other recipes')

# ---- (3) mix check bias
P('')
P('== (3) 16-pool mix check: 16 wide = kd05w192 s951-958 + kd05t25w192 s951-958; the read swapped out t25 s955-958')
W16 = REF + [nm('kd05t25w192', s) for s in range(951, 959)]
full = pool(W16)
groups = {'ref 951-954 (Ra)': [0, 1, 2, 3], 'ref 955-958 (Rb)': [4, 5, 6, 7], 't25 951-954': [8, 9, 10, 11], 't25 955-958 (read)': [12, 13, 14, 15]}
for g, ix in groups.items():
    p4 = pool([W16[i] for i in ix])
    P(f'  group {g:20s}: own 4-pool val {p4[0].mean():.5f} hold {p4[1].mean():.5f}')
rng = np.random.default_rng(20261119)
rs = [tuple(sorted(rng.choice(16, 4, replace=False))) for _ in range(60)]
for k in 'ABC':
    row = []
    for g, ix in groups.items():
        keep = [W16[i] for i in range(16) if i not in ix]
        mp = pool(keep + AM[k]); row.append(f'{g.split(" (")[0]} val {(mp[0] - full[0]).mean():+.5f} hold {(mp[1] - full[1]).mean():+.5f}')
    dv, dh = [], []
    for c in rs:
        keep = [W16[i] for i in range(16) if i not in c]
        mp = pool(keep + AM[k]); dv.append((mp[0] - full[0]).mean()); dh.append((mp[1] - full[1]).mean())
    P(f'  {k}: ' + ' | '.join(row))
    P(f'     mean over 60 random removed 4-subsets: val {np.mean(dv):+.5f} (sd over subsets {np.std(dv, ddof=1):.5f}) hold {np.mean(dh):+.5f} (sd {np.std(dh, ddof=1):.5f})')
json.dump(res, open(OUT / 'v2_last_halves_mix.json', 'w'))
open(OUT / 'v2_last_halves_mix.txt', 'w').write('\n'.join(out) + '\n')
