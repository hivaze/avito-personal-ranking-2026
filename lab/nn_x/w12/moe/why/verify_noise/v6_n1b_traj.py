"""verify_noise V6 (CPU, read-only): (1) N1b re-read of kd05w224 (and kd05w192ps 951-954) against Ra / Rbar70 with selected and with @last
checkpoints; (2) late-cooldown val/holdout trajectories from the json eval history (member-level, arm mean minus reference mean at the same
step, relative to each run's final eval at 98350), SE over runs: does arm A keep improving after the reference has flattened?"""
import os, sys, json, itertools
for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[k] = '12'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, '<repo>/nn_x/w12/recheck/pool_composition')
from pc_common import Meta, ndcg_fast, ROOT  # noqa
import numpy as np
from pathlib import Path

OUT = ROOT / 'lab/nn_x/w12/moe/why/verify_noise'
SCR = Path('/tmp/scratchpad/why/verify_noise')
W5 = ROOT / 'nn_x/w5'; MOE = ROOT / 'lab/nn_x/w12/moe/out'
nm = lambda a, s: f'w5_e10_{a}_s{s}'
m = Meta()
out = []
P = lambda *a: (print(*a, flush=True), out.append(' '.join(str(x) for x in a)))
se = lambda d: float(np.std(d, ddof=1) / np.sqrt(len(d)))
fmt = lambda d: f'{np.mean(d):+.5f}+-{se(d):.5f}'


def Z(v, suf=''):
    f = SCR / f'z_{v}{suf}.npz'
    if f.exists():
        d = np.load(f); return d['v'], d['h']
    r = (m.z('val', np.load(W5 / 'preds' / f'nn_{v}{suf}_trainonly_val.npy').astype(np.float32)).astype(np.float32),
         m.z('hold', np.load(W5 / 'preds/holdout' / f'nn_{v}{suf}_holdout.npy').astype(np.float32)).astype(np.float32))
    np.savez(f, v=r[0], h=r[1]); return r


ev = lambda s: ndcg_fast(s, m.yv, m.ov, m.Mv)
eh = lambda s: ndcg_fast(s, m.yh, m.oh, m.Mh)
pool = lambda vs, suf='': (ev(np.mean([Z(v, suf)[0] for v in vs], 0)), eh(np.mean([Z(v, suf)[1] for v in vs], 0)))
REF = [nm('kd05w192', s) for s in range(951, 959)]
for suf in ('', '@last'):
    pv, ph = {}, {}
    for c in itertools.combinations(range(8), 4):
        pv[c], ph[c] = pool([REF[i] for i in c], suf)
    Rv = np.mean(list(pv.values()), 0); Rh = np.mean(list(ph.values()), 0); ra = pv[(0, 1, 2, 3)], ph[(0, 1, 2, 3)]
    for a in ('kd05w224', 'kd05w192ps'):
        av, ah = pool([nm(a, s) for s in range(951, 955)], suf)
        P(f'N1b {suf or "selected":8s} {a:11s}: vs Ra val {fmt(av - ra[0])} hold {fmt(ah - ra[1])} | vs Rbar70 val {fmt(av - Rv)} hold {fmt(ah - Rh)}')

P('')
P('== late-cooldown trajectories (json eval history; member val/holdout minus the same run at 98350; arm mean minus reference mean, SE over runs)')


def hist(v):
    p = MOE / 'logs' / f'{v}.json'
    p = p if p.exists() else W5 / 'logs' / f'{v}.json'
    return {h['step']: h for h in json.load(open(p))['history']}


steps = [85000, 87500, 90000, 92500, 95000, 97500]
R = [hist(v) for v in REF]
for k, a in (('A', 'kd05w192moeffn'), ('B', 'kd05w192moevo'), ('C', 'kd05w192b3ff1')):
    Hs = [hist(nm(a, s)) for s in range(951, 955)]
    for key in ('val_ndcg', 'holdout_ndcg'):
        row = []
        for st in steps:
            x = np.array([h[st][key] - h[98350][key] for h in Hs]); y = np.array([h[st][key] - h[98350][key] for h in R])
            d = x.mean() - y.mean(); s_ = np.sqrt(x.var(ddof=1) / len(x) + y.var(ddof=1) / len(y))
            row.append(f'{st}:{d * 1e4:+.1f}({s_ * 1e4:.1f})')
        P(f'  {k} {key:12s} (x1e4, SE): ' + ' '.join(row))
open(OUT / 'v6_n1b_traj.txt', 'w').write('\n'.join(out) + '\n')
