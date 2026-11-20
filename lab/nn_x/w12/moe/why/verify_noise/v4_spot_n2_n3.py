"""verify_noise V4 (CPU, read-only): spot checks of N2 (per-feed delta correlations) and N3 (member correlation, pool gain), plus the
pool gain at @last (the noise analyst only used holdout-selected checkpoints)."""
import os, sys, itertools
for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[k] = '12'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, '<repo>/nn_x/w12/recheck/pool_composition')
from pc_common import Meta, ndcg_fast, ROOT  # noqa
import numpy as np
from pathlib import Path

OUT = ROOT / 'lab/nn_x/w12/moe/why/verify_noise'
SCR = Path('/tmp/scratchpad/why/verify_noise')
nm = lambda a, s: f'w5_e10_{a}_s{s}'
m = Meta()
out = []
P = lambda *a: (print(*a, flush=True), out.append(' '.join(str(x) for x in a)))


def Z(v, suf=''):
    d = np.load(SCR / f'z_{v}{suf}.npz'); return d['v'], d['h']


ev = lambda s: ndcg_fast(s, m.yv, m.ov, m.Mv)
eh = lambda s: ndcg_fast(s, m.yh, m.oh, m.Mh)
pool = lambda vs, suf='': (ev(np.mean([Z(v, suf)[0] for v in vs], 0)), eh(np.mean([Z(v, suf)[1] for v in vs], 0)))
REF = [nm('kd05w192', s) for s in range(951, 959)]
AM = {k: [nm(a, s) for s in range(951, 955)] for k, a in (('A', 'kd05w192moeffn'), ('B', 'kd05w192moevo'), ('C', 'kd05w192b3ff1'))}
ra, rb = pool(REF[:4]), pool(REF[4:])
ap = {k: pool(v) for k, v in AM.items()}
for i, sp in ((0, 'val'), (1, 'holdout')):
    dA, dB = ap['A'][i] - ra[i], ap['B'][i] - ra[i]
    P(f'N2 [{sp}] corr(A-Ra, B-Ra) {np.corrcoef(dA, dB)[0, 1]:+.3f} | corr(A-Ra, B-Rb) {np.corrcoef(dA, ap["B"][i] - rb[i])[0, 1]:+.3f}')


def corr64(a, b):
    a = a.astype(np.float64); b = b.astype(np.float64)
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


for suf in ('', '@last'):
    for k, vs in (('Ra', REF[:4]), ('Rb', REF[4:]), ('A', AM['A']), ('B', AM['B']), ('C', AM['C'])):
        rv = np.mean([corr64(Z(a, suf)[0], Z(b, suf)[0]) for a, b in itertools.combinations(vs, 2)])
        rh = np.mean([corr64(Z(a, suf)[1], Z(b, suf)[1]) for a, b in itertools.combinations(vs, 2)])
        pv, ph = pool(vs, suf)
        mv = np.mean([ev(Z(v, suf)[0]) for v in vs], 0); mh = np.mean([eh(Z(v, suf)[1]) for v in vs], 0)
        P(f'N3 {suf or "selected":8s} {k:2s}: member corr val {rv:.4f} hold {rh:.4f} | member val {mv.mean():.5f} pool {pv.mean():.5f} gain {pv.mean() - mv.mean():+.5f}'
          f' | member hold {mh.mean():.5f} pool {ph.mean():.5f} gain {ph.mean() - mh.mean():+.5f}')
# reference 70-subset gain distribution at @last
gv = []
for c in itertools.combinations(range(8), 4):
    vs = [REF[i] for i in c]; pv, _ = pool(vs, '@last'); mv = np.mean([ev(Z(v, '@last')[0]).mean() for v in vs])
    gv.append(pv.mean() - mv)
gv = np.array(gv)
P(f'N3 @last: 70 reference 4-subsets val gain mean {gv.mean():+.5f} sd {gv.std(ddof=1):.5f}')
open(OUT / 'v4_spot_n2_n3.txt', 'w').write('\n'.join(out) + '\n')
