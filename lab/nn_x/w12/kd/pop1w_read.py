"""Read-out of the WIDE popularity-light students abl_e10_pop1kd05w192_s951-954 (rule B / A' in nn_x/w5/logs/E_kd_predeclared.txt): train-only level only (not fine-tuned unless A' reads >= +0.0006).
Prints member / 4-pool levels, and the admixture (1 - w) z(base pool) + w z(pop1 pool) minus base pool for base in {W8, W16} and pop1 pool in {wide x4, d128 x4}, val | holdout | holdout @last. CPU, read-only.
  cd <repo> && CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 .venv/bin/python nn_x/w12/kd/pop1w_read.py"""
import os, sys
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
sys.path.insert(0, '<repo>/nn_x/w12/recheck/pool_composition')
from pc_common import *
m = Meta(); ev = lambda s: ndcg_fast(s, m.yv, m.ov, m.Mv); eh = lambda s: ndcg_fast(s, m.yh, m.oh, m.Mh)
def ld(v, suf=''):
    pv, ph, pt = paths(v, suf); return m.z('val', np.load(pv)), m.z('hold', np.load(ph)), m.z('test', np.load(pt))
wide = lambda seeds: [f'w5_e10_{a}_s{s}' for a in ('kd05w192', 'kd05t25w192') for s in seeds]
G = {'W8': wide(range(951, 955)), 'W16': wide(range(951, 959)), 'pop1 wide x4': [f'abl_e10_pop1kd05w192_s{s}' for s in range(951, 955)], 'pop1 d128 x4': [f'abl_e10_pop1kd05t25_s{s}' for s in range(951, 955)]}
pool = lambda A, i: np.mean([a[i] for a in A], 0)
for suf in ('', '@last'):
    L = {k: [ld(v, suf) for v in vs] for k, vs in G.items()}
    for k in ('pop1 wide x4', 'pop1 d128 x4'):
        mv = np.mean([ev(x[0]).mean() for x in L[k]]); mh = np.mean([eh(x[1]).mean() for x in L[k]])
        print(f'{suf or "selected":8s} {k:13s}: member {mv:.5f} / {mh:.5f} | pool {ev(pool(L[k], 0)).mean():.5f} / {eh(pool(L[k], 1)).mean():.5f} | within-pool corr (val) {np.mean([np.corrcoef(L[k][i][0], L[k][j][0])[0, 1] for i in range(4) for j in range(i + 1, 4)]):.4f}')
    dw = ev(pool(L['pop1 wide x4'], 0)) - ev(pool(L['pop1 d128 x4'], 0)); dh = eh(pool(L['pop1 wide x4'], 1)) - eh(pool(L['pop1 d128 x4'], 1))
    print(f'{suf or "selected":8s} wide - d128 pop1 pool: val {fmt(dw)} | holdout {fmt(dh)}   (plain students: width gave +0.0002..+0.0005 val / +0.0006..+0.0008 holdout at pool level)')
    for b in ('W8', 'W16'):
        bv, bh = pool(L[b], 0), pool(L[b], 1); e0v, e0h = ev(bv), eh(bh)
        for k in ('pop1 wide x4', 'pop1 d128 x4'):
            for w in (0.1, 0.2, 0.3):
                dv = ev((1 - w) * m.z('val', bv) + w * m.z('val', pool(L[k], 0))) - e0v; dh = eh((1 - w) * m.z('hold', bh) + w * m.z('hold', pool(L[k], 1))) - e0h
                print(f'{suf or "selected":8s} {1 - w:.1f} z({b}) + {w:.1f} z({k}): val {fmt(dv)} (M>=50 {fmt(dv[m.Mv >= 50])}, M<50 {fmt(dv[m.Mv < 50])}) | holdout {fmt(dh)}')
    if suf == '':
        c = np.corrcoef(0.8 * m.z('test', pool(L['W16'], 2)) + 0.2 * m.z('test', pool(L['pop1 wide x4'], 2)), pool(L['W16'], 2))[0, 1]; print(f'test corr of 0.8 z(W16) + 0.2 z(pop1 wide) to W16 TO pool: {c:.4f}')
