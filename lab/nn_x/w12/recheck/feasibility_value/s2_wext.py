"""Out-of-sample check of the pool-size forecasts: W8 + every FINISHED extension seed (955-958) vs W8, train-only level.
Forecasts on file: kd_reprice W16 0.4014 +- 0.0002 val / 0.3245 +- 0.0001 hold (not above W8); lb_stats / pool_composition
W16 - W8 = +0.00015 val / +0.00013 hold. Also: the extension seeds as a pool of their own vs W8 subsets of the same size.
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 .venv/bin/python nn_x/w12/recheck/feasibility_value/s2_wext.py"""
import itertools
from fv_common import *

S0 = (951, 952, 953, 954); S1 = (955, 956, 957, 958)
w8n = [(a, s) for a in ('kd05w192', 'kd05t25w192') for s in S0]
extn = [(a, s) for a in ('kd05w192', 'kd05t25w192') for s in S1 if done(a, s)]
print('finished extension members:', [f'{a}_s{s}' for a, s in extn])
W8 = [ld(f'w5_e10_{a}_s{s}') for a, s in w8n]; EX = [ld(f'w5_e10_{a}_s{s}') for a, s in extn]
W8L = [ld(f'w5_e10_{a}_s{s}', '@last') for a, s in w8n]; EXL = [ld(f'w5_e10_{a}_s{s}', '@last') for a, s in extn]
for tag, i, e, A, B in (('val', 0, ev, W8, EX), ('holdout', 1, eh, W8, EX), ('holdout @last', 1, eh, W8L, EXL)):
    w = e(pool(A, i)); both = e(pool(A + B, i)); k = len(B)
    mem8 = np.mean([e(x[i]).mean() for x in A]); memx = np.mean([e(x[i]).mean() for x in B])
    print(f'{tag:14s}: W8 {w.mean():.5f} | W{8 + k} {both.mean():.5f} | W{8 + k} - W8 {fmt(both - w)} | mean member W8 {mem8:.5f}, extension {memx:.5f}')
    if k >= 2:  # the extension pool vs all W8 subsets of the same size (is W8 a lucky draw?)
        px = e(pool(B, i)).mean()
        subs = [e(pool([A[j] for j in c], i)).mean() for c in itertools.combinations(range(8), k)]
        print(f'{"":14s}  extension {k}-pool {px:.5f} vs W8 {k}-subsets mean {np.mean(subs):.5f} (sd {np.std(subs):.5f}, min {np.min(subs):.5f}, max {np.max(subs):.5f})')
# N-law forecast for the realised k
for nm, b in (('val', 0.0025), ('holdout', 0.0021)):
    k = len(EX); print(f'N-law forecast {nm}: W{8 + k} - W8 = b (1/8 - 1/{8 + k}) = {b * (1 / 8 - 1 / (8 + k)):+.5f}')
