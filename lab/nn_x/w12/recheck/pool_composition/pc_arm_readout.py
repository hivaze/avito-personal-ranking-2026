"""Pool-level read-out of a new student arm (CPU, read-only; reads prediction files directly, no cache needed).
  cd <repo> && CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 .venv/bin/python nn_x/w12/recheck/pool_composition/pc_arm_readout.py <arm> <seeds csv> [ref arm = kd05w192]
Prints, paired per feed vs the reference arm on the SAME seeds (val | holdout | holdout @last):
  member delta, n-pool delta, and the ASYMPTOTE delta  A_n = (n * NDCG(n-pool) - mean member NDCG) / (n - 1)
  (exact estimate of the infinite-seed pool value when NDCG(n) = a - b/n, which holds to 1e-4 on the 255 subsets of W8);
  then what the arm adds to W8:  [W8 + arm] vs W8, and vs [W8 + the same number of extra plain wide seeds] when those exist;
  correlation: within arm, arm-vs-reference same seed / different seed, arm pool vs W8 pool (val, test).
Reading rule proposed by the recheck: an arm is a CEILING SHIFT only if the asymptote delta is >= +2 SE on the holdout and >= 0 on val;
a member gain with a higher within-arm correlation is a move along the frontier (kd08 vs kd05: member +0.0002 val / +0.0007 hold, asymptote -0.0011 / +0.0001)."""
import os, sys
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pc_common import *
arm = sys.argv[1]; seeds = [int(s) for s in sys.argv[2].split(',')]; refarm = sys.argv[3] if len(sys.argv) > 3 else 'kd05w192'
m = Meta()
ev = lambda s: ndcg_fast(s, m.yv, m.ov, m.Mv); eh = lambda s: ndcg_fast(s, m.yh, m.oh, m.Mh)
def ld(v, suf=''):
    pv, ph, pt = paths(v, suf)
    return m.z('val', np.load(pv)), m.z('hold', np.load(ph)), (m.z('test', np.load(pt)) if pt.exists() else None)
name = lambda a, s: f'w5_e10_{a}_s{s}'
A = [ld(name(arm, s)) for s in seeds]; R = [ld(name(refarm, s)) for s in seeds]
AL = [ld(name(arm, s), '@last') for s in seeds]; RL = [ld(name(refarm, s), '@last') for s in seeds]
W8n = [name(a, s) for a in ('kd05w192', 'kd05t25w192') for s in (951, 952, 953, 954)]; W8 = [ld(v) for v in W8n]; W8L = [ld(v, '@last') for v in W8n]
n = len(seeds); pool = lambda X, i: np.mean([x[i] for x in X], 0)
def stats(X, i, e):
    mem = np.mean([e(x[i]) for x in X], 0); p = e(pool(X, i)); a = (n * p - mem) / (n - 1) if n > 1 else p
    return mem, p, a
print(f'== {arm} seeds {seeds} vs {refarm} (same seeds)')
for tag, i, e, X, Y in (('val', 0, ev, A, R), ('holdout', 1, eh, A, R), ('holdout @last', 1, eh, AL, RL)):
    ma, pa, aa = stats(X, i, e); mr, pr, ar = stats(Y, i, e)
    print(f'  {tag:14s}: member {ma.mean():.5f} ({fmt(ma-mr)}) | {n}-pool {pa.mean():.5f} ({fmt(pa-pr)}) | asymptote A_{n} {aa.mean():.5f} ({fmt(aa-ar)})')
c = lambda a, b: float(np.corrcoef(a, b)[0, 1])
if n > 1:
    win = np.mean([c(A[i][0], A[j][0]) for i in range(n) for j in range(i + 1, n)]); wref = np.mean([c(R[i][0], R[j][0]) for i in range(n) for j in range(i + 1, n)])
    same = np.mean([c(A[i][0], R[i][0]) for i in range(n)]); diff = np.mean([c(A[i][0], R[j][0]) for i in range(n) for j in range(n) if i != j])
    print(f'  corr (val): within arm {win:.4f} | within {refarm} {wref:.4f} | arm-ref same seed {same:.4f} | arm-ref different seed {diff:.4f}  (a diversity source beyond the seed needs different-seed corr < within-ref corr)')
for tag, i, e, X, Y in (('val', 0, ev, A, W8), ('holdout', 1, eh, A, W8), ('holdout @last', 1, eh, AL, W8L)):
    w = e(pool(Y, i)); both = e(np.mean([x[i] for x in Y] + [x[i] for x in X], 0)); d = both - w; s = ''
    if tag == 'val':
        big = m.Mv >= 50; s = f' (M>=50 {d[big].mean():+.5f}+-{se(d[big]):.5f}, M<50 {d[~big].mean():+.5f}+-{se(d[~big]):.5f})'
    print(f'  [W8 + arm x{n}] vs W8, {tag:14s}: {fmt(d)}{s} | arm pool vs W8: {fmt(e(pool(X, i)) - w)} | corr(arm pool, W8 pool) {c(pool(X, i), pool(Y, i)):.4f}')
if all(x[2] is not None for x in A): print(f'  test: corr(arm pool, W8 pool) {c(pool(A, 2), pool(W8, 2)):.4f}; corr([W8 + arm] pool, W8 pool) {c(np.mean([x[2] for x in W8] + [x[2] for x in A], 0), pool(W8, 2)):.4f}')
print('  reference numbers of this recheck: W8 TO pool val 0.40157 / hold 0.32456; wide-family asymptote val 0.4015-0.4019 / hold 0.3246-0.3248; law: adding k seeds of the same family to W8 is worth b (1/8 - 1/(8+k)), b = 0.0025 val / 0.0021 hold')
