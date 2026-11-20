"""P9: do students of DIFFERENT teachers decorrelate (lineage diversity)? d192 students, 4 seeds each:
 kd05w192 (teacher = 11 hard members), kd05t25w192 (25 members incl. GRU + pop1), kd05g2w192 (teacher = 8 wide students), kx05 (cross-fitted fold teachers).
 Within-arm corr uses different seeds; cross-arm corr is reported for different seeds AND same seed (same init/data order)."""
from lb_common import *
import itertools
arms = ['kd05w192', 'kd05t25w192', 'kd05g2w192', 'kx05']
Z = {a: [ld(f'w5_e10_{a}_s{s}') for s in S4] for a in arms}
print('member-member corr (val / holdout); diagonal = within arm, different seeds')
for a in arms:
    row = []
    for b in arms:
        if a == b: pr = [(Z[a][i], Z[a][j]) for i, j in itertools.combinations(range(4), 2)]
        else: pr = [(Z[a][i], Z[b][j]) for i in range(4) for j in range(4) if i != j]
        row.append(f'{np.mean([wcorr(x[0], y[0]) for x, y in pr]):.4f}/{np.mean([wcorr(x[1], y[1]) for x, y in pr]):.4f}')
    print(f' {a:13s} ' + '  '.join(row))
print('pool-pool corr and equal-count mixed pools (4 = 2 + 2) vs pure 4-pools:')
P4 = {a: pool(Z[a]) for a in arms}
for a, b in itertools.combinations(arms, 2):
    mix = pool(Z[a][:2] + Z[b][2:]); pa, pb = P4[a], P4[b]
    ref_v = 0.5 * (ev(pa[0]) + ev(pb[0])); ref_h = 0.5 * (eh(pa[1]) + eh(pb[1]))
    print(f' {a} x {b}: pool corr val {wcorr(pa[0], pb[0]):.4f} hold {wcorr(pa[1], pb[1]):.4f} | mixed 2+2 minus mean of the two pure 4-pools: val {f(ev(mix[0]) - ref_v)} hold {f(eh(mix[1]) - ref_h)}')
