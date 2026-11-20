"""P1b: pool-level local data slope with paired SE: 3-member pool of fold-K-excluded models (80% of users, seeds 951-953)
vs the 3-member pool of all-user base models on the same seeds. Also 15-member user-bagged pool vs 4 base + law."""
from lb_common import *
base = {s: ld(f'w5_e10_base_s{s}') for s in S4}
xf = {(k, s): ld(f'w5_e10_xf{k}_s{s}') for k in range(5) for s in (951, 952, 953)}
b3 = pool([base[s] for s in (951, 952, 953)]); eb = (ev(b3[0]), eh(b3[1]))
dv = []; dh = []
for k in range(5):
    p = pool([xf[k, s] for s in (951, 952, 953)]); dv.append(ev(p[0]) - eb[0]); dh.append(eh(p[1]) - eb[1])
    print(f' fold {k} excluded: 3-pool val {ev(p[0]).mean():.5f} hold {eh(p[1]).mean():.5f} | vs base 3-pool val {f(dv[-1])} hold {f(dh[-1])}')
dv = np.mean(dv, 0); dh = np.mean(dh, 0)
print(f'POOL-LEVEL slope: mean over 5 folds of [3-pool(80% users) - 3-pool(all users)]: val {f(dv)} | holdout {f(dh)}')
d = np.log2(1 / 0.8); print(f' per data doubling (pool of 3): val {-dv.mean()/d:+.5f} holdout {-dh.mean()/d:+.5f}; member-level (P1): val +0.00744 holdout +0.00649')
x15 = pool(list(xf.values())); b4 = pool(list(base.values()))
print(f'15 user-bagged xf vs 4 base: val {f(ev(x15[0]) - ev(b4[0]))} hold {f(eh(x15[1]) - eh(b4[1]))}')
# by M bucket (val) for the member-level slope
dm = np.mean([ev(xf[k, s][0]) - ev(base[s][0]) for k in range(5) for s in (951, 952, 953)], 0)
for lo, hi in ((0, 20), (20, 50), (50, 120), (120, 400)):
    mk = (Mv >= lo) & (Mv < hi); print(f'   member slope val M in [{lo},{hi}): {f(dm[mk])} (n {mk.sum()})')
