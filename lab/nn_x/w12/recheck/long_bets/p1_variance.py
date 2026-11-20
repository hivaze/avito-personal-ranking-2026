"""P1: where is the remaining estimation error? Seed variance vs data-sampling variance, local data slope, user-bagged pools.
 xf<K>_s<seed>: hard-label base recipe trained WITHOUT user fold K (80% of users); base_s<seed>: all users.
 (a) local data slope: member(xf) - member(base) = effect of -20% users (0.32 doublings).
 (b) corr structure: same fold / different seed  vs  different fold / different seed  (the extra de-correlation = data-sampling share).
 (c) pools at equal count: 3 same-fold members vs 3 different-fold members; 15 xf vs 4 base + ...; 'user-bagged' pool.
 (d) pool-size curves for students (W8/kd05) to read the infinite-member limit under KD."""
from lb_common import *
import itertools
base = {s: ld(f'w5_e10_base_s{s}') for s in S4}
xf = {(k, s): ld(f'w5_e10_xf{k}_s{s}') for k in range(5) for s in (951, 952, 953)}
print('== (a) local data slope (xf = 80% of users) ==')
dv = np.mean([ev(xf[k, s][0]) - ev(base[s][0]) for k in range(5) for s in (951, 952, 953)], 0)
dh = np.mean([eh(xf[k, s][1]) - eh(base[s][1]) for k in range(5) for s in (951, 952, 953)], 0)
print(f' member xf - base (15 pairs, same seed): val {f(dv)} | holdout {f(dh)}')
mv = np.mean([ev(v[0]).mean() for v in xf.values()]); mh = np.mean([eh(v[1]).mean() for v in xf.values()])
bv = np.mean([ev(v[0]).mean() for v in base.values()]); bh = np.mean([eh(v[1]).mean() for v in base.values()])
print(f' mean member: xf val {mv:.5f} hold {mh:.5f} | base val {bv:.5f} hold {bh:.5f}')
d = np.log2(1 / 0.8)
print(f' implied local slope per data doubling: val {-(mv-bv)/d:+.5f}  holdout {-(mh-bh)/d:+.5f}  (0.32 doublings; global two-point slope on record +0.0037)')
print('== (b) correlation structure (val z / holdout z) ==')
def cc(pairs, i):
    return np.mean([wcorr(a[i], b[i]) for a, b in pairs])
same_fold = [(xf[k, s1], xf[k, s2]) for k in range(5) for s1, s2 in itertools.combinations((951, 952, 953), 2)]
diff_fold = [(xf[k1, s1], xf[k2, s2]) for k1, k2 in itertools.combinations(range(5), 2) for s1 in (951, 952, 953) for s2 in (951, 952, 953) if s1 != s2]
diff_fold_same_seed = [(xf[k1, s], xf[k2, s]) for k1, k2 in itertools.combinations(range(5), 2) for s in (951, 952, 953)]
bb = [(base[a], base[b]) for a, b in itertools.combinations(S4, 2)]
for nm, pr in (('same fold, diff seed', same_fold), ('diff fold, diff seed', diff_fold), ('diff fold, same seed', diff_fold_same_seed), ('base-base (all users)', bb)):
    print(f' {nm:24s} n={len(pr):3d}  val {cc(pr,0):.4f}  holdout {cc(pr,1):.4f}')
print(' note: two different folds share 60/80 = 75% of their users; extrapolate the de-correlation to disjoint data with factor 1/(1-0.75) = 4.')
print('== (c) pools at equal count ==')
def P(ms): return pool(ms)
r3s = [P([xf[k, s] for s in (951, 952, 953)]) for k in range(5)]  # 3 members, same fold
r3d = [P([xf[k, s] for k, s in zip(ks, (951, 952, 953))]) for ks in itertools.combinations(range(5), 3)]  # 3 members, 3 different folds
b3 = [P([base[s] for s in c]) for c in itertools.combinations(S4, 3)]
for nm, r in (('3 xf same fold', r3s), ('3 xf different folds', r3d), ('3 base (all users)', b3)):
    print(f' {nm:22s} val {np.mean([ev(x[0]).mean() for x in r]):.5f}  holdout {np.mean([eh(x[1]).mean() for x in r]):.5f}  (n pools {len(r)})')
x15 = P(list(xf.values())); b4 = P(list(base.values()))
x5 = [P([xf[k, s] for k in range(5)]) for s in (951, 952, 953)]
print(f' 15 xf (user-bagged) val {ev(x15[0]).mean():.5f} hold {eh(x15[1]).mean():.5f} | 5 xf (1 per fold) val {np.mean([ev(x[0]).mean() for x in x5]):.5f} hold {np.mean([eh(x[1]).mean() for x in x5]):.5f} | 4 base val {ev(b4[0]).mean():.5f} hold {eh(b4[1]).mean():.5f}')
# 4 xf from 4 different folds vs 4 base, paired
x4 = [P([xf[k, s] for k, s in zip(ks, (951, 952, 953, 951))]) for ks in itertools.combinations(range(5), 4)]
dv = np.mean([ev(x[0]) for x in x4], 0) - ev(b4[0]); dh = np.mean([eh(x[1]) for x in x4], 0) - eh(b4[1])
print(f' 4 xf (4 folds) - 4 base: val {f(dv)} | holdout {f(dh)}   <- user-bagging at equal count (each member has 20% less data)')
print('== (d) pool-size curves (mean over subsets) ==')
arms = {a: [ld(f'w5_e10_{a}_s{s}') for s in S4] for a in ('base', 'kd05', 'kd05w192', 'kd05t25w192')}
W8 = arms['kd05w192'] + arms['kd05t25w192']
def curve(ms, ks):
    out = []
    rng = np.random.default_rng(0)
    for k in ks:
        subs = list(itertools.combinations(range(len(ms)), k))
        if len(subs) > 12: subs = [subs[i] for i in rng.choice(len(subs), 12, replace=False)]
        out.append((k, np.mean([ev(P([ms[i] for i in c])[0]).mean() for c in subs]), np.mean([eh(P([ms[i] for i in c])[1]).mean() for c in subs])))
    return out
for nm, ms, ks in (('base', arms['base'], (1, 2, 4)), ('kd05', arms['kd05'], (1, 2, 4)), ('W8 (wide students)', W8, (1, 2, 4, 8))):
    c = curve(ms, ks); print(f' {nm:20s} ' + ' | '.join(f'k={k}: {a:.5f}/{b:.5f}' for k, a, b in c))
    # 1/k extrapolation from the last two points: n(k) = n_inf - c/k
    (k1, a1, b1), (k2, a2, b2) = c[-2], c[-1]
    print(f'   1/k extrapolation to k=inf: val {a2 + (a2 - a1) * (1 / k2) / (1 / k1 - 1 / k2):.5f}  holdout {b2 + (b2 - b1) * (1 / k2) / (1 / k1 - 1 / k2):.5f}')
iu = np.triu_indices(8, 1)
print(f' W8 member-member corr: val {np.corrcoef([m[0] for m in W8])[iu].mean():.4f} holdout {np.corrcoef([m[1] for m in W8])[iu].mean():.4f}')
np.save(ROOT / 'nn_x/w12/recheck/long_bets/out/w8_to_val_z.npy', P(W8)[0].astype(np.float32)); np.save(ROOT / 'nn_x/w12/recheck/long_bets/out/w8_to_hold_z.npy', P(W8)[1].astype(np.float32))
