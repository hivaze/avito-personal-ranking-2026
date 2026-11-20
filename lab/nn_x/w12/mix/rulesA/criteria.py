"""Declared candidate criteria c1-c5 (DECLARATION block 6) for every eligible family and the variants, primary composition C1."""
import pickle, sys; sys.path.insert(0, '<repo>/src')
from lib import *
import pmix
fams = families(); names = [x[0] for x in fams]; elig = np.array([x[5] for x in fams]); iM = names.index('C(M50)'); W = pmix.weights()
R = {c: pickle.load(open(HERE / f'res_{c}.pkl', 'rb')) for c in COMPS}; A = np.load(HERE / 'pl_PLA_FD16.npz'); an = A['nested']
pv = lambda obs, null: (1 + (null >= obs).sum()) / (1 + len(null))
pr = props('val'); gt = gates(pr); parts = partitions(gt); pt = props('test'); g_t = gates(pt); pub = pt['ts'] < BOUND; parts_t = partitions(g_t)
def cover(key):
    cell, nc = parts_t[key]; return min(min((cell[m] == k).mean() for k in range(nc)) for m in (pub, ~pub))
rows = []
for fi, fam in enumerate(fams):
    if not fam[5]: continue
    r = R['C1']; v = r['vec'][fi]; d = v - r['plain_vec']; m, s = se(d); pA = pv(m, an[:, fi]); dM = (v - r['vec'][iM]).mean(); pAM = pv(dM, an[:, fi] - an[:, iM]) if fi != iM else 1.0
    c1 = m >= 2 * s and pA <= 0.05; c2 = pAM <= 0.05 and (v - r['mrule_vec']).mean() > 0
    oth = [(R[c]['vec'][fi] - R[c]['vec'][iM]).mean() for c in ('C2', 'C3', 'C4')]; allpos = all((R[c]['vec'][fi] - R[c]['plain_vec']).mean() > 0 for c in COMPS)
    c3 = sum(x > 0 for x in oth) >= 2 and allpos
    c4 = se(v - r['mrule_vec'], W['private'])[0] >= 0 and se(d, W['public'])[0] >= 0
    cv = cover(fam[1]); c5 = cv >= 0.05
    rows.append((fam[0], fam[4], m, s, pA, dM, pAM, (v - r['mrule_vec']).mean(), oth, allpos, se(v - r['mrule_vec'], W['private'])[0], se(d, W['public'])[0], cv, c1, c2, c3, c4, c5))
print(f'eligible families {len(rows)} | pass c1: {sum(x[13] for x in rows)} | pass c2: {sum(x[14] for x in rows)} | pass c3: {sum(x[15] for x in rows)} | pass c4: {sum(x[16] for x in rows)} | pass c5: {sum(x[17] for x in rows)}'
      f' | c1 & c2: {sum(x[13] and x[14] for x in rows)} | c1 & c2 & c3: {sum(x[13] and x[14] and x[15] for x in rows)} | ALL: {sum(all(x[13:]) for x in rows)}')
print(f'families with nested > PLAIN on all 4 compositions: {sum(x[9] for x in rows)} of {len(rows)}; with nested > nested C(M50) on all 4: {sum(x[5] > 0 and all(o > 0 for o in x[8]) for x in rows)}; with nested > fixed MRULE on all 4: '
      + str(sum(all((R[c]["vec"][names.index(x[0])] - R[c]["mrule_vec"]).mean() > 0 for c in COMPS) for x in rows)))
print(f'{"family":16s} par | C1 nested-PLAIN (z)        p_A   | - nested C(M50)  p_A,M | - fixed MRULE | C2 / C3 / C4 nested - nested C(M50) | all4>PLAIN | pmix-priv vs MRULE | pmix-pub vs PLAIN | min test cell | c1 c2 c3 c4 c5')
for x in sorted(rows, key=lambda x: -x[2])[:int(sys.argv[1]) if len(sys.argv) > 1 else 25]:
    print(f'{x[0]:16s} {x[1]}   | {x[2]:+.5f}+-{x[3]:.5f} ({x[2] / x[3]:+.1f}) {x[4]:.3f} | {x[5]:+.5f}  {x[6]:.3f}       | {x[7]:+.5f}      | ' + ' / '.join(f'{o:+.5f}' for o in x[8]) + f'      | {str(x[9]):5s}      | {x[10]:+.5f}           | {x[11]:+.5f}          | {x[12]:.3f}         | ' + ' '.join('Y ' if b else 'n ' for b in x[13:]))
# the declared fallback: does the M-rule family "hold"?
r = R['C1']; d = r['vec'][iM] - r['plain_vec']; m, s = se(d); pA = pv(m, an[:, iM])
print(f'\nFALLBACK (block 6): C(M50) nested - PLAIN on C1 {m:+.5f}+-{s:.5f} (z {m / s:+.2f}), p_A {pA:.3f}; on C2 / C3 / C4: ' + ' / '.join(fmt(R[c]['vec'][iM] - R[c]['plain_vec']) for c in ('C2', 'C3', 'C4'))
      + f' -> "{"holds" if (m >= 2 * s or pA <= 0.05) and all((R[c]["vec"][iM] - R[c]["plain_vec"]).mean() > 0 for c in COMPS) else "is not shown (primary composition); positive on all 4" }"')
