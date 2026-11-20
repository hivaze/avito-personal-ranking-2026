"""CHECK 2 (numeric part) + CHECK 5 real statistics: nested re-run with the reviewer's folds (seeds 7001..7010), cross-member-set nested, halves, interaction, random-gate randomisation.
Saves the real statistics to real_stats.npz / real_vecs.npz for the placebo read."""
import sys, time, json
sys.path.insert(0, '<repo>/nn_x/w12/mix/review')
from rlib import *
t0 = time.time()
mv = meta('val'); g = groups(); M = mv['M']; H = mv['H']; big = M >= 50; n = mv['n']
SEEDS = list(range(7001, 7011)); FL = [folds(g, s) for s in SEEDS]
for fo in FL:      # fold integrity: no group in two folds (by construction) - verified explicitly on user ids, user 0 excluded
    u = mv['uid']; nz = u != 0; mx = np.zeros(u.max() + 1, np.int64) - 1; mx[u[nz]] = fo[nz]; assert (mx[u[nz]] == fo[nz]).all()
print(f'folds: 10 seeds x 5, user-grouped (verified: every non-zero user id sits in exactly one fold); user-0 feeds {int((mv["uid"] == 0).sum())} spread over folds {np.bincount(FL[0][mv["uid"] == 0])}')
hq = np.quantile(H, [.2, .35, .5, .65, .8]); print('H cuts for HSTUMP (all-val quantiles, label-free):', hq)
pr = dict(M=M, H=H); fams = fam_defs(pr, hq); fd = dict(fams)
D = {k: tensor(k) - tensor(k)[:, [PLAIN]] for k in COMPS}
DM = {k: np.where(big, D[k][:, C04], 0.0) for k in COMPS}     # fixed M-rule minus plain, per feed
import importlib
sys.path.insert(0, '<repo>/src'); pmix = importlib.import_module('pmix'); W = pmix.weights()


def nested_cross(Dsrc, Dtgt, subs, fold_list, K=5):
    """rule fitted on the training users of Dsrc, read on the held-out users of Dtgt."""
    vec = np.zeros(n)
    for fold in fold_list:
        S = [block_sums(Dsrc, fold, cell, nc, K) for (cell, nc, R) in subs]
        for k in range(K):
            bk = (-np.inf, None, None)
            for si, (cell, nc, R) in enumerate(subs):
                sc, r = best_rule(S[si].sum(0) - S[si][k], R)
                if sc > bk[0] + 1e-12: bk = (sc, si, r)
            cell = subs[bk[1]][0]; mk = fold == k; vec[mk] += Dtgt[mk][np.arange(mk.sum()), bk[2][cell[mk]]] / len(fold_list)
    return vec


RES = {}; VEC = {}
print('\n=== NESTED (10 seeds x 5 folds, reviewer folds); all minus PLAIN; +- = user-clustered SE [per-feed SE]; range = min..max over the 10 fold seeds ===')
for key in ('C1', 'C2', 'C3', 'C4'):
    r = nested(D[key], fams, FL, want_pick=True); RES[key] = r
    print(f'--- {key} {COMPS[key]}  fixed M-rule minus plain {fm(DM[key], g)}')
    for nm in r:
        v = r[nm]['vec']; VEC[(key, nm)] = v; s = r[nm]['seed']
        print(f'  {nm:18s} in-sample {r[nm]["ins"]:+.5f} | nested {fm(v, g)} [{pse(v)[1]:.5f}] (z {cse(v, g)[0] / cse(v, g)[1]:+.2f}) range {min(s):+.5f}..{max(s):+.5f} | minus fixed M-rule {fm(v - DM[key], g)}'
              f' | minus nested C(M50) {fm(v - r["C(M50)"]["vec"], g)} | pmix pub {wse(v, W["public"])[0]:+.5f} priv {wse(v, W["private"])[0]:+.5f}')
    for nm in ('C(M50)', 'MSTEP50', 'MSTEP(25,50,100)', 'MSTEPc(20..150)', 'HSTUMP', 'F(Hq)'):
        from collections import Counter
        c = Counter(r[nm]['picks']); top = c.most_common(4)
        print(f'    picks {nm}: ' + '; '.join(f'sub#{si} {[dec(x) for x in rr]} x{cnt}' for (si, rr), cnt in top) + f' (of {len(r[nm]["picks"])})')
print('\n=== T2 = mean of C1 and C3 nested vectors (A / B post hoc statistic), and mean over C1..C4 ===')
for nm in RES['C1']:
    v = 0.5 * (VEC[('C1', nm)] + VEC[('C3', nm)]); v4 = sum(VEC[(k, nm)] for k in ('C1', 'C2', 'C3', 'C4')) / 4
    print(f'  {nm:18s} T2 {fm(v, g)} | mean C1..C4 {fm(v4, g)}')
v = 0.5 * (DM['C1'] + DM['C3']); v4 = sum(DM[k] for k in ('C1', 'C2', 'C3', 'C4')) / 4
print(f'  fixed M-rule       T2 {fm(v, g)} | mean C1..C4 {fm(v4, g)}')

print('\n=== CHECK 5: M-rule estimates per composition ===')
TW = {'C1': 'C3', 'C3': 'C1', 'C2': 'C4', 'C4': 'C2'}; HEAD = {}
for key in ('C1', 'C2', 'C3', 'C4'):
    h1 = VEC[(key, 'MSTEP50')]; h1c = VEC[(key, 'MSTEPc(20..150)')]; h3 = nested_cross(D[TW[key]], D[key], fd['MSTEP50'], FL); h3a = nested_cross(D[TW[key]], D[key], fd['C(M50)'], FL)
    hd = 0.5 * (h1 + h3); HEAD[key] = hd; VEC[(key, 'H3')] = h3; VEC[(key, 'HEAD')] = hd
    sdp = DM[key].std() / np.sqrt(8562)
    print(f'  {key}: H2 fixed in-sample {fm(DM[key], g)} | H1 nested refit MSTEP50 {fm(h1, g)} (A-grid C(M50) {fm(VEC[(key, "C(M50)")], g)}) | H1c cut free {fm(h1c, g)} | H3 cross-set nested {fm(h3, g)} (A-grid {fm(h3a, g)})'
          f' | HEADLINE (H1+H3)/2 {fm(hd, g)} z {cse(hd, g)[0] / cse(hd, g)[1]:+.2f} | pmix pub {wse(hd, W["public"])[0]:+.5f}+-{wse(hd, W["public"])[1]:.5f} priv {wse(hd, W["private"])[0]:+.5f}+-{wse(hd, W["private"])[1]:.5f}'
          f' | fixed rule pmix pub {wse(DM[key], W["public"])[0]:+.5f}+-{wse(DM[key], W["public"])[1]:.5f} priv {wse(DM[key], W["private"])[0]:+.5f}+-{wse(DM[key], W["private"])[1]:.5f} | paired public SE {sdp:.5f}')
print('  H4 member-disjoint halves (fixed rule minus plain): ' + ' | '.join(f'{k} {fm(DM[k], g)}' for k in ('HA', 'HB', 'WA', 'WB')))
cc = lambda a, b: np.corrcoef(a[big], b[big])[0, 1]
print(f'  per-feed correlation of the fixed-rule contrast on long feeds: HA~HB {cc(DM["HA"], DM["HB"]):.3f} | WA~WB {cc(DM["WA"], DM["WB"]):.3f} | HA~WA (twins) {cc(DM["HA"], DM["WA"]):.3f} | HB~WB {cc(DM["HB"], DM["WB"]):.3f} | HA~WB {cc(DM["HA"], DM["WB"]):.3f} | C1~C3 {cc(DM["C1"], DM["C3"]):.3f} | C1~C2 {cc(DM["C1"], DM["C2"]):.3f} | C2~C4 {cc(DM["C2"], DM["C4"]):.3f}')
m4 = (DM['HA'] + DM['HB'] + DM['WA'] + DM['WB']) / 4; mfd = (DM['HA'] + DM['HB']) / 2
print(f'  mean of the 4 halves {fm(m4, g)} | mean of the 2 full-data halves {fm(mfd, g)} | HB minus HA {fm(DM["HB"] - DM["HA"], g)}')

print('\n=== SPECIFICITY of M as the gate: interaction I = mean D on M>=50 minus mean D on M<50 (D = E(.4,.5) - E(.2,.5)); exact randomisation of the gate (2000 random sets of the size of {M>=50}) ===')
rng = np.random.default_rng(424242); nb_ = int(big.sum())
for key in ('C1', 'C2', 'C3', 'C4', 'HA', 'HB', 'WA', 'WB'):
    d = D[key][:, C04]; mL, sL = cse(d[big], g[big]); mS, sS = cse(d[~big], g[~big]); I = mL - mS; sI = np.hypot(sL, sS)
    obs = DM[key].mean(); null = np.array([d[rng.permutation(n)[:nb_]].sum() / n for _ in range(2000)])
    c4 = D[key][:, C04]; print(f'  {key}: D all {fm(d, g)} | M>=50 {mL:+.5f}+-{sL:.5f} | M<50 {mS:+.5f}+-{sS:.5f} | I {I:+.5f}+-{sI:.5f} (z {I / sI:+.2f}) | fixed M-rule {obs:+.5f} vs random gate of equal size: mean {null.mean():+.5f} sd {null.std():.5f}'
                           f' q95 {np.quantile(null, .95):+.5f} p {(1 + (null >= obs).sum()) / 2001:.3f} | M-rule minus constant 0.4 {fm(DM[key] - c4, g)}')
dp = 0.5 * (D['C1'][:, C04] + D['C2'][:, C04]); mL, sL = cse(dp[big], g[big]); mS, sS = cse(dp[~big], g[~big])
print(f'  full-data mean (C1, C2): I {mL - mS:+.5f}+-{np.hypot(sL, sS):.5f} | M<50 {mS:+.5f}+-{sS:.5f}')
dp = 0.5 * (D['C3'][:, C04] + D['C4'][:, C04]); mL, sL = cse(dp[big], g[big]); mS, sS = cse(dp[~big], g[~big])
print(f'  twins mean (C3, C4):     I {mL - mS:+.5f}+-{np.hypot(sL, sS):.5f} | M<50 {mS:+.5f}+-{sS:.5f}')
dd = D['C1'][:, C04] - D['C3'][:, C04]; print(f'  D on M<50, full-data minus twins: 16-base {fm(dd[~big], g[~big])} | 8-base {fm((D["C2"][:, C04] - D["C4"][:, C04])[~big], g[~big])} | halves HA-WA {fm((D["HA"][:, C04] - D["WA"][:, C04])[~big], g[~big])} HB-WB {fm((D["HB"][:, C04] - D["WB"][:, C04])[~big], g[~big])}')

np.savez(HERE / 'real_vecs.npz', **{f'{k}|{nm}': v for (k, nm), v in VEC.items()}, **{f'DM|{k}': DM[k] for k in DM})
json.dump({k: {nm: dict(ins=RES[k][nm]['ins'], nested=RES[k][nm]['nested'], seed=RES[k][nm]['seed']) for nm in RES[k]} for k in RES}, open(HERE / 'real_stats.json', 'w'), indent=1)
print(f'done {time.time() - t0:.0f}s')
