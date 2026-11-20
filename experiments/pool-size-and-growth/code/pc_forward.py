# Greedy forward selection of member groups with replacement, fitted on one labelled set and read
# on the other (holdout to val and val to holdout), for three candidate sets. It also repeats the
# selection on 10 random halves of the fitting set to see how stable the picks are.
# How it was run: on CPU from lab/nn_x/w12/recheck/pool_composition; the command line is not on
# record. Output: out/forward.log and out/forward.json.
# What it produced: selections fitted on val read -0.00095 +- 0.00028 on the holdout in the widest
# candidate set; selections fitted on the holdout read -0.0005 to 0.0000 on val
# (results/pool_composition_forward.log).
# Kept as it ran: paths and names refer to the lab tree (see lab/ and docs/glossary.md).
# Original: lab/nn_x/w12/recheck/pool_composition/pc_forward.py
"""(1) Group-level forward selection (Caruana greedy with replacement over GROUP pools), fitted on one witness and read on the other.
 Candidate = a group's member-equal z pool; the ensemble = weighted mean of the selected group pools (weight = times selected).
 Three candidate sets: STUD (tf students, d128 + d192), STUD+HARD (adds base, F7x, F7x70, F10, pop1, pop1kd), WIDE (d192 only).
 Also: stability of the selection over 10 random halves of the fitting set, and member-count ("member-equal") vs group-equal vs fitted weights."""
from pc_common import *
m = Meta(); st = Store(m); G = groups(); rng = np.random.default_rng(1114)
Y = dict(val=(m.yv, m.ov, m.Mv), hold=(m.yh, m.oh, m.Mh))
ev = lambda sp, s: ndcg_fast(s, *Y[sp])
G2 = dict(G); G2['kw4'] = G['kw_drop0'] + G['kw_e20'] + G['kw_hl2'] + G['kw_ff4']; G2['newwide'] = G['w192new'] + G['t25w192new']
STUD = ['kd05', 'kd08', 'kd10', 'kd03', 'kd05t25', 'kd08t25', 'w192', 't25w192', 'newwide', 'g2k05', 'g2k03', 'kx05', 'kx08', 'kw4', 't25w192b3', 't25w256']
HARD = ['base', 'F7x', 'F7x70', 'F10', 'pop1', 'pop1kd']
WIDE = ['w192', 't25w192', 'newwide', 'g2k05', 'g2k03', 'kx05', 'kx08', 'kw4', 't25w192b3']
allg = STUD + HARD
GP = {sp: {g: np.asarray(st.pool(sp, G2[g]), np.float32) for g in allg} for sp in ('val', 'hold')}
W8 = G['w192'] + G['t25w192']; e8 = {sp: ev(sp, st.pool(sp, W8)) for sp in ('val', 'hold')}
other = dict(val='hold', hold='val')


def greedy(sp, cands, steps, feeds=None):
    """returns list of chosen groups (with replacement) maximising mean NDCG on split sp (optionally on a subset of feeds)."""
    acc = np.zeros(len(Y[sp][0]), np.float64); chosen = []; path = []
    for t in range(steps):
        best = None
        for g in cands:
            e = ev(sp, ((acc + GP[sp][g]) / (t + 1)).astype(np.float32)); s = e.mean() if feeds is None else e[feeds].mean()
            if best is None or s > best[0]: best = (s, g)
        chosen.append(best[1]); acc += GP[sp][best[1]]; path.append(best[0])
    return chosen, path


def weights_pool(sp, w):
    acc = np.zeros(len(Y[sp][0]), np.float64)
    for g, k in w.items(): acc += k * GP[sp][g]
    return (acc / sum(w.values())).astype(np.float32)


res = {}
STEPS = 24
for setname, cands in (('WIDE', WIDE), ('STUD', STUD), ('STUD+HARD', STUD + HARD)):
    for fit in ('hold', 'val'):
        ch, path = greedy(fit, cands, STEPS); oth = other[fit]
        print(f'\n== {setname}: greedy fitted on {fit.upper()}, read on {oth.upper()} (delta vs W8, paired)')
        rows = []
        for t in (1, 2, 3, 4, 6, 8, 12, 16, 24):
            w = {}
            for g in ch[:t]: w[g] = w.get(g, 0) + 1
            df = ev(fit, weights_pool(fit, w)) - e8[fit]; do = ev(oth, weights_pool(oth, w)) - e8[oth]
            rows.append(dict(t=t, w=w, fit_d=float(df.mean()), fit_se=se(df), oth_d=float(do.mean()), oth_se=se(do)))
            print(f'  step {t:2d}: in-sample {fit} {fmt(df)} | other witness {oth} {fmt(do)} | ' + ' '.join(f'{g}:{k}' for g, k in sorted(w.items(), key=lambda x: -x[1])))
        res[f'{setname}|fit={fit}'] = rows
# stability: 10 random halves of the fitting set, 12 steps, STUD+HARD and WIDE
for setname, cands in (('WIDE', WIDE), ('STUD+HARD', STUD + HARD)):
    for fit in ('hold', 'val'):
        n = len(Y[fit][2]); freq = {g: 0.0 for g in cands}; oos = []; oth = other[fit]; cross = []
        for b in range(10):
            half = rng.permutation(n) < n // 2; ch, _ = greedy(fit, cands, 12, feeds=half)
            w = {}
            for g in ch: w[g] = w.get(g, 0) + 1; freq[g] += 1 / 120
            d = (ev(fit, weights_pool(fit, w)) - e8[fit])[~half]; oos.append(d.mean()); cross.append((ev(oth, weights_pool(oth, w)) - e8[oth]).mean())
        top = sorted(freq.items(), key=lambda x: -x[1])
        print(f'\n== stability {setname}, fit on random halves of {fit.upper()} (12 steps x 10): mean weight share ' + ' '.join(f'{g}:{v:.2f}' for g, v in top if v > 0))
        print(f'   other half of {fit}: delta vs W8 {np.mean(oos):+.5f} (sd over splits {np.std(oos):.5f}); other witness {oth}: {np.mean(cross):+.5f} (sd {np.std(cross):.5f})')
        res[f'stab {setname}|fit={fit}'] = dict(share=dict(top), oos=float(np.mean(oos)), oos_sd=float(np.std(oos)), cross=float(np.mean(cross)), cross_sd=float(np.std(cross)))
# weighting schemes over a fixed set of groups: member-equal vs group-equal
print('\n== fixed sets: member-equal vs group-equal weights (delta vs W8: val | hold)')
for setname, gs in (('all d192 groups', WIDE), ('all tf students', STUD), ('students + hard', STUD + HARD)):
    for scheme in ('member-equal', 'group-equal'):
        w = {g: (len(G2[g]) if scheme == 'member-equal' else 1) for g in gs}
        dv = ev('val', weights_pool('val', w)) - e8['val']; dh = ev('hold', weights_pool('hold', w)) - e8['hold']
        print(f'  {setname:18s} {scheme:12s}: val {fmt(dv)} | hold {fmt(dh)}'); res[f'{setname}|{scheme}'] = dict(d_val=float(dv.mean()), d_val_se=se(dv), d_hold=float(dh.mean()), d_hold_se=se(dh))
json.dump(res, open(OUT/'forward.json', 'w'), indent=1)
