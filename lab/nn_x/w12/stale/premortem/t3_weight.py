"""premortem t3_weight: the popularity-light weight for private (rules R2 constant weight, R3 feed-dependent weight, nested on val user halves).
Reads out/pf.npz. Writes out/t3_weight.md."""
from pm_common import *

m = Meta(test=True, hold=True)
R = dict(np.load(OUT / 'pf.npz')); Wt = m.pmix(); wpriv, wpub = Wt['private'], Wt['public']
big = m.M['val'] >= 50; n = len(big)
out = []
def P_(s=''):
    print(s, flush=True); out.append(s)

WS = (0.0, 0.1, 0.2, 0.3, 0.4)
COMBOS = {'W8 + Pd (scored form)': ('W8', 'W8P{k}'), 'W8 + Px8': ('W8', 'W8P{k}x8'), 'W16 + Pd': ('W16', 'W16P{k}d'), 'W16 + Px8': ('W16', 'W16P{k}x8'), 'W16 + Pw (information)': ('W16', 'W16P{k}w')}
def arr(combo, view):
    base, pat = COMBOS[combo]
    return np.stack([R[f'{base}|{view}'] if w == 0 else R[f'{pat.format(k=int(round(w * 100)))}|{view}'] for w in WS])

# ---------------- R2: constant weight against 0.2
P_('## T3a. Constant weight w against the scored w = 0.2 (paired per feed; FINAL level unless stated). Rule R2')
P_('| base + pop pool | w | val FINAL | val M>=50 | val M<50 | pmix-public | pmix-private (z) | stale-val TO | SVF | holdout TO (reported) | R2 without lagged clause |')
P_('|---|---|---|---|---|---|---|---|---|---|---|')
for combo in COMBOS:
    F, SV, SVF, H = arr(combo, 'fin'), arr(combo, 'sv'), arr(combo, 'svf'), arr(combo, 'hold')
    for i, w in enumerate(WS):
        if w == 0.2: continue
        d = F[i] - F[2]; v = wms(d); pp = wms(d, wpriv); pu = wms(d, wpub); s = wms(SV[i] - SV[2]); sf = wms(SVF[i] - SVF[2]); h = wms(H[i] - H[2])
        ok = (pp[0] >= 1.5 * pp[1]) and (v[0] >= 0) and (s[0] >= 0)
        P_(f'| {combo} | {w:.1f} | {f5(v)} | {wms(d[big])[0]:+.5f} | {wms(d[~big])[0]:+.5f} | {f5(pu)} | {f5(pp)} ({pp[0] / pp[1]:+.2f}) | {f5(s)} | {f5(sf)} | {f5(h)} | {"PASS" if ok else "no"} |')
P_('Lagged holdout on record (W8 + Pd, train-only, private-like view, subset where the gate holds) [D]: S_lag(0.1) +0.00013, S_lag(0.2) +0.00024, S_lag(0.3) -0.00015, S_lag(0.4) -0.00053 '
   '(SE 0.00021-0.00045 each against W8): any w > 0.2 is contradicted there; w = 0.1 reads -0.00011 against 0.2.')

# ---------------- mechanism: level gap pop pool - base pool by feed group
P_(); P_('## T3b. Mechanism (information): popularity-light pool minus base pool by feed group')
P_('| pools | group | feeds | share of val / public / private feeds | val FINAL | val TO | stale-val TO | val FT branch |')
P_('|---|---|---|---|---|---|---|---|')
tM, tc, pub = m.M['test'], m.tcold, m.public
GR = {'M < 50': (~big, tM < 50), 'M >= 50': (big, tM >= 50), 'cold share < 0.30': (m.vcold < 0.30, tc < 0.30), 'cold share 0.30-0.45': ((m.vcold >= 0.30) & (m.vcold < 0.45), (tc >= 0.30) & (tc < 0.45)),
      'cold share >= 0.45': (m.vcold >= 0.45, tc >= 0.45)}
for a, b in (('Pd', 'W8'), ('Px8', 'W16'), ('Pw', 'W16')):
    for g, (mv, mt) in GR.items():
        P_(f'| {a} - {b} | {g} | {int(mv.sum())} | {mv.mean():.3f} / {mt[pub].mean():.3f} / {mt[~pub].mean():.3f} | ' + ' | '.join(f5(wms((R[f"{a}|{k}"] - R[f"{b}|{k}"])[mv])) for k in ('fin', 'to', 'sv', 'ft')) + ' |')
P_(f'Mean feed cold share: val {m.vcold.mean():.3f}, public {np.nanmean(tc[pub]):.3f}, private {np.nanmean(tc[~pub]):.3f}. Share of feeds with M < 50: val {(~big).mean():.3f}, public {(tM[pub] < 50).mean():.3f}, private {(tM[~pub] < 50).mean():.3f}.')

# ---------------- R3: feed-dependent weight, nested on user halves
P_(); P_('## T3c. Feed-dependent weight (rule R3): 3 properties x 25 weight pairs, picked on one user half, scored on the other (200 random user-half splits, both directions)')
PROPS = {'P_M (M >= 50)': big, 'P_c30 (cold share >= 0.30)': m.vcold >= 0.30, 'P_c45 (cold share >= 0.45)': m.vcold >= 0.45}
users = m.vuser; uu, uinv = np.unique(users, return_inverse=True)
rng = np.random.default_rng(20261115)
SUMM = []
for combo in ('W8 + Pd (scored form)', 'W16 + Px8', 'W8 + Px8', 'W16 + Pd', 'W16 + Pw (information)'):
    F, SV = arr(combo, 'fin'), arr(combo, 'sv')
    # in-sample best rule per property
    ins = []
    for pn, g in PROPS.items():
        lo = int(np.argmax(F[:, ~g].mean(1))); hi = int(np.argmax(F[:, g].mean(1)))
        pf = np.where(g, F[hi], F[lo]); d = pf - F[2]; ds = np.where(g, SV[hi], SV[lo]) - SV[2]
        ins.append((pn, WS[lo], WS[hi], wms(d), wms(d, wpriv), wms(ds)))
    oos, oos_sv, oos_pp, ses, picks = [], [], [], [], {}
    oos_c, oos_c_pp, oos_c_sv, cpicks = [], [], [], {}
    for it in range(200):
        hu = rng.random(len(uu)) < 0.5; A = hu[uinv]
        d = np.zeros(n); ds = np.zeros(n); dc = np.zeros(n); dcs = np.zeros(n)
        for tr, te in ((A, ~A), (~A, A)):
            best = None
            for pn, g in PROPS.items():
                lo = int(np.argmax(F[:, tr & ~g].mean(1))); hi = int(np.argmax(F[:, tr & g].mean(1)))
                val = (F[lo][tr & ~g].sum() + F[hi][tr & g].sum()) / tr.sum()
                if best is None or val > best[0]: best = (val, pn, lo, hi)
            _, pn, lo, hi = best; g = PROPS[pn]; picks[(pn, WS[lo], WS[hi])] = picks.get((pn, WS[lo], WS[hi]), 0) + 1
            d[te] = (np.where(g, F[hi], F[lo]) - F[2])[te]; ds[te] = (np.where(g, SV[hi], SV[lo]) - SV[2])[te]
            # constant-only nested pick (5 rules): the out-of-sample value of choosing a constant weight on val
            c = int(np.argmax(F[:, tr].mean(1))); cpicks[WS[c]] = cpicks.get(WS[c], 0) + 1
            dc[te] = (F[c] - F[2])[te]; dcs[te] = (SV[c] - SV[2])[te]
        oos.append(d.mean()); ses.append(se(d)); oos_sv.append(ds.mean()); oos_pp.append(np.average(d, weights=wpriv))
        oos_c.append(dc.mean()); oos_c_pp.append(np.average(dc, weights=wpriv)); oos_c_sv.append(dcs.mean())
    oos, oos_sv, oos_pp = map(np.array, (oos, oos_sv, oos_pp)); sem = float(np.median(ses))
    top = sorted(picks.items(), key=lambda kv: -kv[1])[:4]
    verdict = 'DEFENSIBLE' if (oos.mean() >= 2 * sem and oos_sv.mean() >= 0 and oos_pp.mean() >= 0) else 'not defensible -> constant weight'
    P_(f'### {combo}')
    P_('| in-sample best rule per property (selection-biased) | w (group false) | w (group true) | val FINAL vs 0.2 | pmix-private | stale-val TO |')
    P_('|---|---|---|---|---|---|')
    for pn, lo, hi, v, pp, s in ins: P_(f'| {pn} | {lo:.1f} | {hi:.1f} | {f5(v)} | {f5(pp)} | {f5(s)} |')
    P_()
    P_(f'- NESTED (75 rules): out-of-sample val FINAL vs constant 0.2 = **{oos.mean():+.5f}** (sd over splits {oos.std():.5f}; median per-split paired SE {sem:.5f}; z {oos.mean() / sem:+.2f}); '
       f'same picks under pmix-private weights {oos_pp.mean():+.5f}; on stale-val TO {oos_sv.mean():+.5f}. Most frequent picks: '
       + '; '.join(f'{k[0].split(" ")[0]} {k[1]:.1f}/{k[2]:.1f} ({v / 4:.0f}%)' for k, v in top) + f'. **R3: {verdict}.**')
    P_(f'- NESTED, constant weights only (5 rules): out-of-sample val FINAL vs 0.2 = {np.mean(oos_c):+.5f} (sd {np.std(oos_c):.5f}); pmix-private {np.mean(oos_c_pp):+.5f}; stale-val TO {np.mean(oos_c_sv):+.5f}; picks: '
       + ', '.join(f'{k:.1f}: {v / 4:.0f}%' for k, v in sorted(cpicks.items())) + '.')
    SUMM.append((combo, float(oos.mean()), float(oos_pp.mean()), float(oos_sv.mean()), float(oos.mean() / sem)))
    P_()
P_('### Summary over the five compositions (block 4)')
P_('| composition | nested val | nested pmix-private | nested stale-val TO | z |')
P_('|---|---|---|---|---|')
for c_, a_, b_, s_, z_ in SUMM: P_(f'| {c_} | {a_:+.5f} | {b_:+.5f} | {s_:+.5f} | {z_:+.2f} |')
mean_ = np.mean([x[1] for x in SUMM]); allpos = sum(1 for x in SUMM if x[1] >= 0 and x[2] >= 0 and x[3] >= 0)
P_(f'Mean nested val delta {mean_:+.5f}; compositions with all three columns >= 0: {allpos} of 5 -> ' + ('"replicated"' if (mean_ >= 0.0003 and allpos >= 4) else '"one composition only"') + '.')
open(OUT / 't3_weight.md', 'w').write('\n'.join(out) + '\n')
