"""premortem t2_forecast: private forecast (rule R1), nested pick of the short list, base-draw twins, staleness gradient, label-free day behaviour.
Reads out/pf.npz, cache/pool_*, cache/{valfin,testfin}_*.npy. Writes out/t2_forecast.md, out/t2_forecast.json.
Usage: t2_forecast.py [extra per-feed npz with keys '<name>|fin', '<name>|svf', ...]"""
import sys, json
from math import erf, sqrt
from pm_common import *
from common import FEATS

m = Meta(test=True, hold=False)
R = dict(np.load(OUT / 'pf.npz'))
if len(sys.argv) > 1: R.update(dict(np.load(sys.argv[1])))
Wt = m.pmix(); wpub, wpriv = Wt['public'], Wt['private']
big = m.M['val'] >= 50; n = len(big)
PUB = {'W8P20': 0.3950094, 'W8': 0.394486, 'W16': 0.39448337, 'S16': 0.3940782, 'W8TO': 0.394009, 'C3': 0.393454, 'G3': 0.392820}
Phi = lambda x: 0.5 * (1 + erf(x / sqrt(2)))
out = []
def P_(s=''):
    print(s, flush=True); out.append(s)

def wsd(d, w):
    mu = np.average(d, weights=w); return float(np.sqrt(np.average((d - mu) ** 2, weights=w)))

def forecast(a, b):
    d = R[f'{a}|fin'] - R[f'{b}|fin']; ds = R[f'{a}|svf'] - R[f'{b}|svf']
    v1, v2, v3 = wms(d), wms(d, wpriv), wms(ds); vp = wms(d, wpub)
    c = (v1[0] + v2[0] + v3[0]) / 3; s = max(v1[1], v2[1], v3[1]); md = (max(v1[0], v2[0], v3[0]) - min(v1[0], v2[0], v3[0])) / 2
    sd_val = sqrt(s * s + md * md); se_priv = wsd(d, wpriv) / sqrt(20033)
    r = dict(v1=v1, v2=v2, v3=v3, vp=vp, c=c, s=s, m=md, sd_val=sd_val, se_priv=se_priv, public=None)
    if a in PUB and b in PUB:
        pubd = PUB[a] - PUB[b]; se_pub = float(d.std(ddof=1)) / sqrt(8562); shift = v2[0] - vp[0]
        ps = pubd + shift; sp = sqrt(se_pub ** 2 + shift ** 2)
        w1, w2 = 1 / sd_val ** 2, 1 / sp ** 2
        r.update(public=pubd, se_pub=se_pub, pub_shift=ps, pub_sd=sp, f=(w1 * c + w2 * ps) / (w1 + w2), sd_f=sqrt(1 / (w1 + w2)))
    else:
        r.update(f=c, sd_f=sd_val)
    r['p_exp'] = Phi(r['f'] / r['sd_f']); r['p_real'] = Phi(r['f'] / sqrt(r['sd_f'] ** 2 + se_priv ** 2))
    r['lo'] = r['f'] - 1.645 * sqrt(r['sd_f'] ** 2 + se_priv ** 2); r['hi'] = r['f'] + 1.645 * sqrt(r['sd_f'] ** 2 + se_priv ** 2)
    return r

def ftable(title, pairs):
    P_(); P_(f'## {title}')
    P_('| A - B | V1 val FINAL | V2 pmix-private | V3 SVF | centre c_val ± sd (sampling, spread) | PUBLIC ± SE | public shifted to private ± sd | **FORECAST private delta ± sd** | 90% interval of the REALISED private delta | P(A > B) expected / realised |')
    P_('|---|---|---|---|---|---|---|---|---|---|')
    J = {}
    for a, b in pairs:
        r = forecast(a, b); J[f'{a}-{b}'] = r
        pb = f'{r["public"]:+.6f} ± {r["se_pub"]:.5f}' if r['public'] is not None else 'none'
        ps = f'{r["pub_shift"]:+.5f} ± {r["pub_sd"]:.5f}' if r['public'] is not None else ''
        P_(f'| {a} - {b} | {f5(r["v1"])} | {f5(r["v2"])} | {f5(r["v3"])} | {r["c"]:+.5f} ± {r["sd_val"]:.5f} ({r["s"]:.5f}, {r["m"]:.5f}) | {pb} | {ps} | **{r["f"]:+.5f} ± {r["sd_f"]:.5f}** | [{r["lo"]:+.5f}, {r["hi"]:+.5f}] | {r["p_exp"]:.2f} / {r["p_real"]:.2f} |')
    return J

J = {}
SHORT = ['W8P20', 'W16', 'W8', 'W16P20d', 'W16P20x8']
J['short_vs_W8P20'] = ftable('T2a. Private forecast, short list against W8P20 (rule R1)', [(x, 'W8P20') for x in SHORT if x != 'W8P20'])
J['short_vs_W16'] = ftable('T2b. Private forecast, short list against W16', [(x, 'W16') for x in SHORT if x != 'W16'])
extra = [k.split('|')[0] for k in R if k.endswith('|fin') and k.startswith('RULE')]
INFO = ['W8P20x8', 'W16P20w', 'W16P30x8', 'W16P40x8', 'W16P30d', 'W16P40d', 'W16P40w', 'W8P30x8', 'W8P40x8', 'W8P40', 'S16', 'C3'] + extra
J['info_vs_W8P20'] = ftable('T2c. The same statistic for compositions OUTSIDE the short list, against W8P20 (information; selection guard of R1 applies)', [(x, 'W8P20') for x in INFO])
J['info_vs_W16'] = ftable('T2d. ... and against W16', [(x, 'W16') for x in INFO if x not in ('C3',)])

# ---------------- nested pick (winner's curse)
P_(); P_('## T2e. Nested pick: choose the file with the largest c_val on one user half, score it against W8P20 on the other (200 splits, both directions)')
uu, uinv = np.unique(m.vuser, return_inverse=True); rng = np.random.default_rng(5)
def nested(cands, label):
    F = np.stack([R[f'{c}|fin'] for c in cands]); S = np.stack([R[f'{c}|svf'] for c in cands]); ref = cands.index('W8P20')
    o, op, osv, picks, ins = [], [], [], {}, []
    for it in range(200):
        hu = rng.random(len(uu)) < 0.5; A = hu[uinv]; d = np.zeros(n); ds = np.zeros(n)
        for tr, te in ((A, ~A), (~A, A)):
            c = (F[:, tr].mean(1) + (F[:, tr] * wpriv[tr]).sum(1) / wpriv[tr].sum() + S[:, tr].mean(1)) / 3
            i = int(np.argmax(c)); picks[cands[i]] = picks.get(cands[i], 0) + 1
            d[te] = (F[i] - F[ref])[te]; ds[te] = (S[i] - S[ref])[te]
        o.append(d.mean()); op.append(np.average(d, weights=wpriv)); osv.append(ds.mean())
    cfull = (F.mean(1) + (F * wpriv).sum(1) / wpriv.sum() + S.mean(1)) / 3; ib = int(np.argmax(cfull))
    P_(f'| {label} ({len(cands)} files) | {cands[ib]} ({cfull[ib] - cfull[ref]:+.5f} in c_val over W8P20) | {np.mean(o):+.5f} (sd {np.std(o):.5f}) | {np.mean(op):+.5f} (sd {np.std(op):.5f}) | {np.mean(osv):+.5f} (sd {np.std(osv):.5f}) | '
       + ', '.join(f'{k} {v / 4:.0f}%' for k, v in sorted(picks.items(), key=lambda kv: -kv[1])[:5]) + ' |')
P_('| list | in-sample best (selection-biased) | out-of-sample val FINAL vs W8P20 | out-of-sample pmix-private | out-of-sample SVF | picks |')
P_('|---|---|---|---|---|---|')
nested(SHORT, 'short list')
nested(SHORT + ['W8P20x8', 'W16P20w', 'W16P30x8', 'W16P40x8', 'W16P30d', 'W16P40d', 'W16P30w', 'W16P40w', 'W8P30x8', 'W8P40x8', 'W8P30', 'W8P40', 'W8P10', 'W16P10d', 'W16P10x8'], 'all weights and pools')
nested(['W8P20', 'W16P40x8'], 'W8P20 or W16P40x8')
if extra: nested(['W8P20'] + extra, 'W8P20 or the feed-size rule(s)')

# ---------------- base-draw twins
P_(); P_('## T2f. How much of a composition is the base-seed draw: the same admixtures on the W8b base (seeds 955-958) (information)')
L = lambda k, p: np.load(CACHE / f'pool_{k}_{p}.npy'); Zv = lambda x: m.z('val', x)
def comp(base, pop, w):
    b = {}
    for kd in ('val_ft', 'val_to', 'sv'):
        b[kd] = Zv(L(kd, base)) if pop is None else Zv((1 - w) * Zv(L(kd, base)) + w * Zv(L(kd, pop)))
    return m.ndcg('val', 0.5 * b['val_ft'] + 0.5 * b['val_to']), m.ndcg('val', 0.5 * b['val_ft'] + 0.5 * b['sv']), m.ndcg('val', b['sv'])
P_('| admixture - its base | base W8 (s951-954): val FINAL / pmix-private / SVF / stale TO | base W8b (s955-958) | base W16 |')
P_('|---|---|---|---|')
for pop, w in (('Pd', 0.2), ('Px8', 0.2), ('Pw', 0.2), ('Pd', 0.4), ('Px8', 0.4), ('Pw', 0.4)):
    cells = []
    for base in ('W8', 'W8b', 'W16'):
        f0, s0, t0 = comp(base, None, 0); f1, s1, t1 = comp(base, pop, w); d = f1 - f0
        cells.append(f'{f5(wms(d))} / {wms(d, wpriv)[0]:+.5f} / {wms(s1 - s0)[0]:+.5f} / {wms(t1 - t0)[0]:+.5f}')
    P_(f'| {w:.1f} {pop} | ' + ' | '.join(cells) + ' |')
fa, sa, _ = comp('W8', 'Pd', 0.2); fb, sb, _ = comp('W8b', 'Pd', 0.2); d = fa - fb
P_(f'W8P20 minus its W8b twin (0.8 z(W8b) + 0.2 z(Pd)): val FINAL {f5(wms(d))}, pmix-private {f5(wms(d, wpriv))}, SVF {f5(wms(sa - sb))}. (W8 - W8b: val FINAL {f5(wms(R["W8|fin"] - R["W8b|fin"]))}.)')

# ---------------- staleness gradient
P_(); P_('## T2g. The staleness gradient of the popularity-light contrast (0.8 z(base) + 0.2 z(pop) minus base)')
c = m.vcold
GR = [('all val', np.ones(n, bool)), ('val Tue 09-22 (2 d after the dense cutoff)', m.tue), ('val Thu 09-24 (4 d)', m.thu), ('cold share < 0.15', c < 0.15), ('0.15-0.30', (c >= 0.15) & (c < 0.30)),
      ('0.30-0.45', (c >= 0.30) & (c < 0.45)), ('>= 0.45', c >= 0.45), ('M < 50', ~big), ('M >= 50', big),
      ('M >= 50 and cold >= 0.30', big & (c >= 0.30)), ('M >= 50 and cold < 0.30', big & (c < 0.30)), ('M < 50 and cold >= 0.30', ~big & (c >= 0.30)), ('M < 50 and cold < 0.30', ~big & (c < 0.30))]
tc, tM, pub = m.tcold, m.M['test'], m.public
tg = {'all val': np.ones(len(tM), bool), 'cold share < 0.15': tc < 0.15, '0.15-0.30': (tc >= 0.15) & (tc < 0.30), '0.30-0.45': (tc >= 0.30) & (tc < 0.45), '>= 0.45': tc >= 0.45, 'M < 50': tM < 50, 'M >= 50': tM >= 50,
      'M >= 50 and cold >= 0.30': (tM >= 50) & (tc >= 0.30), 'M >= 50 and cold < 0.30': (tM >= 50) & (tc < 0.30), 'M < 50 and cold >= 0.30': (tM < 50) & (tc >= 0.30), 'M < 50 and cold < 0.30': (tM < 50) & (tc < 0.30)}
P_('| val slice | feeds | share of public / private feeds | W8P20 - W8 FINAL | W8P20 - W8 stale TO | W16P20x8 - W16 FINAL | W16P20x8 - W16 stale TO | W16P40x8 - W16 FINAL | W16P40x8 - W16 stale TO |')
P_('|---|---|---|---|---|---|---|---|---|')
for lab, g in GR:
    sh = f'{tg[lab][pub].mean():.3f} / {tg[lab][~pub].mean():.3f}' if lab in tg else ''
    cells = []
    for a, b in (('W8P20', 'W8'), ('W16P20x8', 'W16'), ('W16P40x8', 'W16')):
        cells += [f5(wms((R[f'{a}|fin'] - R[f'{b}|fin'])[g])), f5(wms((R[f'{a}|sv'] - R[f'{b}|sv'])[g]))]
    P_(f'| {lab} | {int(g.sum())} | {sh} | ' + ' | '.join(cells) + ' |')

# ---------------- label-free day behaviour
P_(); P_('## T2h. Label-free behaviour by day (FINAL scores; per feed, then mean ± SE over feeds)')
tfid = np.repeat(np.arange(len(tM)), tM); vfid = np.repeat(np.arange(n), m.M['val'])
tcold_row = np.nan_to_num(np.asarray(np.load(FEATS / 'test' / 'it_hist_all_luo.npy'), np.float64), nan=0.0) == 0
vcold_row = np.nan_to_num(np.asarray(np.load(FEATS / 'val' / 'it_hist_all_luo.npy'), np.float64), nan=0.0) == 0
def stats(fid, M, off, a, b, cold):
    nf = len(M); a = a.astype(np.float64); b = b.astype(np.float64)
    da = a - (np.bincount(fid, a, nf) / M)[fid]; db = b - (np.bincount(fid, b, nf) / M)[fid]
    va = np.bincount(fid, da * da, nf); vb = np.bincount(fid, db * db, nf)
    r = np.bincount(fid, da * db, nf) / np.maximum(np.sqrt(va * vb), 1e-12)
    oa = np.lexsort((-a, fid))[off]; ob = np.lexsort((-b, fid))[off]; t1 = (oa == ob).astype(np.float64)
    za = da / np.sqrt(np.maximum(va / M, 1e-12))[fid]; zb = db / np.sqrt(np.maximum(vb / M, 1e-12))[fid]
    cf = cold.astype(np.float64); nc = np.bincount(fid, cf, nf); nw = M - nc; both = (nc > 0) & (nw > 0)
    cw = lambda z: np.bincount(fid, z * cf, nf) / np.maximum(nc, 1) - np.bincount(fid, z * (1 - cf), nf) / np.maximum(nw, 1)
    return r, t1, np.where(both, cw(za) - cw(zb), np.nan)
DN = {20722: 'Sat 09-26', 20723: 'Sun 09-27', 20724: 'Mon 09-28', 20725: 'Tue 09-29'}
tgroups = []
for d_ in sorted(DN):
    md = m.tday == d_
    if (md & pub).any() and (md & ~pub).any(): tgroups += [(f'{DN[d_]} am (public)', md & pub), (f'{DN[d_]} pm (private)', md & ~pub)]
    elif md.any(): tgroups.append((f'{DN[d_]} ({"public" if (md & pub).any() else "private"})', md))
tgroups += [('PUBLIC', pub), ('PRIVATE', ~pub)]
vgroups = [('val Tue 09-22', m.tue), ('val Thu 09-24', m.thu)]
DJ = {}
PAIRS = [('W8P20', 'W8'), ('W16', 'W8'), ('W16P20x8', 'W16'), ('W16P40x8', 'W16'), ('W16P20x8', 'W8P20'), ('W16P40x8', 'W8P20'), ('S16', 'W8'), ('W8TO', 'W8')]
if extra: PAIRS += [('RULEM2040', 'W16P20x8'), ('RULEM2040', 'W8P20')]
for a, b in PAIRS:
    ta, tb = np.load(CACHE / f'testfin_{a}.npy'), np.load(CACHE / f'testfin_{b}.npy'); va, vb = np.load(CACHE / f'valfin_{a}.npy'), np.load(CACHE / f'valfin_{b}.npy')
    rt, t1t, cwt = stats(tfid, tM, m.off['test'], ta, tb, tcold_row); rv, t1v, cwv = stats(vfid, m.M['val'], m.off['val'], va, vb, vcold_row)
    P_(); P_(f'### {a} against {b}')
    P_('| group | feeds | mean cold share | per-feed Pearson r of z | same top-1 item | cold-minus-warm offset, A - B (z units) |')
    P_('|---|---|---|---|---|---|')
    ms_ = lambda x, g: (float(np.nanmean(x[g])), float(np.nanstd(x[g]) / np.sqrt(np.isfinite(x[g]).sum())))
    DJ[f'{a}-{b}'] = {}
    for lab, g in vgroups:
        P_(f'| {lab} | {int(g.sum())} | {m.vcold[g].mean():.3f} | {ms_(rv, g)[0]:.5f} ± {ms_(rv, g)[1]:.5f} | {ms_(t1v, g)[0]:.4f} ± {ms_(t1v, g)[1]:.4f} | {ms_(cwv, g)[0]:+.4f} ± {ms_(cwv, g)[1]:.4f} |')
        DJ[f'{a}-{b}'][lab] = dict(r=ms_(rv, g), top1=ms_(t1v, g), cw=ms_(cwv, g))
    for lab, g in tgroups:
        P_(f'| {lab} | {int(g.sum())} | {np.nanmean(tc[g]):.3f} | {ms_(rt, g)[0]:.5f} ± {ms_(rt, g)[1]:.5f} | {ms_(t1t, g)[0]:.4f} ± {ms_(t1t, g)[1]:.4f} | {ms_(cwt, g)[0]:+.4f} ± {ms_(cwt, g)[1]:.4f} |')
        DJ[f'{a}-{b}'][lab] = dict(r=ms_(rt, g), top1=ms_(t1t, g), cw=ms_(cwt, g))
    # the same disagreement on val, by matching cold-share bins, to translate "more disagreement" into "more effect": |delta| per unit of (1 - r)
J['day'] = DJ
json.dump(J, open(OUT / 't2_forecast.json', 'w'), indent=1, default=float)
open(OUT / 't2_forecast.md', 'w').write('\n'.join(out) + '\n')
