"""premortem t1_table: the witness table (FINAL level + per branch) and the sanity comparison with the scored pairs (rule R7).
Reads out/pf.npz (t1_build.py). Writes out/t1_table.md, out/t1_pairs.json."""
import json
from pm_common import *

m = Meta(test=False, hold=False)
R = dict(np.load(OUT / 'pf.npz')); Wt = m.pmix(); wpub, wpriv = Wt['public'], Wt['private']
big = m.M['val'] >= 50
PUB = {'W8P20': 0.3950094, 'W8': 0.394486, 'W16': 0.39448337, 'S16': 0.3940782, 'W8TO': 0.394009, 'C3': 0.393454, 'G3': 0.392820}
LAG = {  # lagged holdout on record, TRAIN-ONLY level, private-like view: (subset 13,424 feeds where the gate holds, all 30,000 feeds) [D poplag_verify 1.3 / witness 3.4]
    ('W8P10', 'W8'): ('+0.00013 ± 0.00021', '+0.00017 ± 0.00014'), ('W8P20', 'W8'): ('+0.00024 ± 0.00031', '+0.00020 ± 0.00021'),
    ('W8P30', 'W8'): ('-0.00015 ± 0.00038', '+0.00007 ± 0.00026'), ('W8P40', 'W8'): ('-0.00053 ± 0.00045', 'n/a')}
out = []
def P_(s=''):
    print(s, flush=True); out.append(s)

def cols(a, b):
    r = {}
    d = R[f'{a}|fin'] - R[f'{b}|fin']
    r['val'] = wms(d); r['M50p'] = wms(d[big]); r['M50m'] = wms(d[~big]); r['tue'] = wms(d[m.tue]); r['thu'] = wms(d[m.thu])
    r['ppub'] = wms(d, wpub); r['ppriv'] = wms(d, wpriv)
    r['sd'] = float(d.std(ddof=1)); r['se_pub'] = r['sd'] / np.sqrt(8562)
    for k in ('ft', 'to', 'sv', 'svf', 'hold'):
        dd = R[f'{a}|{k}'] - R[f'{b}|{k}']; r[k] = wms(dd)
        if k in ('ft', 'to', 'sv', 'svf'): r[k + '_ppriv'] = wms(dd, wpriv); r[k + '_ppub'] = wms(dd, wpub)
    dsv = R[f'{a}|sv'] - R[f'{b}|sv']; r['sv_M50p'] = wms(dsv[big]); r['sv_M50m'] = wms(dsv[~big])
    r['public'] = (PUB[a] - PUB[b]) if a in PUB and b in PUB else None
    return r

NAMES = ['W8', 'W16', 'S16', 'W8P20', 'W8TO', 'C3', 'G3', 'W8b',
         'W8P10', 'W8P30', 'W8P40', 'W16P10d', 'W16P20d', 'W16P30d', 'W16P40d', 'W16P10x8', 'W16P20x8', 'W16P30x8', 'W16P40x8',
         'W16P10w', 'W16P20w', 'W16P30w', 'W16P40w', 'W8P10x8', 'W8P20x8', 'W8P30x8', 'W8P40x8', 'W16TO', 'Pd', 'Pw', 'Px8']
P_('## T1a. Levels (val 14,423 feeds; holdout 30,000 feeds; FINAL = 0.5 z(FT) + 0.5 z(TO); pmix = FINAL reweighted to the public / private mix)')
P_('| file | val FINAL | val FT branch | val TO branch | pmix-public | pmix-private | stale-val TO | SVF | holdout TO | PUBLIC |')
P_('|---|---|---|---|---|---|---|---|---|---|')
for n in NAMES:
    f = R[f'{n}|fin']
    P_(f'| {n} | {f.mean():.5f} | {R[f"{n}|ft"].mean():.5f} | {R[f"{n}|to"].mean():.5f} | {np.average(f, weights=wpub):.5f} | {np.average(f, weights=wpriv):.5f} | '
       f'{R[f"{n}|sv"].mean():.5f} | {R[f"{n}|svf"].mean():.5f} | {R[f"{n}|hold"].mean():.5f} | {PUB.get(n, "")} |')

def table(title, ref, names):
    P_(); P_(f'## {title}')
    P_('| A - ' + ref + ' | val FINAL | val M>=50 | val M<50 | val Tue | val Thu | pmix-public | pmix-private | stale-val TO | SVF | holdout TO (reported) | lagged holdout TO, subset / all 30K [D] | PUBLIC ± SE |')
    P_('|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    J = {}
    for n in names:
        if n == ref: continue
        r = cols(n, ref); J[n] = r
        lg = LAG.get((n, ref)); lg = f'{lg[0]} / {lg[1]}' if lg else ''
        pb = f'{r["public"]:+.6f} ± {r["se_pub"]:.5f}' if r['public'] is not None else ''
        P_(f'| {n} | {f5(r["val"])} | {r["M50p"][0]:+.5f} | {r["M50m"][0]:+.5f} | {r["tue"][0]:+.5f} | {r["thu"][0]:+.5f} | {f5(r["ppub"])} | {f5(r["ppriv"])} | {f5(r["sv"])} | {f5(r["svf"])} | {f5(r["hold"])} | {lg} | {pb} |')
    return J

ALL = {}
ALL['W8'] = table('T1b. Deltas against W8 (FINAL level unless the column says TO)', 'W8', NAMES[:-3])
ALL['W16'] = table('T1c. Deltas against W16', 'W16', ['W8', 'W8P20', 'W16P10d', 'W16P20d', 'W16P30d', 'W16P40d', 'W16P10x8', 'W16P20x8', 'W16P30x8', 'W16P40x8', 'W16P10w', 'W16P20w', 'W16P30w', 'W16P40w', 'S16', 'W16TO'])
ALL['W8P20'] = table('T1d. Deltas against W8P20 (the public-best file)', 'W8P20', ['W8', 'W16', 'S16', 'W8P10', 'W8P30', 'W8P40', 'W16P10d', 'W16P20d', 'W16P30d', 'W16P40d', 'W16P10x8', 'W16P20x8', 'W16P30x8', 'W16P40x8', 'W16P20w', 'W16P30w', 'W16P40w', 'W8P20x8', 'W8P30x8', 'W8P40x8', 'C3'])

P_(); P_('## T1e. Per branch (delta against the reference, same branch): val / pmix-private')
P_('| A - ref | FT branch val | FT branch pmix-private | TO branch val | TO branch pmix-private | TO stale-val | TO stale-val M>=50 | TO stale-val M<50 |')
P_('|---|---|---|---|---|---|---|---|')
for a, b in [('W16', 'W8'), ('S16', 'W8'), ('W8P20', 'W8'), ('C3', 'G3'), ('W8', 'C3'), ('W16P20d', 'W16'), ('W16P20x8', 'W16'), ('W16P30x8', 'W16'), ('W16P40x8', 'W16'), ('W16P20w', 'W16'), ('W16P40w', 'W16'),
             ('W8P30', 'W8'), ('W8P40', 'W8'), ('W16P20x8', 'W8P20'), ('W16P20d', 'W8P20'), ('W16', 'W8P20')]:
    r = cols(a, b)
    P_(f'| {a} - {b} | {f5(r["ft"])} | {f5(r["ft_ppriv"])} | {f5(r["to"])} | {f5(r["to_ppriv"])} | {f5(r["sv"])} | {f5(r["sv_M50p"])} | {f5(r["sv_M50m"])} |')

# ---- R7 sanity comparison with public
P_(); P_('## T1f. Sanity comparison of the columns with the scored pairs (rule R7; nothing fitted)')
PAIRS = [('W16', 'W8'), ('W8P20', 'W8'), ('W8TO', 'W8'), ('S16', 'W8'), ('C3', 'G3'), ('W8', 'C3'), ('G3', 'W8')]
P_('| pair | PUBLIC ± SE | val FINAL | pmix-public | pmix-private | SVF | TO val | TO stale-val | holdout TO |')
P_('|---|---|---|---|---|---|---|---|---|')
S = {}
for a, b in PAIRS:
    r = cols(a, b); S[(a, b)] = r
    P_(f'| {a} - {b} | {r["public"]:+.6f} ± {r["se_pub"]:.5f} | {f5(r["val"])} | {f5(r["ppub"])} | {f5(r["ppriv"])} | {f5(r["svf"])} | {f5(r["to"])} | {f5(r["sv"])} | {f5(r["hold"])} |')
P_()
P_('| witness column | sign right on pairs with abs(public) >= 1 public SE | sign right, all 6 independent pairs | X2 (6 pairs) | slope through the origin of public on the column (6 pairs) |')
P_('|---|---|---|---|---|')
ind = PAIRS[:6]
for k, lab in (('val', 'val FINAL'), ('ppub', 'pmix-public FINAL'), ('ppriv', 'pmix-private FINAL'), ('svf', 'SVF'), ('to', 'TO val'), ('sv', 'TO stale-val'), ('hold', 'holdout TO')):
    big_ = [p for p in ind if abs(S[p]['public']) >= S[p]['se_pub']]
    sr = sum(np.sign(S[p][k][0]) == np.sign(S[p]['public']) for p in big_); sa = sum(np.sign(S[p][k][0]) == np.sign(S[p]['public']) for p in ind)
    x2 = np.mean([(S[p]['public'] - S[p][k][0]) ** 2 / (S[p]['se_pub'] ** 2 + S[p][k][1] ** 2) for p in ind])
    xs = np.array([S[p][k][0] for p in ind]); ys = np.array([S[p]['public'] for p in ind]); sl = float((xs * ys).sum() / (xs * xs).sum())
    P_(f'| {lab} | {sr} / {len(big_)} | {sa} / 6 | {x2:.2f} | {sl:+.2f} |')
P_('(W8TO - W8 has no train-only difference on the TO columns by construction: the TO columns of that pair are 0 and count as wrong sign.)')
json.dump({k: {n: {kk: vv for kk, vv in r.items()} for n, r in v.items()} for k, v in ALL.items()}, open(OUT / 't1_pairs.json', 'w'), indent=1, default=float)
open(OUT / 't1_table.md', 'w').write('\n'.join(out) + '\n')
