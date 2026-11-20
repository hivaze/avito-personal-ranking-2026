"""premortem t5_misc: small reads for the assumption audit. Writes out/t5_misc.md.
 (a) more fine-tune seeds: G3 with 3 FT seeds (ens_G3s3 = C2) against G3 (1 seed); C3 (ens_Q2a8s3) against C1 (ens_Q2a8pop1, 1 FT seed): val FINAL, pmix, and the public deltas on record.
 (b) slope of public on the val columns over the 6 independent scored pairs, with its SE; residuals of the 7 scored files against the level line on record.
 (c) two more scored pairs for the sanity table: C1 - G3, C3 - C2."""
from pm_common import *
m = Meta(test=False, hold=False); Wt = m.pmix(); wpub, wpriv = Wt['public'], Wt['private']
R = dict(np.load(OUT / 'pf.npz'))
out = []
def P_(s=''):
    print(s, flush=True); out.append(s)
def fin(tag):
    return m.ndcg('val', 0.5 * m.z('val', np.load(P / f'ens_{tag}_ft_val.npy')) + 0.5 * m.z('val', np.load(P / f'ens_{tag}_to_val.npy')))
E = {t: fin(t) for t in ('G3', 'G3s3', 'Q2a8pop1', 'Q2a8s3')}
PUBF = {'G3': 0.392820, 'G3s3': 0.392754, 'Q2a8pop1': 0.393394, 'Q2a8s3': 0.393454}
P_('## T5a. More fine-tune seeds (3 seeds per fold against 1), scored pairs on record')
P_('| pair | val FINAL | pmix-public | pmix-private | PUBLIC ± SE |')
P_('|---|---|---|---|---|')
for a, b, lab in (('G3s3', 'G3', 'C2 - G3 (G3 members, FT 3 seeds - 1 seed)'), ('Q2a8s3', 'Q2a8pop1', 'C3 - C1 (G3 + pop1 x8, FT 3 seeds - 1 seed)'),
                  ('Q2a8pop1', 'G3', 'C1 - G3 (pop1 x8 added, FT 1 seed)'), ('Q2a8s3', 'G3s3', 'C3 - C2 (pop1 x8 added, FT 3 seeds)')):
    d = E[a] - E[b]
    P_(f'| {lab} | {f5(wms(d))} | {f5(wms(d, wpub))} | {f5(wms(d, wpriv))} | {PUBF[a] - PUBF[b]:+.6f} ± {d.std(ddof=1) / np.sqrt(8562):.5f} |')
PUB = {'W8P20': 0.3950094, 'W8': 0.394486, 'W16': 0.39448337, 'S16': 0.3940782, 'W8TO': 0.394009, 'C3': 0.393454, 'G3': 0.392820}
P_(); P_('## T5b. Level: the 7 scored files against the line on record (public = 0.0074 + 0.963 val; no refit)')
P_('| file | val FINAL | PUBLIC | line | residual | pmix-public level | PUBLIC - pmix-public level |')
P_('|---|---|---|---|---|---|---|')
res = []
for k, pv in PUB.items():
    f = R[f'{k}|fin']; line = 0.0074 + 0.963 * f.mean(); res.append(pv - line); pl_ = np.average(f, weights=wpub)
    P_(f'| {k} | {f.mean():.5f} | {pv:.6f} | {line:.5f} | {pv - line:+.5f} | {pl_:.5f} | {pv - pl_:+.5f} |')
P_(f'Residual sd over the 7 files {np.std(res, ddof=1):.5f} (on record: 0.0006); mean {np.mean(res):+.5f}.')
P_(); P_('## T5c. Slope of the public delta on each val column, 6 independent scored pairs (through the origin; SE from the public SEs only)')
P_('| column | slope ± SE | chi2 of public against slope = 1 (6 df) |')
P_('|---|---|---|')
PAIRS = [('W16', 'W8'), ('W8P20', 'W8'), ('W8TO', 'W8'), ('S16', 'W8'), ('C3', 'G3'), ('W8', 'C3')]
for view, lab in (('fin', 'val FINAL'), ('ppub', 'pmix-public FINAL'), ('ppriv', 'pmix-private FINAL'), ('svf', 'SVF')):
    xs, ys, ss = [], [], []
    for a, b in PAIRS:
        d = R[f'{a}|fin'] - R[f'{b}|fin']; dv = R[f'{a}|svf'] - R[f'{b}|svf']
        x = wms(d)[0] if view == 'fin' else wms(d, wpub)[0] if view == 'ppub' else wms(d, wpriv)[0] if view == 'ppriv' else wms(dv)[0]
        xs.append(x); ys.append(PUB[a] - PUB[b]); ss.append(d.std(ddof=1) / np.sqrt(8562))
    xs, ys, ss = map(np.array, (xs, ys, ss)); w = 1 / ss ** 2
    sl = (w * xs * ys).sum() / (w * xs * xs).sum(); sse = 1 / np.sqrt((w * xs * xs).sum()); chi = (((ys - xs) / ss) ** 2).sum()
    P_(f'| {lab} | {sl:+.2f} ± {sse:.2f} | {chi:.1f} |')
open(OUT / 't5_misc.md', 'w').write('\n'.join(out) + '\n')
