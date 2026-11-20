"""premortem t7_robust: robustness splits of the candidate upside files against W8P20 / W16 and what a public slot would read. Writes out/t7_robust.md."""
from pm_common import *
m = Meta(test=True, hold=False); Wt = m.pmix(); wpub, wpriv = Wt['public'], Wt['private']
R = dict(np.load(OUT / 'pf.npz')); R.update(dict(np.load(OUT / 'pf_rule.npz')))
big = m.M['val'] >= 50; c = m.vcold; n = len(big)
out = []
def P_(s=''):
    print(s, flush=True); out.append(s)
uu, uinv = np.unique(m.vuser, return_inverse=True); rng = np.random.default_rng(1); hu = (rng.random(len(uu)) < 0.5)[uinv]
SL = [('all', np.ones(n, bool)), ('Tue', m.tue), ('Thu', m.thu), ('M<50', ~big), ('M>=50', big), ('cold<0.30', c < 0.30), ('cold 0.30-0.45', (c >= 0.30) & (c < 0.45)), ('cold>=0.45', c >= 0.45),
      ('user half A', hu), ('user half B', ~hu)]
for view, lab in (('fin', 'val FINAL'), ('sv', 'stale-val TO'), ('svf', 'SVF')):
    P_(f'## T7 ({lab}) by slice')
    P_('| A - B | ' + ' | '.join(s[0] for s in SL) + ' |')
    P_('|---|' + '---|' * len(SL))
    for a, b in (('RULEM1040', 'W8P20'), ('RULEM2040', 'W8P20'), ('W16P40x8', 'W8P20'), ('W16P20x8', 'W8P20'), ('RULEM1040', 'W16'), ('RULEM2040', 'W16'), ('W16P40x8', 'W16P20x8'), ('RULEM1040', 'W16P20x8'), ('RULEM2040', 'W16P20x8'), ('RULEM2040w', 'W16P20w'), ('W8P20', 'W16'), ('W8P20', 'W8')):
        d = R[f'{a}|{view}'] - R[f'{b}|{view}']
        P_(f'| {a} - {b} | ' + ' | '.join(f5(wms(d[g])) for _, g in SL) + ' |')
    P_()
P_('## T7b. What a public slot would read (expected public delta = pmix-public of the val FINAL delta; public SE = sd of the per-feed delta / sqrt(8,562))')
P_('| A - B | val FINAL | pmix-public | pmix-private | public SE of the pair | holdout TO (reported) |')
P_('|---|---|---|---|---|---|')
for a, b in (('RULEM1040', 'W8P20'), ('RULEM2040', 'W8P20'), ('W16P40x8', 'W8P20'), ('W16P30x8', 'W8P20'), ('W16P20x8', 'W8P20'), ('W16P20w', 'W8P20'), ('RULEM2040w', 'W8P20'), ('RULEM1040', 'W16'), ('RULEM2040', 'W16'), ('W16P40x8', 'W16'), ('W16P20x8', 'W16'), ('RULEM1040', 'W16P20x8'), ('RULEM2040', 'W16P20x8'), ('RULEM2040w', 'W16P20w')):
    d = R[f'{a}|fin'] - R[f'{b}|fin']; h = R[f'{a}|hold'] - R[f'{b}|hold']
    P_(f'| {a} - {b} | {f5(wms(d))} | {f5(wms(d, wpub))} | {f5(wms(d, wpriv))} | {d.std(ddof=1) / np.sqrt(8562):.5f} | {f5(wms(h))} |')
open(OUT / 't7_robust.md', 'w').write('\n'.join(out) + '\n')
