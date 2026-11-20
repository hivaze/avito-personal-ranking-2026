"""premortem t10_branch: popularity-light weight per branch (information; PREDECLARED block 7). Writes out/t10_branch.md."""
from pm_common import *
m = Meta(test=False, hold=False); Wt = m.pmix(); wpriv = Wt['private']
L = lambda k, p: np.load(CACHE / f'pool_{k}_{p}.npy'); Zv = lambda x: m.z('val', x)
out = []
def P_(s=''):
    print(s, flush=True); out.append(s)
P_('## T10. Popularity-light weight per branch: FINAL = 0.5 z((1 - a) z(base FT) + a z(pop FT)) + 0.5 z((1 - b) z(base TO) + b z(pop TO)); delta against the base (a = b = 0)')
for base, pop in (('W8', 'Pd'), ('W16', 'Px8')):
    zb = {k: Zv(L(k, base)) for k in ('val_ft', 'val_to', 'sv')}; zp = {k: Zv(L(k, pop)) for k in ('val_ft', 'val_to', 'sv')}
    mix = lambda k, w: zb[k] if w == 0 else Zv((1 - w) * zb[k] + w * zp[k])
    f0 = m.ndcg('val', 0.5 * zb['val_ft'] + 0.5 * zb['val_to']); s0 = m.ndcg('val', 0.5 * zb['val_ft'] + 0.5 * zb['sv'])
    P_(); P_(f'**{base} + {pop}** (cells: val FINAL ± SE / pmix-private / half-stale)')
    P_('| weight in the FT branch (a) | TO branch b = 0 | b = 0.2 | b = 0.4 |')
    P_('|---|---|---|---|')
    for a in (0.0, 0.2, 0.4):
        cells = []
        for b in (0.0, 0.2, 0.4):
            if a == 0 and b == 0: cells.append('0'); continue
            f = m.ndcg('val', 0.5 * mix('val_ft', a) + 0.5 * mix('val_to', b)); s = m.ndcg('val', 0.5 * mix('val_ft', a) + 0.5 * mix('sv', b))
            cells.append(f'{f5(wms(f - f0))} / {wms(f - f0, wpriv)[0]:+.5f} / {wms(s - s0)[0]:+.5f}')
        P_(f'| a = {a:.1f} | ' + ' | '.join(cells) + ' |')
open(OUT / 't10_branch.md', 'w').write('\n'.join(out) + '\n')
