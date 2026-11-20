"""premortem t8_sameview: is the gain of the 0.2 popularity-light admixture on an 8-base specific to the view, or what any 20% of fresh members give?
Same-view control = 0.8 z(base 8) + 0.2 z(the other 8 wide students). FINAL level (per branch, 0.5 / 0.5). Writes out/t8_sameview.md."""
from pm_common import *
m = Meta(test=False, hold=True); Wt = m.pmix(); wpriv = Wt['private']
L = lambda k, p: np.load(CACHE / f'pool_{k}_{p}.npy'); Zv = lambda x: m.z('val', x)
def comp(base, pop, w):
    b = {kd: (Zv(L(kd, base)) if pop is None else Zv((1 - w) * Zv(L(kd, base)) + w * Zv(L(kd, pop)))) for kd in ('val_ft', 'val_to', 'sv')}
    h = L('hold', base) if pop is None else (1 - w) * m.z('hold', L('hold', base)) + w * m.z('hold', L('hold', pop))
    return dict(fin=m.ndcg('val', 0.5 * b['val_ft'] + 0.5 * b['val_to']), svf=m.ndcg('val', 0.5 * b['val_ft'] + 0.5 * b['sv']), sv=m.ndcg('val', b['sv']), hold=m.ndcg('hold', m.z('hold', h)))
out = []
def P_(s=''):
    print(s, flush=True); out.append(s)
P_('## T8. Popularity-light admixture against a same-view admixture of the same weight (FINAL level; paired per feed)')
P_('| base | admixture (w 0.2) | val FINAL vs base | pmix-private | SVF | stale-val TO | holdout TO |')
P_('|---|---|---|---|---|---|---|')
for base, other in (('W8', 'W8b'), ('W8b', 'W8')):
    b0 = comp(base, None, 0); rows = {}
    for pop in ('Pd', 'Pw', 'Px8', other):
        r = comp(base, pop, 0.2); rows[pop] = r
        P_(f'| {base} | {pop}{" (same view: the other 8 wide students)" if pop == other else ""} | {f5(wms(r["fin"] - b0["fin"]))} | {f5(wms(r["fin"] - b0["fin"], wpriv))} | {f5(wms(r["svf"] - b0["svf"]))} | {f5(wms(r["sv"] - b0["sv"]))} | {f5(wms(r["hold"] - b0["hold"]))} |')
    for pop in ('Pd', 'Pw', 'Px8'):
        r, s = rows[pop], rows[other]
        P_(f'| {base} | {pop} MINUS same view | {f5(wms(r["fin"] - s["fin"]))} | {f5(wms(r["fin"] - s["fin"], wpriv))} | {f5(wms(r["svf"] - s["svf"]))} | {f5(wms(r["sv"] - s["sv"]))} | {f5(wms(r["hold"] - s["hold"]))} |')
open(OUT / 't8_sameview.md', 'w').write('\n'.join(out) + '\n')
