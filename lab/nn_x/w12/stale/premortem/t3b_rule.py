"""premortem t3b_rule: per-feed arrays and test / val FINAL scores of the feed-size rule that R3 picks most often on W16 + Px8
(RULEM1040 = popularity-light weight 0.1 on feeds with M < 50, 0.4 on feeds with M >= 50; pools W16 and Px8; per branch, FINAL 0.5 / 0.5).
Writes out/pf_rule.npz, cache/{testfin,valfin}_RULEM1040.npy. In memory compositions only: no submission file."""
from pm_common import *
m = Meta(test=True, hold=False)
R = dict(np.load(OUT / 'pf.npz')); big = m.M['val'] >= 50
O = {}
for name, lo, hi in (('RULEM1040', 'W16P10x8', 'W16P40x8'), ('RULEM2040', 'W16P20x8', 'W16P40x8'), ('RULEM1040w8', 'W8P10x8', 'W8P40x8'), ('RULEM1040d', 'W8P10', 'W8P40'), ('RULEM2040w', 'W16P20w', 'W16P40w')):
    for v in ('fin', 'ft', 'to', 'sv', 'svf'):
        O[f'{name}|{v}'] = np.where(big, R[f'{hi}|{v}'], R[f'{lo}|{v}'])
    O[f'{name}|hold'] = R[f'{hi}|hold']                      # every holdout feed has M >= 50
    for sp, M in (('test', m.M['test']), ('val', m.M['val'])):
        rb = np.repeat(M >= 50, M)
        np.save(CACHE / f'{sp}fin_{name}.npy', np.where(rb, np.load(CACHE / f'{sp}fin_{hi}.npy'), np.load(CACHE / f'{sp}fin_{lo}.npy')).astype(np.float32))
    print(name, 'val FINAL', O[f'{name}|fin'].mean().round(5), 'SVF', O[f'{name}|svf'].mean().round(5))
np.savez(OUT / 'pf_rule.npz', **O)
