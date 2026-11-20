"""H. The FT half by positive class (cold = it_hist_all_luo == 0, the r6_eval definition) and under the public / private feed mix (src/pmix.py weights:
M bin x cold-share bin of val re-weighted to the test parts; label-free weights). Pool level: FT pool - TO pool, file (0.5/0.5) - TO pool."""
from lib import *
import pmix
luo = np.load(FEATS / 'val/it_hist_all_luo.npy'); cold = np.nan_to_num(np.asarray(luo, np.float64), nan=0.0) == 0
npos = np.add.reduceat(yv, ov); ncold = np.add.reduceat(yv * cold, ov); cs = np.add.reduceat(cold.astype(float), ov) / Mv
SL = {'all': np.ones(len(Mv), bool), 'coldpos': (npos > 0) & (ncold == npos), 'warmpos': (npos > 0) & (ncold == 0), 'mixed': (npos > 0) & (ncold > 0) & (ncold < npos),
      'cold share <15%': cs < 0.15, '15-30%': (cs >= 0.15) & (cs < 0.30), '30-45%': (cs >= 0.30) & (cs < 0.45), '>=45%': cs >= 0.45}
print('slice sizes:', {k: int(v.sum()) for k, v in SL.items()}, '| val cold share of candidates', cold.mean().round(3))
Wt = pmix.weights(); wpub, wpri = np.asarray(Wt['public'], float), np.asarray(Wt['private'], float)
def wmean(d, w): m = (d * w).sum() / w.sum(); s = np.sqrt(((w * (d - m)) ** 2).sum()) / w.sum(); return f'{m:+.5f}+-{s:.5f}'
POOLS = {'W8': W8, 'N8': N8, 'S16': W8 + N8, 'C3': C3M, 'G3': GROUPS['H:gru'] + GROUPS['H:m02'] + GROUPS['H:m02p70k']}
for k, ms in POOLS.items():
    a, b = zv(pool(ms, 'to', 'val')), zv(pool(ms, 'ft', 'val')); e0, e1, eh, eq = ev(a), ev(b), ev(0.5 * a + 0.5 * b), ev(0.75 * a + 0.25 * b)
    print(f'\n{k}:')
    for nm, d in (('FT pool - TO pool', e1 - e0), ('file 0.5/0.5 - TO pool', eh - e0), ('w_ft 0.25 - TO pool', eq - e0)):
        print(f'  {nm:24s} ' + ' | '.join(f'{s} {f(d[m])}' for s, m in SL.items()))
        print(f'  {"":24s} public mix {wmean(d, wpub)} | private mix {wmean(d, wpri)}')
