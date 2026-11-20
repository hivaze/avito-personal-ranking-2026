"""K. Information: optimal global cold offset delta (z_TO + delta * cold) by val day, W8 and C3 TO pools; and the FT pool minus TO pool by day."""
from lib import *
luo = np.load(FEATS / 'val/it_hist_all_luo.npy'); cold = (np.nan_to_num(np.asarray(luo, np.float64), nan=0.0) == 0).astype(np.float64)
day = (tv.astype(np.int64) + 3 * 3600) // 86400; dd = np.unique(day); D = {'Tue': day == dd[0], 'Thu': day == dd[2]}
grid = np.round(np.arange(-0.06, 0.19, 0.03), 2)
for k, ms in (('W8', W8), ('S16', W8 + N8), ('C3', C3M)):
    a = zv(pool(ms, 'to', 'val')); e0 = ev(a); E = np.array([ev(a + g * cold) - e0 for g in grid])
    for dn, m in D.items():
        i = int(np.argmax(E[:, m].mean(1)))
        print(f'{k:4s} {dn}: ' + ' '.join(f'{g:+.2f}:{E[q][m].mean():+.5f}' for q, g in enumerate(grid)) + f' | best {grid[i]:+.2f} ({f(E[i][m])})')
    d = E[:, D['Thu']].mean(1) - E[:, D['Tue']].mean(1); j = list(grid).index(0.06)
    sed = np.sqrt(E[j][D['Thu']].var() / D['Thu'].sum() + E[j][D['Tue']].var() / D['Tue'].sum())
    print(f'     Thu - Tue at delta +0.06: {d[j]:+.5f} +- {sed:.5f}')
