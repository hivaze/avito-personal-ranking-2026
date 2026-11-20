import hi, numpy as np
D = hi.Data()
M = D.col('M'); r = D.col('rank'); cni = D.col('cat_newest_idx')
f = lambda k: np.nan_to_num(D.col(k).astype(np.float64), nan=0.0)
NV = 0
def rep(name, make, grid):
    global NV; NV += 1
    res = D.crossfit(make, grid)
    best_in = max(grid, key=lambda w: res['curve'][w][0])
    print(f"{name:40s} oof {res['oof']:+.5f} se {res['se']:.5f} z {res['z']:+.2f} ch {res['chosen']} | in w={best_in} {res['curve'][best_in][0]:+.5f} ({res['curve'][best_in][1]:.5f})", flush=True)
lcni = np.where(cni < 0, np.log1p(300), np.log1p(np.maximum(cni, 0)))
G = {'cni<=5': ((cni >= 0) & (cni <= 5)).astype(float), '-log1p_cni': -lcni / np.log1p(300),
     'h1': f('h1'), 'h5': f('h5'), 'h10': f('h10'), 'h32': f('h32'), 'hmean5': f('hmean5'), 'ucos': f('ucos')}
grid = [0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
for T in [100, 150, 200]:
    for gn, g in G.items():
        gz = hi.zfeed(g, D.off, D.sz) * (M >= T)
        rep(f'M>={T} z({gn})', lambda w: D.s + w * 0.2 * gz, grid)
print('variants', NV)
