import hi, numpy as np
D = hi.Data()
M = D.col('M'); cni = D.col('cat_newest_idx')
f = lambda k: np.nan_to_num(D.col(k).astype(np.float64), nan=0.0)
NV = 0
def rep(name, g, grid):
    global NV; NV += 1
    res = D.crossfit(lambda w: D.s + w * g, grid)
    best_in = max(grid, key=lambda w: res['curve'][w][0])
    print(f"{name:34s} oof {res['oof']:+.5f} se {res['se']:.5f} z {res['z']:+.2f} ch {res['chosen']} | in w={best_in} {res['curve'][best_in][0]:+.5f} ({res['curve'][best_in][1]:.5f})", flush=True)
grid = [0, 0.0025, 0.005, 0.01, 0.02, 0.04, 0.06, 0.1]
for k in ['h1', 'h5', 'h10', 'hmean5', 'h32', 'hmax', 'c_contact', 'c_vfeed', 'c_vsearch', 'ucos']:
    g = hi.zfeed(f(k), D.off, D.sz)
    rep(f'z({k}) all', g, grid)
    rep(f'-z({k}) all', -g, grid)
print('variants', NV)
