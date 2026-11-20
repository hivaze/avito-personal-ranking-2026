import hi, numpy as np
D = hi.Data()
raw = lambda k: D.col(k).astype(np.float64)
r = D.col('rank')
NV = 0
def rep(name, g, grid, verbose=False):
    global NV; NV += 1
    res = D.crossfit(lambda w: D.s + w * g, grid, verbose=verbose)
    best_in = max(grid, key=lambda w: res['curve'][w][0])
    print(f"{name:40s} oof {res['oof']:+.5f} se {res['se']:.5f} z {res['z']:+.2f} ch {res['chosen']} | in w={best_in} {res['curve'][best_in][0]:+.5f} ({res['curve'][best_in][1]:.5f})", flush=True)
grid = [0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12]
vf = raw('c_vfeed'); vs = raw('c_vsearch'); hm = raw('hmax')
g = -hi.zfeed(np.nan_to_num(vf, nan=0), D.off, D.sz)
rep('-z(c_vfeed)', g, grid, verbose=True)
d = np.where(np.isnan(vf) | np.isnan(vs), 0, vs - vf)
rep('z(vsearch - vfeed)', hi.zfeed(d, D.off, D.sz), grid, verbose=True)
rep('raw(vsearch - vfeed)', d, [0, 0.05, 0.1, 0.2, 0.3, 0.5], verbose=True)
# vfeed is the nearest event (feed-origin) vs other
d2 = np.where(np.isnan(vf), 0, hm - vf)   # >0: nearest history item not a feed view
rep('raw(hmax - vfeed)', d2, [0, 0.05, 0.1, 0.2, 0.3, 0.5], verbose=True)
print('variants', NV)
