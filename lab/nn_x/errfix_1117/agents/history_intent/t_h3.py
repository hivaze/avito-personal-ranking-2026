import hi, numpy as np
D = hi.Data()
nt = D.col('newest_type'); ns = D.col('newest_src'); r = D.col('rank')
h1 = np.nan_to_num(D.col('h1').astype(float), nan=-1); h5 = np.nan_to_num(D.col('h5').astype(float), nan=-1)
NV = 0
def rep(name, g, grid=(0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75), verbose=False):
    global NV; NV += 1
    gs = [-w for w in grid]
    res = D.crossfit(lambda w: D.s + w * g, gs, verbose=verbose)
    best_in = max(gs, key=lambda w: res['curve'][w][0])
    print(f"{name:52s} n={int((g!=0).sum()):6d} oof {res['oof']:+.5f} se {res['se']:.5f} z {res['z']:+.2f} ch {res['chosen']} | in w={best_in} {res['curve'][best_in][0]:+.5f}", flush=True)
srch = (nt == 1) & (ns == 2)
oth = (nt == 1) & np.isin(ns, [3, 4, 5, 6, 7])
rep('newest view src=search & h1>=0.9', (srch & (h1 >= 0.9)).astype(float), verbose=True)
rep('newest view src in 3-7 & h1>=0.9', (oth & (h1 >= 0.9)).astype(float), verbose=True)
rep('newest view src=search & h5>=0.9', (srch & (h5 >= 0.9)).astype(float))
print('variants', NV)
# positives/rows in segment
m = (srch | oth) & (h1 >= 0.9)
print('rows', m.sum(), 'positives', D.y[m].sum(), 'feeds', len(np.unique(D.feed[m])))
