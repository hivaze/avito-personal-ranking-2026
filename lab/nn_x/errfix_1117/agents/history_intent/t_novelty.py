import hi, numpy as np
D = hi.Data()
r = D.col('rank')
z = lambda k, fill=0.0: hi.zfeed(np.nan_to_num(D.col(k).astype(float), nan=fill), D.off, D.sz)
NV = 0
def rep(name, g, grid=(0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08), verbose=True):
    global NV; NV += 1
    res = D.crossfit(lambda w: D.s + w * g, list(grid), verbose=verbose)
    print(f"{name:40s} oof {res['oof']:+.5f} se {res['se']:.5f} z {res['z']:+.2f} ch {res['chosen']}", flush=True)
    return res
comp = -(z('hmean5') + z('h1') + z('h10') + z('c_contact') + z('c_vfeed') + hi.zfeed(D.col('same_cat_newest').astype(float), D.off, D.sz)) / 6
comp = comp / comp.std()
rep('novelty composite (all ranks)', comp)
rep('novelty composite (rank<=30)', comp * (r <= 30))
print('variants', NV)
