import hi, numpy as np
D = hi.Data()
M = D.col('M'); cni = D.col('cat_newest_idx')
f = lambda k: np.nan_to_num(D.col(k).astype(float), nan=0.0)
big = M >= 150
for name, g in [('cni<=5', ((cni >= 0) & (cni <= 5)).astype(float)), ('h1', f('h1')), ('hmean5', f('hmean5')), ('h10', f('h10'))]:
    gz = hi.zfeed(g, D.off, D.sz)
    X = np.c_[D.s, gz * big]
    b, nll = hi.clogit_fit(X, D.y, D.off, D.sz)
    b0, nll0 = hi.clogit_fit(np.c_[D.s], D.y, D.off, D.sz)
    print(f'{name:8s} coef on z(g) in M>=150: {b[1]:+.4f} (score coef {b[0]:.3f}); dNLL {nll0-nll:.2f}; shift in score units per sd {b[1]/b[0]:+.4f}')
