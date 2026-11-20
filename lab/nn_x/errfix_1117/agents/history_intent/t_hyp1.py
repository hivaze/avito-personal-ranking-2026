import hi, numpy as np
D = hi.Data()
r = D.col('rank'); nt = D.col('newest_type'); ns = D.col('newest_src')
raw = lambda k: D.col(k).astype(np.float64)
f = lambda k: np.nan_to_num(raw(k), nan=-1.0)
NV = 0
def rep(name, g, grid=(0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0), sign=1):
    global NV; NV += 1
    gs = [sign * w for w in grid]
    res = D.crossfit(lambda w: D.s + w * g, gs)
    best_in = max(gs, key=lambda w: res['curve'][w][0])
    print(f"{name:52s} n={int((g!=0).sum()):6d} oof {res['oof']:+.5f} se {res['se']:.5f} z {res['z']:+.2f} ch {res['chosen']} | in w={best_in} {res['curve'][best_in][0]:+.5f}", flush=True)
hmax = f('hmax'); cfav = f('c_fav'); ccon = f('c_contact'); cord = f('c_order'); h1 = f('h1'); h5 = f('h5')
favnear = (raw('c_fav') >= hmax - 1e-6)
for K in [3, 10, 1000]:
    for t in [0.8, 0.9, 0.95]:
        rep(f'H1 demote fav-nearest c_fav>={t} rank<={K}', (favnear & (cfav >= t) & (r <= K)).astype(float), sign=-1)
con = np.isin(nt, [2, 3])
for t in [0.9, 0.95]:
    rep(f'H2 boost newest-contact & c_contact>={t}', (con & (ccon >= t)).astype(float))
    rep(f'H2b boost newest-contact & h1>={t}', (con & (h1 >= t)).astype(float))
    rep(f'H2c boost any & c_contact>={t} rank>10', ((ccon >= t) & (r > 10)).astype(float))
srch = (nt == 1) & (ns != 1) & (ns != 0)
for t in [0.85, 0.9, 0.95]:
    rep(f'H3 demote newest view non-feed & h1>={t}', (srch & (h1 >= t)).astype(float), sign=-1)
for t in [0.9, 0.95]:
    rep(f'H4 boost c_order>={t}', (cord >= t).astype(float))
print('variants', NV)
