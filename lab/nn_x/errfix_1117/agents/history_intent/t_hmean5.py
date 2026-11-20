import hi, numpy as np
D = hi.Data(); df = D.df
hm = np.nan_to_num(D.col('hmean5'), nan=0.0); r = D.col('rank')
NV = 0
def rep(name, make, grid):
    global NV; NV += 1
    res = D.crossfit(make, grid)
    best_in = max(grid, key=lambda w: res['curve'][w][0])
    print(f"{name:45s} oof {res['oof']:+.5f} se {res['se']:.5f} z {res['z']:+.2f} chosen {res['chosen']} | best in-sample w={best_in} {res['curve'][best_in][0]:+.5f}", flush=True)
grid = [0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0]
for K in [3, 10, 30, 100000]:
    for t in [0.6, 0.7, 0.8]:
        g = np.maximum(hm - t, 0) * (r <= K)
        rep(f'relu(hmean5-{t}) rank<={K}', lambda w: D.s + w * g, grid)
# linear in hmean5 (centred within feed)
for K in [10, 100000]:
    z = hi.zfeed(hm, D.off, D.sz)
    g = z * (r <= K)
    rep(f'z(hmean5) rank<={K}', lambda w: D.s + w * g, [0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3])
    g = hm * (r <= K)
    rep(f'hmean5 rank<={K}', lambda w: D.s + w * g, [0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0])
print('variants', NV)
