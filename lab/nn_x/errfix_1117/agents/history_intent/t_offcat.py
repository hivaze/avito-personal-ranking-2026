import hi, numpy as np
D = hi.Data()
r = D.col('rank'); scn = D.col('same_cat_newest'); cnf = D.col('cat_n_in_feed'); M = D.col('M'); crk = D.col('cat_rank_in_feed'); cni = D.col('cat_newest_idx')
cshare = cnf / M
NV = 0
def rep(name, g, grid=(0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5), sign=-1, verbose=False):
    global NV; NV += 1
    gs = [sign * w for w in grid]
    res = D.crossfit(lambda w: D.s + w * g, gs, verbose=verbose)
    best_in = max(gs, key=lambda w: res['curve'][w][0])
    print(f"{name:52s} n={int((g!=0).sum()):6d} oof {res['oof']:+.5f} se {res['se']:.5f} z {res['z']:+.2f} ch {res['chosen']} | in w={best_in} {res['curve'][best_in][0]:+.5f}", flush=True)
off = ~scn
for K in [3, 5, 10, 20, 100000]:
    rep(f'demote off-newest-cat rank<={K}', (off & (r <= K)).astype(float))
for K in [3, 10, 100000]:
    rep(f'demote off-newest-cat cshare<0.2 rank<={K}', (off & (cshare < 0.2) & (r <= K)).astype(float))
    rep(f'demote cni==-1 rank<={K}', ((cni < 0) & (r <= K)).astype(float))
    rep(f'demote off-cat & cat_rank_in_feed==1 rank<={K}', (off & (crk == 1) & (r <= K)).astype(float))
print('variants', NV)
