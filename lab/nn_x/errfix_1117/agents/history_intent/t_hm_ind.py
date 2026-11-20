import hi, numpy as np
D = hi.Data()
hm = np.nan_to_num(D.col('hmean5').astype(float), nan=0); r = D.col('rank')
NV = 0
def rep(name, g, grid=(0, 0.03, 0.06, 0.1, 0.15, 0.2, 0.3, 0.4), verbose=True):
    global NV; NV += 1
    res = D.crossfit(lambda w: D.s + w * g, list(grid), verbose=verbose)
    print(f"{name:40s} oof {res['oof']:+.5f} se {res['se']:.5f} z {res['z']:+.2f} ch {res['chosen']}", flush=True)
for K in [10, 30]:
    for t in [0.75, 0.8]:
        rep(f'1[hmean5>={t}] & rank<={K}', ((hm >= t) & (r <= K)).astype(float))
# pairwise in top10: pairs with exactly one positive where exactly one has hm>=.8
rows = []
for o, m in zip(D.off, D.sz):
    idx = o + np.argsort(r[o:o+m])[:10]
    yy = D.y[idx]; hh = hm[idx] >= 0.8; ss = D.s[idx]
    for a in range(len(idx)):
        for b in range(a+1, len(idx)):
            if yy[a] != yy[b] and hh[a] != hh[b]:
                rows.append((ss[a]-ss[b], yy[a], hh[a]))
rows = np.array(rows)
gap, upper_pos, upper_hm = rows.T
print('pairs', len(rows))
for lo, hi_ in [(0, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.4), (0.4, 9)]:
    m = (gap >= lo) & (gap < hi_)
    a = m & (upper_hm == 1); b = m & (upper_hm == 0)
    print(f'gap [{lo},{hi_}): upper is hm>=.8: P(upper pos)={upper_pos[a].mean():.3f} n={a.sum()} | upper is hm<.8: P(upper pos)={upper_pos[b].mean():.3f} n={b.sum()}')
