"""R7: are short val feeds 'positives-kept' sub-feeds? positives per feed by M: val vs train exploration (holdout) vs thinning models.
 Relevant to HOLD-SHORT (lb_stats#5, pool_composition#12) and to kwsub (--aug_cand keeps every positive)."""
from lo_common import *
nv = np.add.reduceat(yv.astype(np.int64), ov); nh = np.add.reduceat(yh.astype(np.int64), oh)
B = [(5, 10), (10, 17), (17, 30), (30, 50), (50, 75), (75, 100), (100, 150), (150, 200), (200, 400)]
print(' M bucket   | val: n, mean npos, P(npos>=2), npos/M | holdout: n, mean npos, P(npos>=2), npos/M')
for a, b in B:
    v = (Mv >= a) & (Mv < b); h = (Mh >= a) & (Mh < b)
    s = f' {a:3d}-{b:3d}  | {int(v.sum()):5d} {nv[v].mean():.3f} {np.mean(nv[v]>=2):.3f} {np.mean(nv[v]/Mv[v]):.4f}'
    if h.sum(): s += f' | {int(h.sum()):5d} {nh[h].mean():.3f} {np.mean(nh[h]>=2):.3f} {np.mean(nh[h]/Mh[h]):.4f}'
    print(s)
# simulate: sub-sample holdout feeds (M>=100) to the val M<50 size distribution under two rules, compare npos distribution with real val M<50
rng = np.random.default_rng(5); src = np.where(Mh >= 100)[0]; tgtM = Mv[Mv < 50]
m = rng.choice(tgtM, len(src)); keep_all, thin = [], []
for f_, mm in zip(src, m):
    y_ = yh[oh[f_]:oh[f_]+Mh[f_]]; npos = int(y_.sum())
    keep_all.append(min(npos, mm))                      # rule A: positives always kept (aug_cand / lb_stats HOLD-SHORT)
    k = rng.hypergeometric(npos, Mh[f_]-npos, mm)       # rule B: uniform random m-subset
    thin.append(k)
keep_all = np.array(keep_all); thin = np.array(thin); ok = thin > 0
rv = nv[Mv < 50]
print(f' real val M<50: mean npos {rv.mean():.3f}, P(>=2) {np.mean(rv>=2):.3f}, P(>=3) {np.mean(rv>=3):.3f}')
print(f' rule A (positives kept) from holdout M>=100: mean npos {keep_all.mean():.3f}, P(>=2) {np.mean(keep_all>=2):.3f}, P(>=3) {np.mean(keep_all>=3):.3f}')
print(f' rule B (uniform subset, feeds with no positive dropped; {ok.mean():.3f} survive): mean npos {thin[ok].mean():.3f}, P(>=2) {np.mean(thin[ok]>=2):.3f}, P(>=3) {np.mean(thin[ok]>=3):.3f}')
