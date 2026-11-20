"""Where are the pos gaps (removed already-viewed items)? Compare likely-random vs likely-production feeds; full-train gap profile."""
from common import *
import json
nb.set_num_threads(40)

@nb.njit(parallel=True, cache=True)
def gap_stats(pos, off, M):
    n = len(M)
    ngap = np.zeros(n); relgap = np.full(n, np.nan); gap_top10 = np.zeros(n)
    for f in nb.prange(n):
        o = off[f]; m = M[f]
        mx = 0
        for j in range(o, o + m):
            if pos[j] > mx:
                mx = pos[j]
        present = np.zeros(mx + 1, np.bool_)
        for j in range(o, o + m):
            present[pos[j]] = True
        s = 0.0; c = 0; t10 = 0
        for p in range(mx + 1):
            if not present[p]:
                c += 1; s += p / max(mx, 1)
                if p < 10:
                    t10 += 1
        ngap[f] = c; gap_top10[f] = t10
        if c > 0:
            relgap[f] = s / c
    return ngap, relgap, gap_top10

F = feeds('train'); off = F['cand_off'].to_numpy(); M = F['M'].to_numpy()
pos = np.asarray(cand('train', 'pos')).astype(np.int64)
ngap, relgap, gt10 = gap_stats(pos, off, M)
res = {'mean_ngap': float(ngap.mean()), 'mean_relgap': float(np.nanmean(relgap)), 'frac_no_gap': float((ngap == 0).mean()),
       'gap_share_of_maxpos': float((ngap / (ngap + M)).mean())}
# missing-position histogram over absolute pos (full train)
@nb.njit(cache=True)
def miss_hist(pos, off, M):
    h = np.zeros(300); pres = np.zeros(300)
    for f in range(len(M)):
        o = off[f]; m = M[f]; mx = 0
        for j in range(o, o + m):
            if pos[j] > mx: mx = pos[j]
        p = np.zeros(mx + 1, np.bool_)
        for j in range(o, o + m): p[pos[j]] = True
        for q in range(mx + 1):
            if p[q]: pres[q] += 1
            else: h[q] += 1
    return h, pres
h, pres = miss_hist(pos, off, M)
res['missing_rate_by_pos'] = (h / (h + pres)).tolist()
print(res['mean_ngap'], res['frac_no_gap'], res['gap_share_of_maxpos'], 'missing rate pos0..9', np.round(h[:10] / (h[:10] + pres[:10]), 3), 'pos 50,100,200', np.round((h / (h + pres))[[50, 100, 200]], 3))
d = np.load(f'{OUT}/04b_detect.npz'); sel = d['sel']; post = d['post']; Ms = d['M']
g = (Ms >= 50) & ~np.isnan(post)
for nm, m in [('random_like', g & (post > 0.7)), ('prod_like', g & (post < 0.05))]:
    s = sel[m]
    r = dict(n=int(m.sum()), ngap=float(ngap[s].mean()), relgap=float(np.nanmean(relgap[s])), gap_top10=float(gt10[s].mean()), M=float(M[s].mean()))
    res[nm] = r; print(nm, r)
json.dump(res, open(f'{OUT}/04e_gaps.json', 'w'))
