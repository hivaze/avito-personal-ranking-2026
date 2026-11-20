"""Single-feature NDCG (both directions, tie modes) on train (full) vs val; plus within-feed logistic slopes with/without pos control."""
from common import *
import json, time
import metrics
nb.set_num_threads(40)
res = {}
FE = ['f1', 'f2', 'f3', 'f4', 'f5']
for split in ['val', 'train']:
    F = feeds(split); off = F['cand_off'].to_numpy(); M = F['M'].to_numpy()
    t = np.asarray(cand(split, 'target'))
    r = {}
    rnd = ndcg_pf(np.zeros(len(t)), t, off, M, mode=2)
    r['random_expected'] = float(rnd.mean())
    r['constant_pessimistic'] = float(ndcg_pf(np.zeros(len(t)), t, off, M, mode=0).mean())
    r['constant_optimistic'] = float(ndcg_pf(np.zeros(len(t)), t, off, M, mode=1).mean())
    r['random_expected_at30'] = float(ndcg_pf(np.zeros(len(t)), t, off, M, k=30, mode=2).mean())
    for k in FE:
        x = np.asarray(cand(split, k)).astype(np.float64)
        for sgn, nm in [(1, k), (-1, '-' + k)]:
            r[nm] = {mn: float(ndcg_pf(sgn * x, t, off, M, mode=mi).mean()) for mi, mn in [(0, 'pess'), (2, 'expected'), (1, 'opt')]}
            r[nm]['at30_pess'] = float(ndcg_pf(sgn * x, t, off, M, k=30, mode=0).mean())
        if split == 'val':
            # paired SE vs random (expected ties)
            a = ndcg_pf(x, t, off, M, mode=2)
            r[k]['se_vs_random'] = paired(a, rnd)[1]
    res[split] = r
    print(split, json.dumps(r, indent=0), flush=True)

# AUC-style within-feed comparison: per-feed probability that a random positive outranks a random negative (ties 0.5), train vs val
@nb.njit(parallel=True, cache=True)
def feed_auc(x, t, off, M):
    n = len(M); out = np.full(n, np.nan)
    for f in nb.prange(n):
        o = off[f]; m = M[f]
        s = 0.0; c = 0
        for i in range(o, o + m):
            if t[i] == 1:
                for j in range(o, o + m):
                    if t[j] == 0:
                        c += 1
                        if x[i] > x[j]: s += 1
                        elif x[i] == x[j]: s += 0.5
        if c > 0: out[f] = s / c
    return out
res['feed_auc'] = {}
for split in ['val', 'train']:
    F = feeds(split); off = F['cand_off'].to_numpy(); M = F['M'].to_numpy()
    if split == 'train':
        sel = subsample_feeds(F, 500_000, seed=3)
        d, off, M = gather(split, sel, F, FE + ['target'])
    else:
        d = {k: np.asarray(cand(split, k)) for k in FE + ['target']}
    res['feed_auc'][split] = {k: float(np.nanmean(feed_auc(d[k].astype(np.float64), d['target'], off.astype(np.int64), M.astype(np.int64)))) for k in FE}
    print('feed AUC', split, res['feed_auc'][split])
json.dump(res, open(f'{OUT}/06_single_feature.json', 'w'), indent=1)
