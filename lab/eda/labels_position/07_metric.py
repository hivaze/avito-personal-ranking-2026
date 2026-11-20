"""Metric behaviour on val: full vs @30, dependence on M, AUC->NDCG mapping, ties, public/private composition of test."""
from common import *
import json
from scipy.stats import norm
nb.set_num_threads(40)
res = {}
F = feeds('val'); off = F['cand_off'].to_numpy(); M = F['M'].to_numpy()
t = np.asarray(cand('val', 't'.join(['', 'arget']))) if False else np.asarray(cand('val', 'target'))
npos = np.add.reduceat(t, off, dtype=np.int64)
f1 = np.asarray(cand('val', 'f1')).astype(np.float64); f5 = np.asarray(cand('val', 'f5')).astype(np.float64)
f3 = np.asarray(cand('val', 'f3')).astype(np.float64)
disc_cum = np.cumsum(1 / np.log2(np.arange(2, 400)))
def exp_random(M, k):
    return k * disc_cum[M - 1] / M / disc_cum[k - 1]
rnd = exp_random(M, npos)
res['random_closed_form'] = float(rnd.mean())
scorers = {'random': np.random.default_rng(0).random(len(t)), 'f1': f1, 'f1+1e-3*f5': f1 + 1e-3 * f5, 'f1_rank+f3_rank+f5_rank': None}
# simple rank-sum combo within feed
fid = np.repeat(np.arange(len(M)), M)
df = pl.DataFrame({'fid': fid, 'f1': f1, 'f3': f3, 'f5': f5}).with_columns([pl.col(c).rank('average').over('fid').alias(c + 'r') for c in ['f1', 'f3', 'f5']])
scorers['f1_rank+f3_rank+f5_rank'] = (df['f1r'] + df['f3r'] + df['f5r']).to_numpy() + 1e-6 * f5
res['full_vs_at'] = {}
per = {}
for nm, s in scorers.items():
    full = ndcg_pf(s, t, off, M); per[nm] = full
    res['full_vs_at'][nm] = {'full': float(full.mean()), **{f'@{k}': float(ndcg_pf(s, t, off, M, k=k).mean()) for k in [10, 30, 100]}}
print(res['full_vs_at'])
# by M bin
edges = [5, 10, 20, 30, 50, 75, 100, 150, 200, 250, 300]
Mb = np.digitize(M, edges[1:-1])
tab = []
for b in range(len(edges) - 1):
    m = Mb == b
    row = dict(M=f'[{edges[b]},{edges[b + 1]})', share=float(m.mean()), npos=float(npos[m].mean()), random=float(rnd[m].mean()),
               headroom_share=float((1 - rnd[m]).sum() / (1 - rnd).sum()), f1=float(per['f1'][m].mean()), combo=float(per['f1_rank+f3_rank+f5_rank'][m].mean()))
    row['f1_gain_share'] = float((per['f1'][m] - rnd[m]).sum() / (per['f1'] - rnd).sum())
    row['combo_gain_share'] = float((per['f1_rank+f3_rank+f5_rank'][m] - rnd[m]).sum() / (per['f1_rank+f3_rank+f5_rank'] - rnd).sum())
    tab.append(row)
res['by_M'] = tab
for r in tab: print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
# single-positive feeds: NDCG as function of rank
res['single_pos_ndcg_at_rank'] = {r: float(1 / np.log2(r + 1)) for r in [1, 2, 3, 5, 10, 20, 30, 50, 100, 200, 300]}
# AUC -> NDCG mapping via simulation on val's (M, npos) structure
rng = np.random.default_rng(1)
sim = []
for d in [0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]:
    vals = []
    for rep in range(3):
        s = rng.standard_normal(len(t)) + d * t
        vals.append(ndcg_pf(s, t, off, M).mean())
    sim.append(dict(d=d, auc=float(norm.cdf(d / np.sqrt(2))), ndcg=float(np.mean(vals))))
res['auc_to_ndcg_sim'] = sim
print(sim)
# measured per-feed AUC of f1 for reference
# ties: constant & discrete features
res['ties'] = {}
for nm, s in [('constant', np.zeros(len(t))), ('f3', f3), ('f1', f1), ('f3+1e-6*f1 (tie-break)', f3 + 1e-6 * f1)]:
    res['ties'][nm] = {mn: float(ndcg_pf(s, t, off, M, mode=mi).mean()) for mi, mn in [(0, 'pess'), (2, 'expected'), (1, 'opt')]}
print(res['ties'])
# float32 rounding of a smooth score: ties created
s64 = per['f1_rank+f3_rank+f5_rank']
# day stability
day = F.with_columns(pl.from_epoch(pl.col('timestamp') + 3 * 3600, time_unit='s').dt.date().cast(pl.Utf8).alias('d'))['d'].to_numpy()
res['by_day'] = {dd: dict(n=int((day == dd).sum()), random=float(rnd[day == dd].mean()), f1=float(per['f1'][day == dd].mean())) for dd in np.unique(day)}
print(res['by_day'])
# public/private composition of test: first 30% of feeds by timestamp
T = feeds('test').sort(['timestamp', 'feed_id'])
Mt = T['M'].to_numpy(); n = len(Mt); cut = int(0.3 * n)
# expected random NDCG using val's npos|M-bin empirical distribution
Mbt = np.digitize(Mt, edges[1:-1])
er = np.zeros(n)
for b in range(len(edges) - 1):
    m = Mbt == b
    pool = np.where(Mb == b)[0]
    pick = rng.choice(pool, size=m.sum())
    k = np.minimum(npos[pick], Mt[m])
    er[m] = exp_random(Mt[m], k)
res['test_public_private'] = dict(public=dict(n=cut, M_mean=float(Mt[:cut].mean()), M_median=float(np.median(Mt[:cut])), exp_random=float(er[:cut].mean()),
                                              ts_end=int(T['timestamp'][cut - 1])),
                                  private=dict(n=n - cut, M_mean=float(Mt[cut:].mean()), M_median=float(np.median(Mt[cut:])), exp_random=float(er[cut:].mean())),
                                  val=dict(M_mean=float(M.mean()), exp_random=float(rnd.mean())))
print(res['test_public_private'])
json.dump(res, open(f'{OUT}/07_metric.json', 'w'), indent=1)
