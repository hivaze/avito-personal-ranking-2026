# Paired analysis of the loss-zoo proxy runs: single-model deltas vs the 'ce' baseline (pooled over seeds) with slices,
# and loss-diversity ensembles: z-mean(ce_si, X_sj) vs z-mean(ce_si, ce_sj), i != j (seed diversity in both arms).
import sys, glob, re, numpy as np
sys.path.insert(0, '<repo>/solvers/solvers')
from proxy_lib import *
TAG = sys.argv[1] if len(sys.argv) > 1 else 'A'
BASE = sys.argv[2] if len(sys.argv) > 2 else 'ce'
OUT = '<repo>/solvers/out/solvers'
runs = {}
for fn in glob.glob(f'{OUT}/zoo_{TAG}_*_s*.npz'):
    k, s = re.match(rf'.*zoo_{TAG}_(.+)_s(\d+)\.npz', fn).groups(); runs.setdefault(k, {})[int(s)] = fn
fv, Mv, ov = feed_meta('val'); yv = np.load(FLAT / 'val/cand_target.npy').astype(np.float64)
fid = np.repeat(np.arange(len(Mv)), Mv); K = np.bincount(fid, weights=yv)
dayv = day_of(fv['timestamp'].to_numpy()); tue = dayv == dayv.min()
sl = {'M<50': Mv < 50, 'M>=50': Mv >= 50, 'Tue': tue, 'Thu': ~tue, 'K=1': K == 1, 'K>=2': K >= 2}
def zf(s):  # per-feed z-score
    mu = np.bincount(fid, weights=s) / Mv; d = s - mu[fid]; sd = np.sqrt(np.bincount(fid, weights=d * d) / Mv) + 1e-9
    return d / sd[fid]
base = runs[BASE]; seeds = sorted(base)
L = {k: {s: np.load(fn) for s, fn in v.items()} for k, v in runs.items()}
print(f'tag {TAG}  base {BASE}  seeds {seeds}')
print('%-12s %8s %8s | %-18s | %-18s | %s' % ('kind', 'val', 'hold', 'dval (SE) z', 'dhold (SE)', '  '.join(sl)))
rows = []
for k in sorted(L):
    ss = [s for s in seeds if s in L[k]]
    if not ss: continue
    v = np.mean([L[k][s]['val'] for s in ss], 0); h = np.mean([L[k][s]['hold'] for s in ss], 0)
    dv = np.mean([L[k][s]['val'] - L[BASE][s]['val'] for s in ss], 0); dh = np.mean([L[k][s]['hold'] - L[BASE][s]['hold'] for s in ss], 0)
    sev = dv.std() / np.sqrt(len(dv)); seh = dh.std() / np.sqrt(len(dh))
    sls = '  '.join('%+.4f' % dv[m].mean() for m in sl.values())
    print('%-12s %.5f %.5f | %+.5f (%.5f) %+5.1f | %+.5f (%.5f) | %s  [n_seeds %d]' % (k, v.mean(), h.mean(), dv.mean(), sev, dv.mean() / max(sev, 1e-12), dh.mean(), seh, sls, len(ss)))
# per-seed val deltas
print('\nper-seed val deltas vs', BASE)
for k in sorted(L):
    print('%-12s' % k, ' '.join('%+.5f' % (L[k][s]['val'].mean() - L[BASE][s]['val'].mean()) for s in seeds if s in L[k]))
# diversity ensembles (needs saved scores)
if 'sval' in L[BASE][seeds[0]].files and len(seeds) >= 2:
    print('\nloss-diversity ensembles: z-mean(base_si, X_sj) minus z-mean(base_si, base_sj), pairs (i,j) cyclic, i != j')
    Z = {k: {s: zf(L[k][s]['sval'].astype(np.float64)) for s in L[k]} for k in L}
    pairs = [(seeds[i], seeds[(i + 1) % len(seeds)]) for i in range(len(seeds))] if len(seeds) > 2 else [(seeds[0], seeds[1])]
    ref = np.mean([ndcg_per_feed(Z[BASE][i] + Z[BASE][j], yv, ov, Mv) for i, j in pairs], 0)
    single = np.mean([L[BASE][s]['val'] for s in seeds], 0)
    print('base single %.5f  base seed-pair ensemble %.5f (+%.5f)' % (single.mean(), ref.mean(), ref.mean() - single.mean()))
    for k in sorted(Z):
        if k == BASE: continue
        pp = [(i, j) for i, j in pairs if j in Z[k]]
        if not pp: continue
        e = np.mean([ndcg_per_feed(Z[BASE][i] + Z[k][j], yv, ov, Mv) for i, j in pp], 0)
        d = e - ref
        # rank correlation proxy: mean within-feed Pearson of z-scores between base and X (same seed)
        s0 = [s for s in seeds if s in Z[k]][0]
        corr = np.bincount(fid, weights=Z[BASE][s0] * Z[k][s0]) / Mv
        print('%-12s ens %.5f  d vs seed-ens %+.5f (SE %.5f, z %+.1f)  mean within-feed corr(base,X) %.3f' % (k, e.mean(), d.mean(), d.std() / np.sqrt(len(d)), d.mean() / (d.std() / np.sqrt(len(d))), corr.mean()))
