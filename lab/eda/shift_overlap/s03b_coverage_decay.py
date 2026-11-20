"""Item coverage decay with time gap since train, and value of train-derived item stats as a function of gap
(simulate test's larger gap on val by truncating train at earlier cutoffs)."""
from common import *
import time
t0 = time.time()
NID = 111_400_000
rng = np.random.default_rng(0)
F = {s: feeds(s) for s in SPLITS}
ftr = F['train']
tr_date = ftr['date'].cast(pl.Int32).to_numpy()
tr_M = ftr['M'].to_numpy(); tr_H = ftr['H'].to_numpy()
tr_cand_day = np.repeat(tr_date, tr_M)
tr_hist_day = np.repeat(tr_date, tr_H)
tr_ci = np.asarray(cand('train', 'item_id')); tr_hi = np.asarray(hist('train', 'item_id'))
days = np.unique(tr_date); days = days[np.bincount(tr_date - days.min())[days - days.min()] > 10000]
print('train days', [str(np.datetime64(int(d), 'D')) for d in days])
# last day an item was seen as candidate / in history in train
last_c = np.full(NID, -1, np.int32); np.maximum.at(last_c, tr_ci, tr_cand_day.astype(np.int32)) if False else None
# faster: since arrays are chronological, the last write wins with fancy assignment
last_c[tr_ci] = tr_cand_day
last_h = np.full(NID, -1, np.int32); last_h[tr_hi] = tr_hist_day
first_c = np.full(NID, -1, np.int32); first_c[tr_ci[::-1]] = tr_cand_day[::-1]
print('built last/first seen', round(time.time() - t0), flush=True)

print('\n==== coverage of candidate rows by item seen in train (cand|hist) per day of val/test')
for s in ['val', 'test']:
    f = F[s]; d = np.repeat(f['date'].cast(pl.Int32).to_numpy(), f['M'].to_numpy())
    it = np.asarray(cand(s, 'item_id'))
    for dd in np.unique(d):
        m = d == dd
        if m.sum() < 10000: continue
        x = it[m]
        cov_c = np.mean(last_c[x] >= 0); cov_any = np.mean((last_c[x] >= 0) | (last_h[x] >= 0))
        lastseen = np.maximum(last_c[x], last_h[x]); seen = lastseen >= 0
        gap_last = dd - lastseen[seen]
        print(f'  {s} {np.datetime64(int(dd),"D")}: days after train end {dd - days.max():2d}; in train cand {cov_c*100:5.2f}%  in train cand|hist {cov_any*100:5.2f}%  '
              f'seen on last train day {np.mean(lastseen==days.max())*100:5.2f}%  first-seen-in-train median day offset {np.median(dd-first_c[x][first_c[x]>=0]):.0f}')

# ----- simulate cutoffs: train stats computed on train feeds with date <= cutoff; evaluate on val
it = np.asarray(cand('val', 'item_id')); t = np.asarray(cand('val', 'target')).astype(float)
M = F['val']['M'].to_numpy(); off = F['val']['cand_off'].to_numpy(); fid = np.repeat(np.arange(len(M)), M)
f1 = np.asarray(cand('val', 'f1'), dtype=np.float64)
def within_rank(x):
    x = np.asarray(x, dtype=np.float64)
    df = pl.DataFrame({'f': fid, 'x': x})
    r = df.with_columns(pl.col('x').rank('average').over('f').alias('r'))['r'].to_numpy() - 1
    return r / np.maximum(np.repeat(M, M) - 1, 1)
r_f1 = within_rank(f1)
base = ndcg_per_feed(f1, t, off, M)
base_r = np.mean([ndcg_per_feed(f1 + rng.random(len(f1)) * 1e-7, t, off, M).mean() for _ in range(5)])
print(f'\nval f1 NDCG pessimistic {base.mean():.4f}; with random tie-break {base_r:.4f}; f1 ties at min value share {np.mean(f1==f1.min())*100:.1f}%')
val_first_day = F['val']['date'].cast(pl.Int32).min()
print('\n==== train-history count (log) + f1 on val, using train data up to cutoff (gap = val day - cutoff)')
for cut in days[::-1]:
    mh = tr_hist_day <= cut; mc = tr_cand_day <= cut
    hc = np.bincount(tr_hi[mh], minlength=NID)[it].astype(float)
    cc = np.bincount(tr_ci[mc], minlength=NID)[it].astype(float)
    out = []
    for name, x in [('hist', hc), ('cand', cc)]:
        pf = ndcg_per_feed(r_f1 + 0.5 * within_rank(x), t, off, M); d = pf - base
        out.append(f'{name}: cover {np.mean(x>0)*100:5.1f}% +f1 {pf.mean():.4f} (d {d.mean():+.4f}+-{d.std()/np.sqrt(len(d)):.4f})')
    print(f'  cutoff {np.datetime64(int(cut),"D")} (gap to first val day {val_first_day-cut:2d}d): ' + ' | '.join(out), flush=True)
print('done', round(time.time() - t0))
