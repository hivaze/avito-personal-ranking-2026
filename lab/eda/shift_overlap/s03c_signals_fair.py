"""Re-evaluation of item-popularity signals on val against a FAIR baseline: f1 with random tie-break (same noise everywhere),
plus a random-signal control. Combination: within-feed pct-rank(f1) + w * pct-rank(signal)."""
from common import *
import time
t0 = time.time()
NID = 111_400_000
rng = np.random.default_rng(11)
F = {s: feeds(s) for s in SPLITS}
fv = F['val']; M = fv['M'].to_numpy(); off = fv['cand_off'].to_numpy(); fid = np.repeat(np.arange(len(M)), M)
it = np.asarray(cand('val', 'item_id')); t = np.asarray(cand('val', 'target')).astype(float)
f1 = np.asarray(cand('val', 'f1'), dtype=np.float64)
NZ = rng.random(len(t)) * 1e-9  # shared tie-break noise
def prank(x):
    r = pl.DataFrame({'f': fid, 'x': np.asarray(x, dtype=np.float64)}).with_columns(pl.col('x').rank('average').over('f').alias('r'))['r'].to_numpy() - 1
    return r / np.maximum(np.repeat(M, M) - 1, 1)
r_f1 = prank(f1)
base = ndcg_per_feed(r_f1 + NZ, t, off, M)
print(f'base: f1 random-tiebreak NDCG {base.mean():.4f} (pessimistic {ndcg(f1, t, off, M):.4f})')
def comb(x, ws=(0.1, 0.25, 0.5)):
    rx = prank(x); out = []
    for w in ws:
        d = ndcg_per_feed(r_f1 + w * rx + NZ, t, off, M) - base
        out.append((w, d.mean(), d.std() / np.sqrt(len(d))))
    return out
def alone(x):
    x = np.asarray(x, dtype=np.float64)
    return ndcg(x, t, off, M), ndcg(prank(x) + NZ, t, off, M)
def show(name, x):
    p, r = alone(x)
    c = comb(x)
    best = max(c, key=lambda z: z[1])
    cov = np.mean(np.asarray(x) != np.median(x))
    print(f'  {name:46s} alone pess {p:.4f} randtie {r:.4f} | +f1: ' + '  '.join(f'w{w}:{m:+.4f}' for w, m, s in c) + f'  (SE {best[2]:.4f})', flush=True)
C = {}
for s in SPLITS:
    C[f'{s}_cand'] = np.bincount(np.asarray(cand(s, 'item_id')), minlength=NID).astype(np.int32)
    C[f'{s}_hist'] = np.bincount(np.asarray(hist(s, 'item_id')), minlength=NID).astype(np.int32)
C['train_pos'] = np.bincount(np.asarray(cand('train', 'item_id')), weights=np.asarray(cand('train', 'target'), dtype=np.float32), minlength=NID).astype(np.int32)
print('counts built', round(time.time() - t0), flush=True)
print('\n==== signals (delta vs fair f1 baseline; SE of paired per-feed diff ~0.001)')
show('CONTROL: random signal', rng.random(len(t)))
show('val cand count (#val feeds with item)', C['val_cand'][it])
show('val hist count (all events, all val feeds)', C['val_hist'][it])
show('val hist>0 flag', (C['val_hist'][it] > 0).astype(float))
show('train cand count', C['train_cand'][it])
show('train hist count', C['train_hist'][it])
show('train pos count', C['train_pos'][it])
show('train CTR smoothed (pos+1)/(cnt+45)', (C['train_pos'][it] + 1) / (C['train_cand'][it] + 45))
show('seen in train (cand|hist) flag', ((C['train_cand'][it] + C['train_hist'][it]) > 0).astype(float))
show('log1p(train hist) - log1p(train cand)', np.log1p(C['train_hist'][it]) - np.log1p(C['train_cand'][it]))
et = np.asarray(hist('val', 'event_type')).astype(np.int16); hi = np.asarray(hist('val', 'item_id'))
show('val hist count, event_type 1', np.bincount(hi[et == 1], minlength=NID)[it])
show('val hist count, event_type 2-10', np.bincount(hi[et > 1], minlength=NID)[it])
et_tr = np.asarray(hist('train', 'event_type')).astype(np.int16); hi_tr = np.asarray(hist('train', 'item_id'))
for e in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
    show(f'train hist count, event_type {e}', np.bincount(hi_tr[et_tr == e], minlength=NID)[it])
del et_tr, hi_tr

# ---- cutoff simulation with fair baseline
print('\n==== train hist / cand count computed on train feeds up to cutoff (fair baseline), w=0.25 and 0.5')
ftr = F['train']; tr_date = ftr['date'].cast(pl.Int32).to_numpy()
cd = np.repeat(tr_date, ftr['M'].to_numpy()); hd = np.repeat(tr_date, ftr['H'].to_numpy())
tr_ci = np.asarray(cand('train', 'item_id')); tr_hi = np.asarray(hist('train', 'item_id'))
vd0 = fv['date'].cast(pl.Int32).min()
for cut in sorted(set(tr_date[np.isin(tr_date, np.unique(tr_date)[np.bincount(tr_date - tr_date.min())[np.unique(tr_date) - tr_date.min()] > 10000])]), reverse=True)[::2]:
    hc = np.bincount(tr_hi[hd <= cut], minlength=NID)[it]; cc = np.bincount(tr_ci[cd <= cut], minlength=NID)[it]
    ch = comb(hc, (0.25, 0.5)); cc_ = comb(cc, (0.25, 0.5))
    print(f'  cutoff {np.datetime64(int(cut),"D")} gap {vd0-cut:2d}d: hist cover {np.mean(hc>0)*100:5.1f}% ' + ' '.join(f'w{w}:{m:+.4f}' for w, m, s in ch) +
          f' | cand cover {np.mean(cc>0)*100:5.1f}% ' + ' '.join(f'w{w}:{m:+.4f}' for w, m, s in cc_), flush=True)
# recent-window only (last k days of train) to separate volume from recency
print('\n==== train hist count from a fixed 3-day window ending at cutoff (equal volume, varying gap), w=0.25')
days = np.unique(tr_date); days = days[np.bincount(tr_date - tr_date.min())[days - tr_date.min()] > 10000]
for j in range(len(days) - 1, 1, -3):
    win = days[j - 2:j + 1]
    m = np.isin(hd, win); hc = np.bincount(tr_hi[m], minlength=NID)[it]
    mc = np.isin(cd, win); cc = np.bincount(tr_ci[mc], minlength=NID)[it]
    ch = comb(hc, (0.25,)); cc_ = comb(cc, (0.25,))
    print(f'  window {[str(np.datetime64(int(x),"D"))[5:] for x in win]} gap {vd0-win[-1]:2d}d: hist cover {np.mean(hc>0)*100:5.1f}% d {ch[0][1]:+.4f} | cand cover {np.mean(cc>0)*100:5.1f}% d {cc_[0][1]:+.4f}', flush=True)
print('done', round(time.time() - t0))
