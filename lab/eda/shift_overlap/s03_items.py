"""Item overlap / cold start across splits and transductive popularity signals evaluated on val (exact NDCG)."""
from common import *
import json, time
t0 = time.time()
NID = 111_400_000
rng = np.random.default_rng(0)
F = {s: feeds(s) for s in SPLITS}

def bc(a, w=None):
    return np.bincount(np.asarray(a), weights=w, minlength=NID)

# ---- count tables
C = {}
for s in SPLITS:
    C[f'{s}_cand'] = bc(cand(s, 'item_id')).astype(np.int32)
    C[f'{s}_hist'] = bc(hist(s, 'item_id')).astype(np.int32)
    print(s, 'counts done', round(time.time() - t0), flush=True)
C['train_pos'] = np.bincount(np.asarray(cand('train', 'item_id')), weights=np.asarray(cand('train', 'target'), dtype=np.float32), minlength=NID).astype(np.int32)
C['val_pos'] = np.bincount(np.asarray(cand('val', 'item_id')), weights=np.asarray(cand('val', 'target'), dtype=np.float32), minlength=NID).astype(np.int32)
print('unique items:', {k: int((v > 0).sum()) for k, v in C.items()})

# ---- overlap for val / test candidates (row-weighted and unique-item-weighted)
print('\n==== overlap of candidate items (row-weighted % | unique-item %)')
for s in ['val', 'test']:
    it = np.asarray(cand(s, 'item_id'))
    ui = np.unique(it)
    def rep(name, mask_fn):
        r = mask_fn(it).mean() * 100; u = mask_fn(ui).mean() * 100
        print(f'  {s}: {name:45s} rows {r:6.2f}%  uniq {u:6.2f}%')
    rep('in train candidates', lambda x: C['train_cand'][x] > 0)
    rep('in train history', lambda x: C['train_hist'][x] > 0)
    rep('in train cand OR hist', lambda x: (C['train_cand'][x] > 0) | (C['train_hist'][x] > 0))
    rep('in train positives (target=1 somewhere)', lambda x: C['train_pos'][x] > 0)
    rep(f'in {s} history (same split, any feed)', lambda x: C[f'{s}_hist'][x] > 0)
    rep(f'in >=2 {s} feeds as candidate', lambda x: C[f'{s}_cand'][x] >= 2)
    if s == 'test':
        rep('in val candidates', lambda x: C['val_cand'][x] > 0)
        rep('in val history', lambda x: C['val_hist'][x] > 0)
        rep('in train|val (cand|hist)', lambda x: (C['train_cand'][x] + C['train_hist'][x] + C['val_cand'][x] + C['val_hist'][x]) > 0)
        rep('COLD: not in train/val nor test history', lambda x: (C['train_cand'][x] + C['train_hist'][x] + C['val_cand'][x] + C['val_hist'][x] + C['test_hist'][x]) == 0)
    else:
        rep('COLD: not in train nor val history', lambda x: (C['train_cand'][x] + C['train_hist'][x] + C['val_hist'][x]) == 0)
    rep('not in train (cand|hist)', lambda x: (C['train_cand'][x] + C['train_hist'][x]) == 0)
# history items overlap
for s in ['val', 'test']:
    hi = np.asarray(hist(s, 'item_id'))
    print(f'  {s} history rows: in train hist {np.mean(C["train_hist"][hi]>0)*100:.2f}%  in train cand {np.mean(C["train_cand"][hi]>0)*100:.2f}%  in train any {np.mean((C["train_hist"][hi]+C["train_cand"][hi])>0)*100:.2f}%')

# ---- item frequency distributions
print('\n==== item frequency distributions (as candidate, #feeds per item)')
for s in SPLITS:
    c = C[f'{s}_cand']; c = c[c > 0]
    nf = len(F[s])
    q = np.quantile(c, [.5, .9, .99, .999])
    srt = np.sort(c)[::-1]
    top1 = srt[: max(1, len(srt) // 100)].sum() / srt.sum()
    print(f'{s}: uniq items {len(c)}, rows {c.sum()}, rows/uniq {c.sum()/len(c):.2f}, q50/90/99/99.9 = {q.tolist()}, max {c.max()} ({c.max()/nf*100:.2f}% of feeds), share of rows from top1% items {top1*100:.1f}%, items seen once {np.mean(c==1)*100:.1f}%')
for s in ['val', 'test']:
    it = np.asarray(cand(s, 'item_id')); cc = C[f'{s}_cand'][it]
    print(f'{s} candidate rows by in-split item count: =1 {np.mean(cc==1)*100:.1f}%  2-4 {np.mean((cc>=2)&(cc<=4))*100:.1f}%  5-19 {np.mean((cc>=5)&(cc<20))*100:.1f}%  20+ {np.mean(cc>=20)*100:.1f}%')

# ---- cold vs warm target rates in val
print('\n==== val target rate by item status')
it = np.asarray(cand('val', 'item_id')); t = np.asarray(cand('val', 'target')).astype(float)
M = F['val']['M'].to_numpy(); off = F['val']['cand_off'].to_numpy(); fid = np.repeat(np.arange(len(M)), M)
groups = {
    'in train cand': C['train_cand'][it] > 0,
    'not in train cand': C['train_cand'][it] == 0,
    'in train hist': C['train_hist'][it] > 0,
    'not in train cand nor hist': (C['train_cand'][it] + C['train_hist'][it]) == 0,
    'val cand count==1': C['val_cand'][it] == 1,
    'val cand count>=5': C['val_cand'][it] >= 5,
    'in val hist (other users)': C['val_hist'][it] > 0,
    'not in val hist': C['val_hist'][it] == 0,
}
for k, m in groups.items():
    print(f'  {k:30s} share {m.mean()*100:6.2f}%  pos rate {t[m].mean()*100:.3f}%')

# ---- signals evaluated on val (exact NDCG); ties: pessimistic (as metric) and with random tie-break (mean of 5)
def within_rank(x):
    """percentile rank within feed (0..1), average ties."""
    x = np.asarray(x, dtype=np.float64)
    order = np.lexsort((x, fid))
    r = np.empty(len(x)); r[order] = np.arange(len(x)) - np.repeat(off, M)
    # average ties within feed
    df = pl.DataFrame({'f': fid, 'x': x, 'r': r})
    r = df.with_columns(pl.col('r').mean().over(['f', 'x']))['r'].to_numpy()
    return r / np.maximum(np.repeat(M, M) - 1, 1)

def ev(x):
    x = np.asarray(x, dtype=np.float64)
    pess = ndcg(x, t, off, M)
    rnd = np.mean([ndcg(x + rng.random(len(x)) * 1e-9 * (np.abs(x).max() + 1) if False else x + rng.random(len(x)) * 1e-6 * max(1e-12, np.min(np.diff(np.unique(x))) if len(np.unique(x)) > 1 else 1), t, off, M) for _ in range(5)])
    return pess, rnd

f1 = np.asarray(cand('val', 'f1'), dtype=np.float64)
base_f1 = ndcg_per_feed(f1, t, off, M)
rand = np.mean([ndcg(rng.random(len(t)), t, off, M) for _ in range(5)])
print(f'\n==== single-signal NDCG on val (random {rand:.4f}, f1 {base_f1.mean():.4f})')
r_f1 = within_rank(f1)
sig = {}
sig['val cand count (#val feeds w/ item)'] = C['val_cand'][it].astype(float)
sig['val hist count (any event, all val feeds)'] = C['val_hist'][it].astype(float)
sig['train cand count'] = C['train_cand'][it].astype(float)
sig['train hist count'] = C['train_hist'][it].astype(float)
sig['train pos count'] = C['train_pos'][it].astype(float)
a, b = 1.0, 1.0 / 0.022
sig['train CTR smoothed (pos+1)/(cnt+45)'] = (C['train_pos'][it] + a) / (C['train_cand'][it] + b)
sig['val hist / val cand ratio (log1p h - log1p c)'] = np.log1p(C['val_hist'][it]) - np.log1p(C['val_cand'][it])
sig['train hist / train cand (log1p h - log1p c)'] = np.log1p(C['train_hist'][it]) - np.log1p(C['train_cand'][it])
results = {}
for k, x in sig.items():
    p, r = ev(x)
    comb = r_f1 + 0.5 * within_rank(x)
    pf = ndcg_per_feed(comb, t, off, M)
    d = pf - base_f1
    results[k] = (p, r, pf.mean(), d.mean(), d.std() / np.sqrt(len(d)))
    print(f'  {k:50s} pess {p:.4f} randtie {r:.4f} | f1rank+0.5*rank(sig) {pf.mean():.4f} (paired d vs f1 {d.mean():+.4f} +- {d.std()/np.sqrt(len(d)):.4f})')
# event-type-specific val history counts
et = np.asarray(hist('val', 'event_type')).astype(np.int16); hi = np.asarray(hist('val', 'item_id'))
print('  per event_type: val-history count of candidate item (NDCG pess / randtie / combined with f1)')
for e in range(1, 11):
    ce = np.bincount(hi[et == e], minlength=NID)[it].astype(float)
    p, r = ev(ce)
    pf = ndcg_per_feed(r_f1 + 0.5 * within_rank(ce), t, off, M); d = pf - base_f1
    print(f'    et={e:2d} cover {np.mean(ce>0)*100:5.2f}%  pess {p:.4f} randtie {r:.4f} | +f1 {pf.mean():.4f} (d {d.mean():+.4f} +- {d.std()/np.sqrt(len(d)):.4f})  pos-rate cover {t[ce>0].mean()*100:.2f}% vs {t[ce==0].mean()*100:.2f}%')
# event-type-specific train history counts
et_tr = np.asarray(hist('train', 'event_type')).astype(np.int16); hi_tr = np.asarray(hist('train', 'item_id'))
print('  per event_type: TRAIN-history count of candidate item')
for e in range(1, 11):
    ce = np.bincount(hi_tr[et_tr == e], minlength=NID)[it].astype(float)
    p, r = ev(ce)
    pf = ndcg_per_feed(r_f1 + 0.5 * within_rank(ce), t, off, M); d = pf - base_f1
    print(f'    et={e:2d} cover {np.mean(ce>0)*100:5.2f}%  pess {p:.4f} randtie {r:.4f} | +f1 {pf.mean():.4f} (d {d.mean():+.4f} +- {d.std()/np.sqrt(len(d)):.4f})')
del et_tr, hi_tr

# ---- leave-one-feed-out in-split candidate count is identical to count-1 -> same ranking; also normalized per 10k feeds
print('\n==== in-split popularity scale: mean in-split cand count per candidate row, per 10k feeds')
for s in SPLITS:
    itx = np.asarray(cand(s, 'item_id')) if s != 'train' else np.asarray(cand(s, 'item_id')[::50])
    cc = C[f'{s}_cand'][itx]; hc = C[f'{s}_hist'][itx]
    nf = len(F[s])
    print(f'  {s}: mean cand count {cc.mean():.2f} (per10k feeds {cc.mean()/nf*1e4:.3f}); mean hist count {hc.mean():.2f} (per10k {hc.mean()/nf*1e4:.3f}); frac in-split hist>0 {np.mean(hc>0)*100:.1f}%')
json.dump({k: list(v) for k, v in results.items()}, open(f'{OUT}/s03.json', 'w'))
print('done', round(time.time() - t0))
