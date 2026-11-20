"""(a) Day-of-week stability in train (f1 NDCG / positive rate per day); (b) upper bound of the later-feed-history leak in val/test."""
from common import *
from featlib import subset
rng = np.random.default_rng(3)
ftr = feeds('train')
dd = ftr['date'].cast(pl.Int32).to_numpy()
def offs(Mx): return np.r_[0, np.cumsum(Mx)[:-1]]
print('==== train per-day (20k random feeds/day): f1 NDCG (random tiebreak), random NDCG, pos/feed, M')
for d in np.unique(dd):
    idx = np.where(dd == d)[0]
    if len(idx) < 10000: continue
    T = subset('train', rng.choice(idx, 20000, replace=False))
    t = T['C']['target'].astype(float); o = offs(T['M'])
    a = ndcg(T['C']['f1'] + rng.random(len(t)) * 1e-7, t, o, T['M']); r = ndcg(rng.random(len(t)), t, o, T['M'])
    print(f'  {np.datetime64(int(d),"D")} {str(np.datetime64(int(d),"D").astype("datetime64[D]").item().strftime("%a"))}: f1 {a:.4f} random {r:.4f} lift {a-r:+.4f}  pos/feed {np.bincount(T["cf"], weights=t).mean():.3f}  M {T["M"].mean():.1f}', flush=True)

print('\n==== later-feed history leak: upper bound on val (oracle uses only data available in the split files)')
F = {s: feeds(s) for s in ['val', 'test']}
for s in ['val', 'test']:
    f = F[s].filter(pl.col('user_id') != 0).sort(['user_id', 'timestamp', 'feed_id'])
    f = f.with_columns(pl.col('fidx').shift(-1).over('user_id').alias('nxt'), (pl.col('timestamp').shift(-1).over('user_id') - pl.col('timestamp')).alias('gap'))
    p = f.filter(pl.col('nxt').is_not_null() & (pl.col('gap') > 0))
    D = subset(s)
    # candidate flag: item appears in the history of the user's next feed
    hist_items = {}
    Hf = D['hf']; Hi = D['Hh']['item_id']
    nxt = dict(zip(p['fidx'].to_list(), p['nxt'].to_list()))
    flag = np.zeros(len(D['cf']), bool)
    hoff = np.r_[0, np.cumsum(D['H'])]
    coff = np.r_[0, np.cumsum(D['M'])]
    for a, b in nxt.items():
        hs = set(Hi[hoff[b]:hoff[b + 1]].tolist())
        ci = D['C']['item_id'][coff[a]:coff[a + 1]]
        flag[coff[a]:coff[a + 1]] = np.fromiter((x in hs for x in ci), bool, len(ci))
    ts = D['f']['timestamp'].to_numpy(); o = np.argsort(ts, kind='stable'); pub = np.zeros(len(ts), bool); pub[o[:int(round(0.3 * len(ts)))]] = True
    aff = np.array(list(nxt.keys()), dtype=int)
    print(f'{s}: affected feeds {len(aff)} ({len(aff)/len(ts)*100:.2f}%), of which public-period {pub[aff].sum()} / private-period {(~pub[aff]).sum()}; flagged candidates {flag.sum()}')
    if s == 'val':
        t = D['C']['target'].astype(float); f1 = D['C']['f1'].astype(float)
        nz = rng.random(len(t)) * 1e-7  # same tie-break noise for both scorings (paired)
        base = ndcg_per_feed(f1 + nz, t, offs(D['M']), D['M'])
        leak = ndcg_per_feed(f1 + 10 * flag + nz, t, offs(D['M']), D['M'])
        print(f'   val f1 NDCG {base.mean():.4f} -> with leak flag {leak.mean():.4f} (delta {leak.mean()-base.mean():+.4f}); on affected feeds {base[aff].mean():.3f} -> {leak[aff].mean():.3f}')
        exp_gain_test = (leak[aff].mean() - base[aff].mean())
        print(f'   implied test gain if same per-feed effect: {exp_gain_test*len(nxt)/len(ts):+.4f} (val rate) ')
    else:
        print(f'   implied test-all gain at val per-feed effect {exp_gain_test:.3f}: {exp_gain_test*len(aff)/len(ts):+.4f}; public-only {exp_gain_test*pub[aff].sum()/pub.sum():+.4f}; private-only {exp_gain_test*(~pub[aff]).sum()/(~pub).sum():+.4f}')
