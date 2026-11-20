"""Feed-level distributions (time, M, H) and user overlap across splits."""
from common import *
pl.Config.set_tbl_rows(40)
import json
res = {}
F = {s: feeds(s) for s in SPLITS}
for s, f in F.items():
    print(f'=== {s}: feeds={len(f)} users={f["user_id"].n_unique()}')
    bd = f.group_by('date').agg(pl.len().alias('feeds'), pl.col('M').mean().alias('M_mean'), pl.col('H').mean().alias('H_mean'),
                                pl.col('user_id').n_unique().alias('users')).sort('date')
    bd = bd.with_columns(pl.col('date').dt.strftime('%a').alias('dow'))
    print(bd)
    q = np.quantile(f['M'].to_numpy(), [0, .1, .25, .5, .75, .9, .99, 1])
    qh = np.quantile(f['H'].to_numpy(), [0, .1, .25, .5, .75, .9, .99, 1])
    print('M q', q, 'mean', f['M'].mean()); print('H q', qh, 'mean', f['H'].mean(), 'H==256', (f['H'] == 256).mean(), 'H==0', (f['H'] == 0).mean())
    hr = f.group_by('hour').agg(pl.len().alias('n')).sort('hour')
    hrs = hr['n'].to_numpy() / len(f)
    res[s] = dict(hour_share=hrs.tolist(), Mq=q.tolist(), Hq=qh.tolist(), M_mean=float(f['M'].mean()), H_mean=float(f['H'].mean()),
                  H256=float((f['H'] == 256).mean()), H0=float((f['H'] == 0).mean()))
# hour shares
print('hour share (MSK) train/val/test:')
for h in range(24):
    print(h, ' '.join(f'{res[s]["hour_share"][h]*100:5.2f}' for s in SPLITS))
print('JS(hour) train-val %.4f train-test %.4f val-test %.4f' % (
    js_div(res['train']['hour_share'], res['val']['hour_share']), js_div(res['train']['hour_share'], res['test']['hour_share']),
    js_div(res['val']['hour_share'], res['test']['hour_share'])))
# M histogram JS (bins of 10)
bins = np.arange(0, 310, 10)
Mh = {s: np.histogram(F[s]['M'].to_numpy(), bins)[0] for s in SPLITS}
Hb = np.array([0, 1, 16, 32, 64, 128, 192, 255, 256, 257])
Hh = {s: np.histogram(F[s]['H'].to_numpy(), Hb)[0] for s in SPLITS}
for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
    print(f'JS(M) {a}-{b} {js_div(Mh[a], Mh[b]):.4f}  JS(H) {js_div(Hh[a], Hh[b]):.4f}')
print('H bins', Hb); [print(s, np.round(Hh[s] / Hh[s].sum() * 100, 2)) for s in SPLITS]

# ---------------- users
U = {s: F[s]['user_id'].unique().to_numpy() for s in SPLITS}
tr_set = U['train']
for s in ['val', 'test']:
    inn = np.isin(U[s], tr_set)
    f = F[s]
    fe_in = np.isin(f['user_id'].to_numpy(), tr_set)
    print(f'{s}: users in train {inn.mean()*100:.2f}% ({inn.sum()}/{len(inn)}); feeds whose user in train {fe_in.mean()*100:.2f}%')
inn = np.isin(U['test'], U['val'])
print(f'test users in val {inn.mean()*100:.2f}% ({inn.sum()}); test feeds whose user in val {np.isin(F["test"]["user_id"].to_numpy(), U["val"]).mean()*100:.2f}%')
inn2 = np.isin(U['test'], np.union1d(U['val'], U['train']))
print(f'test users in train|val {inn2.mean()*100:.2f}%')
# feeds per user
for s in SPLITS:
    c = F[s].group_by('user_id').agg(pl.len().alias('n'))['n'].to_numpy()
    vals, cnt = np.unique(np.minimum(c, 10), return_counts=True)
    print(f'{s} feeds/user: mean {c.mean():.3f} max {c.max()} dist(capped10):', dict(zip(vals.tolist(), (cnt / cnt.sum()).round(4).tolist())))
    print(f'   share of feeds belonging to multi-feed users: {(c[c>1].sum()/c.sum())*100:.2f}%')
# for test (and val) users: train feeds / candidates / positives
tr = F['train'].select('user_id', 'M').group_by('user_id').agg(pl.len().alias('tr_feeds'), pl.col('M').sum().alias('tr_cands'))
# train positives per user
tgt = cand('train', 'target')
fp = np.add.reduceat(np.asarray(tgt, dtype=np.int64), F['train']['cand_off'].to_numpy())
trp = pl.DataFrame({'user_id': F['train']['user_id'], 'pos': fp}).group_by('user_id').agg(pl.col('pos').sum().alias('tr_pos'))
tr = tr.join(trp, on='user_id')
for s in ['val', 'test']:
    j = F[s].select('user_id').join(tr, on='user_id', how='left').fill_null(0)
    for c in ['tr_feeds', 'tr_cands', 'tr_pos']:
        x = j[c].to_numpy()
        print(f'{s} feeds: {c} quantiles(0,.1,.25,.5,.75,.9,.99)={np.quantile(x,[0,.1,.25,.5,.75,.9,.99]).round(1).tolist()} mean={x.mean():.1f} zero={np.mean(x==0)*100:.2f}%')
    x = j['tr_feeds'].to_numpy()
    print(f'   {s} feeds: tr_feeds buckets 0:{np.mean(x==0):.3f} 1:{np.mean(x==1):.3f} 2-4:{np.mean((x>=2)&(x<=4)):.3f} 5-9:{np.mean((x>=5)&(x<=9)):.3f} 10+:{np.mean(x>=10):.3f}')
# time gaps between consecutive feeds of the same user within val/test
for s in ['train', 'val', 'test']:
    f = F[s].sort(['user_id', 'timestamp', 'feed_id'])
    g = f.with_columns((pl.col('timestamp') - pl.col('timestamp').shift(1).over('user_id')).alias('gap')).filter(pl.col('gap').is_not_null())
    gap = g['gap'].to_numpy()
    print(f'{s}: consecutive same-user gaps n={len(gap)}; quantiles sec (0,.1,.25,.5,.75,.9)={np.quantile(gap,[0,.1,.25,.5,.75,.9]).round(0).tolist()}; '
          f'gap=0: {np.mean(gap==0)*100:.1f}%  <=10min {np.mean(gap<=600)*100:.1f}%  <=1h {np.mean(gap<=3600)*100:.1f}%  <=1d {np.mean(gap<=86400)*100:.1f}%')
json.dump(res, open(f'{OUT}/s01.json', 'w'))

print("\n==== user_id==0 (missing user id) ====")
for s in SPLITS:
    f=F[s]; z=(f['user_id']==0)
    print(f"{s}: user0 feeds {z.sum()} ({z.mean()*100:.2f}%), M mean {f.filter(z)['M'].mean():.1f} vs {f.filter(~z)['M'].mean():.1f}, H mean {f.filter(z)['H'].mean():.1f} vs {f.filter(~z)['H'].mean():.1f}, H==0 {(f.filter(z)['H']==0).mean()*100:.1f}%")
Unz={s:U[s][U[s]!=0] for s in SPLITS}
for s in ['val','test']:
    print(f"{s} (excluding user0): users in train {np.isin(Unz[s],Unz['train']).mean()*100:.2f}%  feeds of users in train {np.isin(F[s].filter(pl.col('user_id')!=0)['user_id'].to_numpy(),Unz['train']).mean()*100:.2f}%")
print(f"test users (excl 0) in val: {np.isin(Unz['test'],Unz['val']).sum()} = {np.isin(Unz['test'],Unz['val']).mean()*100:.2f}%")
for s in SPLITS:
    c = F[s].filter(pl.col('user_id')!=0).group_by('user_id').agg(pl.len().alias('n'))['n'].to_numpy()
    print(f'{s} excl user0 feeds/user mean {c.mean():.4f}; users with >1 feed {np.sum(c>1)} ({np.mean(c>1)*100:.2f}%), feeds in multi-feed users {(c[c>1].sum()/c.sum())*100:.2f}%')
    f = F[s].filter(pl.col('user_id')!=0).sort(['user_id', 'timestamp', 'feed_id'])
    g = f.with_columns((pl.col('timestamp') - pl.col('timestamp').shift(1).over('user_id')).alias('gap')).filter(pl.col('gap').is_not_null())
    gap = g['gap'].to_numpy()
    print(f'   gaps n={len(gap)} q(0,.1,.25,.5,.75,.9)={np.quantile(gap,[0,.1,.25,.5,.75,.9]).round(0).tolist()} gap=0 {np.mean(gap==0)*100:.1f}% <=10min {np.mean(gap<=600)*100:.1f}% <=1h {np.mean(gap<=3600)*100:.1f}% <=1d {np.mean(gap<=86400)*100:.1f}%')
