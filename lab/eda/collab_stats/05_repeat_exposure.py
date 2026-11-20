import numpy as np, polars as pl, time
from common import *
t0=time.time()
V=Val('val'); RP=V.rand_pf()
rel_items=np.load(f'{OUT}/rel_items.npy'); rel_users=np.load(f'{OUT}/rel_users.npy')
lut=np.full(NITEM,-1,np.int32); lut[rel_items]=np.arange(len(rel_items),dtype=np.int32)
ulut=np.full(rel_users.max()+1,-1,np.int32); ulut[rel_users]=np.arange(len(rel_users),dtype=np.int32)
P=pl.read_parquet(f'{OUT}/train_user_item_pairs.parquet')
agg=P.group_by(['u','i']).agg(pl.len().alias('n_prev'), pl.col('t').sum().alias('pos_prev'), pl.col('ts').max().alias('last_ts'), pl.col('pos').min().alias('min_pos'))
vu=np.repeat(V.f['user_id'].to_numpy(),V.M); vi=np.asarray(arr('val','cand_item_id'))
q=pl.DataFrame({'u':ulut[vu],'i':lut[vi],'row':np.arange(len(vi))})
j=q.join(agg,on=['u','i'],how='left').sort('row')
n_prev=j['n_prev'].fill_null(0).to_numpy().astype(float); pos_prev=j['pos_prev'].fill_null(0).to_numpy().astype(float)
t=V.target
print('val cand rows with prior train impression of same (user,item):', (n_prev>0).mean(), 'feeds with any:', (np.bincount(V.fid,weights=(n_prev>0))>0).mean())
for nm,m in [('never shown',n_prev==0),('shown before, never pos',(n_prev>0)&(pos_prev==0)),('shown before & pos before',pos_prev>0)]:
    print(f'  {nm:28s} rows={m.sum():8d} target rate={t[m].mean():.4f}')
for k in [1,2,3,5,10]:
    m=(n_prev>=k)&(pos_prev==0); print(f'  shown>={k} never pos: rows={m.sum()} rate={t[m].mean():.4f}')
# within feeds where user is in train only
U=np.load(f'{OUT}/user_cat_aggs.npz'); intr=U['nfeeds'][ulut[vu]]>0
fin = np.bincount(V.fid, weights=(n_prev>0).astype(float))>0
print('feeds with user in train', (np.bincount(V.fid,weights=intr)>0).mean(), ' feeds with >=1 previously-shown candidate', fin.mean())
for nm,s in [('n_prev',n_prev),('-n_prev',-n_prev),('pos_prev',pos_prev),('pos_prev - 0.1*n_neg', pos_prev-0.1*(n_prev-pos_prev))]:
    a,d,se=V.cmp(s,RP); pf=V.pf_rt(s)
    print(f'{nm:24s} ndcg all={a:.4f} lift={d:+.4f}+-{se:.4f} | on feeds w/ >=1 prev-shown cand: {pf[fin].mean():.4f} vs rand {RP[fin].mean():.4f}')
# in val itself: same user seen same item in earlier val feed?
f=V.f.with_columns(pl.col('M'))
ts=np.repeat(V.f['timestamp'].to_numpy(),V.M)
d=pl.DataFrame({'u':vu,'i':vi,'ts':ts,'t':t,'row':np.arange(len(vi))})
d2=d.join(d.select(['u','i',pl.col('ts').alias('ts2'),pl.col('t').alias('t2')]),on=['u','i']).filter(pl.col('ts2')<pl.col('ts'))
print('val rows with earlier val impression of same (u,i):', d2['row'].n_unique(), 'target rate', d2.unique('row')['t'].mean())
# history leakage check: positives of earlier feed of same user appearing in later feed history (train sample)
np.save(f'{OUT}/val_repeat.npy', np.stack([n_prev,pos_prev]))
tlog(t0,'done')
