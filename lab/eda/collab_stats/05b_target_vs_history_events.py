"""Which history event types does a train positive/negative (user,item) pair show in the same user's OTHER feeds' histories?
Also: is the positive in an earlier feed visible in a LATER feed's history (=> event types that encode target action)."""
import numpy as np, polars as pl, time
from common import *
t0=time.time()
f=feeds('train'); M=f['M'].to_numpy(); H=f['H'].to_numpy()
rng=np.random.RandomState(0)
uc=f.group_by('user_id').len()
multi=uc.filter((pl.col('len')>=2)&(pl.col('len')<=20))['user_id'].to_numpy()
samp=set(rng.choice(multi, 60000, replace=False).tolist())
fs=f.filter(pl.col('user_id').is_in(list(samp))).sort(['user_id','timestamp'])
print('sample feeds', len(fs), 'users', len(samp))
ci=arr('train','cand_item_id'); ct=arr('train','cand_target'); hi=arr('train','hist_item_id'); he=arr('train','hist_event_type')
C=[];Hh=[]
for row in fs.iter_rows(named=True):
    o,m=row['cand_off'],row['M']; C.append(pl.DataFrame({'fidx':np.full(m,row['fidx'],np.int32),'u':np.full(m,row['user_id'],np.int32),'ts':np.full(m,row['timestamp'],np.int32),'i':np.asarray(ci[o:o+m]),'t':np.asarray(ct[o:o+m]).astype(np.int8)}))
    o,h=row['hist_off'],row['H']; Hh.append(pl.DataFrame({'fidx2':np.full(h,row['fidx'],np.int32),'u':np.full(h,row['user_id'],np.int32),'ts2':np.full(h,row['timestamp'],np.int32),'i':np.asarray(hi[o:o+h]),'et':np.asarray(he[o:o+h]).astype(np.int8)}))
C=pl.concat(C); Hh=pl.concat(Hh)
tlog(t0,f'cands {len(C)} hist {len(Hh)}')
J=C.join(Hh,on=['u','i']).filter(pl.col('fidx')!=pl.col('fidx2'))
J=J.with_columns(pl.when(pl.col('ts2')>pl.col('ts')).then(pl.lit('later')).when(pl.col('ts2')<pl.col('ts')).then(pl.lit('earlier')).otherwise(pl.lit('same_ts')).alias('when'))
base=C.group_by('t').len().rename({'len':'n_cands'})
print(base)
# fraction of candidates (by target) that appear in later/earlier other-feed history, per event type
x=J.unique(['fidx','i','when','et']).group_by(['t','when','et']).len()
x=x.join(base,on='t').with_columns((pl.col('len')/pl.col('n_cands')).alias('frac')).sort(['when','et','t'])
pl.Config.set_tbl_rows(100)
print(x.pivot(on='t',index=['when','et'],values='frac').sort(['when','et']))
y=J.unique(['fidx','i','when']).group_by(['t','when']).len().join(base,on='t').with_columns((pl.col('len')/pl.col('n_cands')).alias('frac')).sort(['when','t'])
print(y)
tlog(t0,'done')
