"""Deduplicated (user,item) interaction table from train+val+test histories (+ train positive impressions)."""
import numpy as np, polars as pl, time, os
os.environ.setdefault('POLARS_MAX_THREADS','40')
from common import *
t0=time.time()
parts=[]
for s in ['train','val','test']:
    f=feeds(s); H=f['H'].to_numpy()
    u=np.repeat(f['user_id'].to_numpy().astype(np.int32), H)
    it=np.load(f'{FLAT}/{s}/hist_item_id.npy'); et=np.load(f'{FLAT}/{s}/hist_event_type.npy')
    parts.append(pl.DataFrame({'u':u,'i':it,'m':(np.int32(1)<<et.astype(np.int32)).astype(np.int32)}))
    tlog(t0,f'{s} hist rows {len(it)}')
# train positive candidates as pseudo event type 11
f=feeds('train'); M=f['M'].to_numpy()
tg=np.load(f'{FLAT}/train/cand_target.npy').astype(bool)
u=np.repeat(f['user_id'].to_numpy().astype(np.int32), M)[tg]
it=np.load(f'{FLAT}/train/cand_item_id.npy')[tg]
parts.append(pl.DataFrame({'u':u,'i':it,'m':np.full(len(u),1<<11,np.int32)}))
df=pl.concat(parts); del parts
tlog(t0,f'concat {len(df)}')
g=df.group_by(['u','i']).agg(pl.col('m').bitwise_or().alias('m'), pl.len().alias('n'))
tlog(t0,f'dedup -> {len(g)}')
g.write_parquet(f'{OUT}/ui_interactions.parquet')
# stats
m=g['m'].to_numpy()
for e in range(1,12): print('event', e, 'pairs with it', int(((m>>e)&1).sum()))
tlog(t0,'done')
