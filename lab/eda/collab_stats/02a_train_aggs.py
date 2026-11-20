"""Compute per-item / per-category / per-user train aggregates (impressions, clicks, expected clicks under
position curve) restricted to items relevant for val/test; per-day breakdown for windowing."""
import numpy as np, polars as pl, time
from common import *
t0=time.time()
ftr = feeds('train')
M = ftr['M'].to_numpy().astype(np.int64)
ts = ftr['timestamp'].to_numpy().astype(np.int64)
day_f = ((ts - ts.min()) // 86400).astype(np.int8)  # 0..13
print('feeds per day', np.bincount(day_f))
item = np.load(f'{FLAT}/train/cand_item_id.npy')
tgt = np.load(f'{FLAT}/train/cand_target.npy').astype(np.float32)
pos = np.load(f'{FLAT}/train/cand_pos.npy').astype(np.int64)
cat = np.load(f'{FLAT}/train/cand_category_id.npy').astype(np.int64)
fidx = np.repeat(np.arange(len(M)), M)
tlog(t0,'loaded')
# position curve
npos = np.bincount(pos, minlength=300); cpos = np.bincount(pos, weights=tgt, minlength=300)
ctr_pos = cpos/np.maximum(npos,1)
np.save(f'{OUT}/ctr_pos.npy', ctr_pos)
print('pos curve (pos: ctr, n):', [(p, round(ctr_pos[p],4), int(npos[p])) for p in [0,1,2,3,5,10,20,30,50,100,150,200,250,299]])
# by M-bin too
Mrow = M[fidx]
for lo,hi in [(5,20),(20,60),(60,150),(150,301)]:
    m=(Mrow>=lo)&(Mrow<hi)
    n_=np.bincount(pos[m],minlength=300); c_=np.bincount(pos[m],weights=tgt[m],minlength=300)
    cc=c_/np.maximum(n_,1)
    print(f' M[{lo},{hi}) rows {m.sum()} ctr={tgt[m].mean():.4f}', [round(cc[p],4) for p in [0,1,2,5,10,20,50,100] if p<hi])
# expected clicks
w = ctr_pos[pos].astype(np.float32)
k_f = np.bincount(fidx, weights=tgt, minlength=len(M))
sw_f = np.bincount(fidx, weights=w, minlength=len(M))
E_feed = (k_f[fidx]*w/sw_f[fidx]).astype(np.float32)
E_unif = (k_f/M)[fidx].astype(np.float32)
del sw_f
tlog(t0,'expected clicks')
# relevant items
rel_items = np.unique(np.concatenate([np.asarray(arr(s,'cand_item_id')) for s in ['val','test']]))
lut = np.full(NITEM, -1, np.int32); lut[rel_items] = np.arange(len(rel_items), dtype=np.int32)
np.save(f'{OUT}/rel_items.npy', rel_items)
r = lut[item]; m = r>=0
print('train rows on relevant items', m.mean())
R = len(rel_items); D=14
dayrow = day_f[fidx]
key = (r[m].astype(np.int64)*D + dayrow[m])
aggs = {}
aggs['n'] = np.bincount(key, minlength=R*D).reshape(R,D).astype(np.float32)
for nm,v in [('c',tgt),('Eg',w),('Ef',E_feed),('Eu',E_unif)]:
    aggs[nm] = np.bincount(key, weights=v[m], minlength=R*D).reshape(R,D).astype(np.float32)
# sum of pos (mean pos is a "production rank" signal) and min pos
aggs['spos'] = np.bincount(key, weights=pos[m].astype(np.float64), minlength=R*D).reshape(R,D).astype(np.float32)
aggs['sinvlog'] = np.bincount(key, weights=(1/np.log2(pos[m]+2.0)), minlength=R*D).reshape(R,D).astype(np.float32)
np.savez(f'{OUT}/item_day_aggs.npz', **aggs)
tlog(t0,'item aggs')
# global item impressions over ALL items for popularity ranking (dense)
# categories
CD = 54
for nm,v in [('n',None),('c',tgt),('Eg',w),('Ef',E_feed),('Eu',E_unif)]:
    aggs_c = np.bincount(cat, weights=v, minlength=CD)
    np.save(f'{OUT}/cat_{nm}.npy', aggs_c)
# users: user-level and user x category
uid = ftr['user_id'].to_numpy().astype(np.int64)
rel_users = np.unique(np.concatenate([feeds(s)['user_id'].to_numpy() for s in ['val','test']]))
ulut = np.full(uid.max()+3_000_000, -1, np.int32); ulut[rel_users]=np.arange(len(rel_users))
np.save(f'{OUT}/rel_users.npy', rel_users)
ur = ulut[uid][fidx]; mu = ur>=0
print('train rows from relevant users', mu.mean())
U=len(rel_users)
ukey = ur[mu].astype(np.int64)*CD + cat[mu]
uc = {}
uc['n'] = np.bincount(ukey, minlength=U*CD).reshape(U,CD).astype(np.float32)
for nm,v in [('c',tgt),('Ef',E_feed),('Eu',E_unif)]:
    uc[nm] = np.bincount(ukey, weights=v[mu], minlength=U*CD).reshape(U,CD).astype(np.float32)
uc['nfeeds'] = np.bincount(ulut[uid][ulut[uid]>=0], minlength=U).astype(np.float32)
np.savez(f'{OUT}/user_cat_aggs.npz', **uc)
tlog(t0,'user aggs')
# repeated exposure: (user,item) pairs from train among relevant users & relevant items
mm = mu & m
pairs = pl.DataFrame({'u': ur[mm], 'i': r[mm], 't': tgt[mm].astype(np.int8), 'ts': ts[fidx][mm], 'pos': pos[mm].astype(np.int16)})
pairs.write_parquet(f'{OUT}/train_user_item_pairs.parquet')
print('pairs rows', len(pairs))
tlog(t0,'done')
