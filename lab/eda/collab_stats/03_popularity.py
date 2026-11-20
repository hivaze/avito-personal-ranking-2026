import numpy as np, polars as pl, time
from common import *
t0=time.time()
V = Val('val'); RP = V.rand_pf()
rel_items = np.load(f'{OUT}/rel_items.npy')
lut = np.full(NITEM, -1, np.int32); lut[rel_items]=np.arange(len(rel_items),dtype=np.int32)
R=len(rel_items)
vi = np.asarray(arr('val','cand_item_id')); r = lut[vi]
res = {}
def rep(nm, s, store=True):
    a,d,se = V.cmp(s, RP); print(f'{nm:40s} ndcg={a:.4f} lift_vs_random={d:+.4f}+-{se:.4f}', flush=True)
    if store: res[nm]=s
rep('item_id (higher=newer?)', vi.astype(np.float64))
# ---- train history counts by event type for relevant items
hi = np.load(f'{FLAT}/train/hist_item_id.npy'); he = np.load(f'{FLAT}/train/hist_event_type.npy').astype(np.int64)
hr = lut[hi]; m = hr>=0
print('train hist rows on relevant items:', m.mean())
HC = np.bincount(hr[m].astype(np.int64)*11+he[m], minlength=R*11).reshape(R,11).astype(np.float32)
del hi, he
tlog(t0,'train hist counts')
np.save(f'{OUT}/train_hist_counts.npy', HC)
rep('train_hist_cnt_all', HC.sum(1)[r])
for e in range(1,11):
    rep(f'train_hist_cnt_et{e}', HC[r,e])
# ---- first/last seen as train candidate (feed timestamp) -> recency/age
ftr=feeds('train'); M=ftr['M'].to_numpy(); ts=ftr['timestamp'].to_numpy()
ci = np.load(f'{FLAT}/train/cand_item_id.npy'); cr = lut[ci]; mc = cr>=0
tsr = np.repeat(ts, M)[mc]
last = np.full(R, -1, np.int64); np.maximum.at(last, cr[mc], tsr)
first = np.full(R, 2**40, np.int64); np.minimum.at(first, cr[mc], tsr)
del ci, tsr
vts = np.repeat(V.f['timestamp'].to_numpy().astype(np.int64), V.M)
age_first = np.where(first[r] < 2**40, vts-first[r], -1)
rep('train_last_seen (recency, higher=recent)', last[r].astype(float))
rep('-age since first train impression', np.where(age_first>=0, -age_first, -1e9).astype(float))
tlog(t0,'recency')
# item_id vs first seen correlation
ok = first<2**40
print('spearman(item_id, first_seen_ts) over relevant items seen in train:', pl.DataFrame({'a':rel_items[ok],'b':first[ok]}).select(pl.corr('a','b',method='spearman')).item())
# ---- transductive within split counts
def trans(split):
    f=feeds(split); Ms=f['M'].to_numpy(); c=np.asarray(arr(split,'cand_item_id'))
    u=np.repeat(f['user_id'].to_numpy(),Ms)
    cnt_c = np.bincount(lut[c], minlength=R)
    # distinct users
    du = pl.DataFrame({'i':lut[c],'u':u}).unique().group_by('i').len()
    dU = np.zeros(R); dU[du['i'].to_numpy()] = du['len'].to_numpy()
    h=np.asarray(arr(split,'hist_item_id')); he=np.asarray(arr(split,'hist_event_type')).astype(np.int64)
    hr=lut[h]; mm=hr>=0
    HCs = np.bincount(hr[mm].astype(np.int64)*11+he[mm], minlength=R*11).reshape(R,11)
    return cnt_c, dU, HCs, c
cc, dU, HCv, _ = trans('val')
rep('val_cand_feed_count (transductive)', cc[r].astype(float))
rep('val_cand_distinct_users', dU[r])
rep('val_hist_cnt_all (transductive)', HCv.sum(1)[r].astype(float))
for e in range(1,11): rep(f'val_hist_cnt_et{e}', HCv[r,e].astype(float), store=False)
# also test-split counts used on val (is "future" popularity informative?)
cct, dUt, HCt, _ = trans('test')
rep('test_cand_feed_count (future split)', cct[r].astype(float))
rep('test_hist_cnt_all (future split)', HCt.sum(1)[r].astype(float))
np.savez(f'{OUT}/trans_counts.npz', val_cc=cc, val_du=dU, val_hc=HCv, test_cc=cct, test_du=dUt, test_hc=HCt, last=last, first=first)
# ---- leave-one-feed-out item CTR inside val (upper bound of transductive label use; what val labels can give test)
t=V.target
ci_ = np.bincount(r, weights=t, minlength=R); ni_ = np.bincount(r, minlength=R)
loo_c = ci_[r]-t; loo_n = ni_[r]-1
p0=t.mean()
for a in [1,5,20]:
    rep(f'val LOO item ctr bayes a{a} (in-split labels)', (loo_c+a*p0)/(loo_n+a))
print('val rows whose item appears in another val feed:', (loo_n>0).mean())
# time-honest version: val day0 labels -> val day2 feeds
vday = (V.f['timestamp'].to_numpy()-V.f['timestamp'].min())//86400
rowday = np.repeat(vday, V.M)
m0 = rowday==0
c0 = np.bincount(r[m0], weights=t[m0], minlength=R); n0=np.bincount(r[m0], minlength=R)
s = np.where(rowday>=1, (c0[r]+5*p0)/(n0[r]+5), p0)
fm = vday>=1
pf = V.pf_rt(s); print(f'day0-labels item ctr -> day2 feeds: ndcg={pf[fm].mean():.4f} random={RP[fm].mean():.4f} lift={pf[fm].mean()-RP[fm].mean():+.4f}; day2 rows covered {(n0[r][rowday>=1]>0).mean():.4f}')
import pickle; pickle.dump({k:np.asarray(v,dtype=np.float32) for k,v in res.items()}, open(f'{OUT}/pop_feats_val.pkl','wb'))
tlog(t0,'done')
