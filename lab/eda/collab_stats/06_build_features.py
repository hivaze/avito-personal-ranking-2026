"""Assemble per-candidate feature table for a split (val; test analogue uses the same code paths)."""
import numpy as np, polars as pl, time, sys, glob, pickle
from common import *
t0=time.time()
split='val'
f=feeds(split); M=f['M'].to_numpy(); n=len(f); fid=np.repeat(np.arange(n),M)
rel_items=np.load(f'{OUT}/rel_items.npy'); lut=np.full(NITEM,-1,np.int32); lut[rel_items]=np.arange(len(rel_items),dtype=np.int32)
ci=np.asarray(arr(split,'cand_item_id')); r=lut[ci]
F={}
F['fidx']=fid.astype(np.int32)
for c in ['f1','f2','f3','f4','f5']: F[c]=np.asarray(arr(split,'cand_'+c)).astype(np.float32)
F['cat']=np.asarray(arr(split,'cand_category_id')).astype(np.int32)
F['M']=M[fid].astype(np.float32)
if split!='test': F['target']=np.asarray(arr(split,'cand_target')).astype(np.int8)
# --- item TE (train, all days)
A=dict(np.load(f'{OUT}/item_day_aggs.npz')); W={k:v.sum(1) for k,v in A.items()}
nn_=W['n'][r]; cc=W['c'][r]; p0=W['c'].sum()/W['n'].sum()
F['ite_n_imp']=nn_; F['ite_clicks']=cc
F['ite_ctr_a20']=(cc+20*p0)/(nn_+20)
F['ite_coec_Ef_a20']=(cc+20)/(W['Ef'][r]+20)
F['ite_coec_Eg_a20']=(cc+20)/(W['Eg'][r]+20)
F['ite_mean_pos']=np.where(nn_>0,W['spos'][r]/np.maximum(nn_,1),np.nan).astype(np.float32)
F['ite_mean_invlogpos']=np.where(nn_>0,W['sinvlog'][r]/np.maximum(nn_,1),np.nan).astype(np.float32)
# --- popularity
HC=np.load(f'{OUT}/train_hist_counts.npy')
F['pop_trhist_all']=HC.sum(1)[r]
for e in range(1,11): F[f'pop_trhist_et{e}']=HC[r,e]
T=dict(np.load(f'{OUT}/trans_counts.npz'))
vts=np.repeat(f['timestamp'].to_numpy().astype(np.int64),M)
last=T['last'][r]; first=T['first'][r]
F['pop_age_last_seen_h']=np.where(last>0,(vts-last)/3600,np.nan).astype(np.float32)
F['pop_age_first_seen_h']=np.where(first<2**40,(vts-first)/3600,np.nan).astype(np.float32)
pre='val' if split=='val' else 'test'
F['tr_cand_feed_cnt']=T[f'{pre}_cc'][r].astype(np.float32)
F['tr_cand_dist_users']=T[f'{pre}_du'][r].astype(np.float32)
F['tr_hist_all']=T[f'{pre}_hc'].sum(1)[r].astype(np.float32)
F['tr_hist_et1']=T[f'{pre}_hc'][r,1].astype(np.float32)
# --- user / user x cat (train)
rel_users=np.load(f'{OUT}/rel_users.npy'); ulut=np.full(rel_users.max()+1,-1,np.int32); ulut[rel_users]=np.arange(len(rel_users),dtype=np.int32)
U=dict(np.load(f'{OUT}/user_cat_aggs.npz'))
vu=ulut[np.repeat(f['user_id'].to_numpy(),M)]; vc=F['cat']
n_uc=U['n'][vu,vc]; c_uc=U['c'][vu,vc]; n_u=U['n'][vu].sum(1); c_u=U['c'][vu].sum(1)
p_u=(c_u+20*0.0217)/(n_u+20)
F['usr_ctr']=p_u.astype(np.float32); F['usr_nfeeds']=U['nfeeds'][vu]; F['usr_impr']=n_u
F['uc_ctr']=((c_uc+5*p_u)/(n_uc+5)).astype(np.float32); F['uc_clicks']=c_uc; F['uc_impr']=n_uc
F['uc_coec']=((c_uc+2)/(U['Ef'][vu,vc]+2)).astype(np.float32)
# --- own history category affinity (reference group)
H=f['H'].to_numpy(); hc=np.asarray(arr(split,'hist_category_id')).astype(int); he=np.asarray(arr(split,'hist_event_type')).astype(int); hp=np.asarray(arr(split,'hist_hpos'))
hf=np.repeat(np.arange(n),H)
def catcount(mask):
    X=np.zeros((n,54),np.float32); np.add.at(X,(hf[mask],hc[mask]),1); return X[fid,vc]
F['oh_cat_cnt']=catcount(np.ones(len(hc),bool)); F['oh_cat_cnt_et1']=catcount(he==1); F['oh_cat_cnt_last20']=catcount(hp<20)
F['oh_cat_share']=F['oh_cat_cnt']/np.maximum(H[fid],1)
F['oh_H']=H[fid].astype(np.float32)
# --- repeat exposure + future-history
if split=='val':
    rp=np.load(f'{OUT}/val_repeat.npy'); F['rep_n_prev']=rp[0].astype(np.float32); F['rep_pos_prev']=rp[1].astype(np.float32)
    F['rep_future_hist']=np.load(f'{OUT}/val_future_hist.npy').astype(np.float32)
# --- CF
for path in sorted(glob.glob(f'{OUT}/als_*_{split}.npz'))+sorted(glob.glob(f'{OUT}/i2v_*_{split}.npz')):
    tag=path.split('/')[-1].replace(f'_{split}.npz','')
    d=np.load(path)
    for k in d.files:
        if k=='cov': continue
        F[f'cf_{tag}_{k}']=d[k].astype(np.float32)
df=pl.DataFrame(F)
df.write_parquet(f'{OUT}/features_{split}.parquet')
print(df.shape); print(df.columns)
tlog(t0,'done')
