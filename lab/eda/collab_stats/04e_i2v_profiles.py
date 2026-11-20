"""Score val candidates with saved item2vec embeddings using different history-profile constructions."""
import numpy as np, polars as pl, time, sys
from common import *
t0=time.time()
TAG=sys.argv[1] if len(sys.argv)>1 else 'i2v_d128_e10_nf0'
E=np.load(f'{OUT}/{TAG}_emb_sub.npz')
ids=E['item_id']; ein=E['ein'].astype(np.float32)
ein/=np.maximum(np.linalg.norm(ein,axis=1,keepdims=True),1e-6)
lut=np.full(NITEM,-1,np.int64); lut[ids]=np.arange(len(ids))
V=Val('val'); RP=V.rand_pf()
f=V.f; n=V.n; H=f['H'].to_numpy()
ci=lut[np.asarray(arr('val','cand_item_id'))]; cc=np.asarray(arr('val','cand_category_id')).astype(int)
hi=lut[np.asarray(arr('val','hist_item_id'))]; hp=np.asarray(arr('val','hist_hpos')); he=np.asarray(arr('val','hist_event_type')).astype(int); hc=np.asarray(arr('val','hist_category_id')).astype(int)
hf=np.repeat(np.arange(n),H); cf=V.fid
okc=ci>=0
D=ein.shape[1]
def score_profile(w, name):
    m=(hi>=0)&(w>0)
    prof=np.zeros((n,D),np.float32); np.add.at(prof,hf[m],ein[hi[m]]*w[m,None])
    prof/=np.maximum(np.linalg.norm(prof,axis=1,keepdims=True),1e-6)
    s=np.full(len(ci),np.nan,np.float32); s[okc]=np.einsum('ij,ij->i',prof[cf[okc]],ein[ci[okc]])
    x=np.where(np.isnan(s),np.nanmin(s)-1,s)
    a,d,se=V.cmp(x,RP); print(f'{name:40s} ndcg={a:.4f} lift={d:+.4f}+-{se:.4f}',flush=True)
    return s
out={}
for K in [3,5,10]:
    out[f'last{K}']=score_profile((hp<K).astype(np.float32), f'mean last{K}')
for tau in [2,5,10,30]:
    out[f'exp{tau}']=score_profile(np.exp(-hp/tau).astype(np.float32), f'exp-decay tau={tau}')
out['et1_exp5']=score_profile((np.exp(-hp/5)*(he==1)).astype(np.float32), 'exp tau=5, event 1 only')
out['etnot1_exp10']=score_profile((np.exp(-hp/10)*(he!=1)).astype(np.float32), 'exp tau=10, events !=1 only')
# category-conditional profile: per (feed, category) mean of history items in same category
m=hi>=0
key=hf[m].astype(np.int64)*54+hc[m]
uk,inv=np.unique(key,return_inverse=True)
wv=np.exp(-hp[m]/10).astype(np.float32)
P=np.zeros((len(uk),D),np.float32); np.add.at(P,inv,ein[hi[m]]*wv[:,None])
P/=np.maximum(np.linalg.norm(P,axis=1,keepdims=True),1e-6)
ck=cf.astype(np.int64)*54+cc; pos=np.searchsorted(uk,ck); pos=np.minimum(pos,len(uk)-1); hit=(uk[pos]==ck)&okc
s=np.full(len(ci),np.nan,np.float32); s[hit]=np.einsum('ij,ij->i',P[pos[hit]],ein[ci[hit]])
x=np.where(np.isnan(s),np.nanmin(s)-1,s); a,d,se=V.cmp(x,RP); print(f'{"same-category profile (tau=10)":40s} ndcg={a:.4f} lift={d:+.4f}+-{se:.4f} coverage={hit.mean():.3f}')
out['samecat_exp10']=s
np.savez(f'{OUT}/cf_{TAG}_profiles_val.npz',**out)
tlog(t0,'done')
