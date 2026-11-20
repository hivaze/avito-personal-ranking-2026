"""Implicit ALS on deduped user-item interactions from train+val+test histories (+train positives). Score val candidates."""
import os, sys
os.environ['OPENBLAS_NUM_THREADS']='1'; os.environ['MKL_NUM_THREADS']='1'
import numpy as np, polars as pl, time, scipy.sparse as sp
from common import *
import implicit
t0=time.time()
FACT=int(sys.argv[1]) if len(sys.argv)>1 else 64
ITERS=int(sys.argv[2]) if len(sys.argv)>2 else 12
WMODE=sys.argv[3] if len(sys.argv)>3 else 'evt'
TAG=f'als_f{FACT}_i{ITERS}_{WMODE}'
g=pl.read_parquet(f'{OUT}/ui_interactions.parquet')
rel_items=np.load(f'{OUT}/rel_items.npy')
# item filter: >=2 distinct users or relevant candidate item
icnt=g.group_by('i').len()
keep=icnt.filter(pl.col('len')>=2)['i'].to_numpy()
keep=np.union1d(keep, np.intersect1d(rel_items, icnt['i'].to_numpy()))
ilut=np.full(NITEM,-1,np.int32); ilut[keep]=np.arange(len(keep),dtype=np.int32)
users=g['u'].unique().sort().to_numpy()
ulut=np.full(users.max()+1,-1,np.int32); ulut[users]=np.arange(len(users),dtype=np.int32)
ii=ilut[g['i'].to_numpy()]; mk=ii>=0
uu=ulut[g['u'].to_numpy()][mk]; ii=ii[mk]; m=g['m'].to_numpy()[mk]
nbits=np.zeros(len(m),np.float32)
for e in range(1,11): nbits+=((m>>e)&1)
tp=((m>>11)&1).astype(np.float32)
if WMODE=='evt':
    w=1.0+nbits+3.0*tp       # confidence weight before alpha
elif WMODE=='bin':
    w=np.ones(len(m),np.float32)
elif WMODE=='et1only':
    w=(((m>>1)&1)+3.0*tp).astype(np.float32); 
X=sp.csr_matrix((w.astype(np.float32),(uu,ii)),shape=(len(users),len(keep)))
X.eliminate_zeros()
tlog(t0,f'matrix users={X.shape[0]} items={X.shape[1]} nnz={X.nnz}')
from implicit.cpu.als import AlternatingLeastSquares
model=AlternatingLeastSquares(factors=FACT, regularization=0.05, alpha=2.0, iterations=ITERS, num_threads=40, calculate_training_loss=False, random_state=0)
model.fit(X, show_progress=False)
tlog(t0,'fit done')
UF=model.user_factors; IF=model.item_factors
# score val & test candidates
for split in ['val','test']:
    f=feeds(split); Mm=f['M'].to_numpy()
    u=ulut[np.repeat(f['user_id'].to_numpy(),Mm)]
    it=np.asarray(arr(split,'cand_item_id')); ix=ilut[it]
    ok=(u>=0)&(ix>=0)
    s=np.full(len(it),np.nan,np.float32)
    s[ok]=np.einsum('ij,ij->i',UF[u[ok]],IF[ix[ok]])
    # cosine of candidate with mean of last-k history item factors
    H=f['H'].to_numpy(); hi=np.asarray(arr(split,'hist_item_id')); hpos=np.asarray(arr(split,'hist_hpos'))
    hix=ilut[hi]; hfid=np.repeat(np.arange(len(f)),H)
    IFn=IF/np.maximum(np.linalg.norm(IF,axis=1,keepdims=True),1e-6)
    out={'dot':s,'cov':ok}
    for K in [5,20,256]:
        mh=(hix>=0)&(hpos<K)
        prof=np.zeros((len(f),FACT),np.float32)
        np.add.at(prof, hfid[mh], IFn[hix[mh]])
        prof/=np.maximum(np.linalg.norm(prof,axis=1,keepdims=True),1e-6)
        fid=np.repeat(np.arange(len(f)),Mm)
        sk=np.full(len(it),np.nan,np.float32); ok2=ix>=0
        sk[ok2]=np.einsum('ij,ij->i',prof[fid[ok2]],IFn[ix[ok2]])
        out[f'lastk{K}']=sk
    np.savez(f'{OUT}/{TAG}_{split}.npz',**out)
    print(split,'coverage user&item', ok.mean(), 'item only', (ix>=0).mean())
V=Val('val'); RP=V.rand_pf()
d=dict(np.load(f'{OUT}/{TAG}_val.npz'))
for k in ['dot','lastk5','lastk20','lastk256']:
    s=d[k]; 
    for fill in ['min','0']:
        x=np.where(np.isnan(s), (np.nanmin(s)-1 if fill=='min' else 0.0), s)
        a,dd,se=V.cmp(x,RP); print(f'{TAG} {k:9s} fill={fill:3s} ndcg={a:.4f} lift={dd:+.4f}+-{se:.4f}', flush=True)
tlog(t0,'done')
