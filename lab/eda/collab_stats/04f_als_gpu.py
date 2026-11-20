"""Implicit-feedback ALS (Hu, Koren, Volinsky 2008) with conjugate-gradient updates on GPU (torch, CUDA_VISIBLE_DEVICES=N).
(The CPU 'implicit' library stalled on lock contention on this heavily shared machine.)
Matrix: deduped (user,item) from train+val+test histories (+ train positive impressions); items with >=2 users or val/test candidates."""
import os, sys, time
import numpy as np, polars as pl, torch
from common import *
t0=time.time()
FACT=int(sys.argv[1]) if len(sys.argv)>1 else 64
ITERS=int(sys.argv[2]) if len(sys.argv)>2 else 12
WMODE=sys.argv[3] if len(sys.argv)>3 else 'evt'
ALPHA=float(sys.argv[4]) if len(sys.argv)>4 else 2.0
REG=float(sys.argv[5]) if len(sys.argv)>5 else 0.05
TAG=f'alsgpu_f{FACT}_i{ITERS}_{WMODE}_a{ALPHA:g}'
dev='cuda'
g=pl.read_parquet(f'{OUT}/ui_interactions.parquet')
u_raw=g['u'].to_numpy(); i_raw=g['i'].to_numpy(); m=g['m'].to_numpy(); del g
rel_items=np.load(f'{OUT}/rel_items.npy')
icnt=np.bincount(i_raw, minlength=NITEM)
keepm=icnt>=2; keepm[rel_items]|=icnt[rel_items]>=1
ilut=np.full(NITEM,-1,np.int64); keep=np.nonzero(keepm)[0]; ilut[keep]=np.arange(len(keep))
ulut=np.full(u_raw.max()+1,-1,np.int64); users=np.unique(u_raw); ulut[users]=np.arange(len(users))
ii=ilut[i_raw]; mk=ii>=0; ii=ii[mk]; uu=ulut[u_raw[mk]]; m=m[mk]
nbits=np.zeros(len(m),np.float32)
for e in range(1,11): nbits+=((m>>e)&1)
tp=((m>>11)&1).astype(np.float32)
w={'evt':1.0+nbits+3.0*tp,'bin':np.ones(len(m),np.float32)}[WMODE]
nU,nI,nnz=len(users),len(keep),len(m)
tlog(t0,f'users={nU} items={nI} nnz={nnz}')
U_=torch.from_numpy(uu.astype(np.int32)).to(dev); I_=torch.from_numpy(ii.astype(np.int32)).to(dev)
Cm1=torch.from_numpy((ALPHA*w).astype(np.float32)).to(dev)   # c_ui - 1
del uu, ii, m, nbits, tp, w
gen=torch.Generator(device=dev); gen.manual_seed(0)
X=(torch.randn(nU,FACT,device=dev,generator=gen)*0.01)
Y=(torch.randn(nI,FACT,device=dev,generator=gen)*0.01)
CH=int(os.environ.get("ALS_CH","40000000"))
def Av(V, other, rows, cols, n_rows):
    """returns sum_j (c-1)(o_j . v_r) o_j  scattered to rows  (+ handled outside: G v + reg v)"""
    out=torch.zeros(n_rows,FACT,device=dev)
    for s in range(0,nnz,CH):
        r=rows[s:s+CH].long(); c=cols[s:s+CH].long(); cw=Cm1[s:s+CH]
        o=other[c]; d=(o*V[r]).sum(1)*cw
        out.index_add_(0,r,o*d[:,None])
    return out
def bvec(other, rows, cols, n_rows):
    out=torch.zeros(n_rows,FACT,device=dev)
    for s in range(0,nnz,CH):
        r=rows[s:s+CH].long(); c=cols[s:s+CH].long()
        out.index_add_(0,r,other[c]*(1.0+Cm1[s:s+CH])[:,None])
    return out
def update(Xs, other, rows, cols, n_rows, cg_steps=3):
    G=other.T@other + REG*torch.eye(FACT,device=dev)
    b=bvec(other,rows,cols,n_rows)
    r=b-(Xs@G + Av(Xs,other,rows,cols,n_rows)); p=r.clone(); rs=(r*r).sum(1)
    for _ in range(cg_steps):
        Ap=p@G + Av(p,other,rows,cols,n_rows)
        a=rs/(p*Ap).sum(1).clamp(min=1e-20)
        Xs+=a[:,None]*p; r-=a[:,None]*Ap
        rsn=(r*r).sum(1); p=r+(rsn/rs.clamp(min=1e-20))[:,None]*p; rs=rsn
    return Xs
for it in range(ITERS):
    X=update(X,Y,U_,I_,nU); Y=update(Y,X,I_,U_,nI)
    torch.cuda.synchronize(); tlog(t0,f'iter {it+1}/{ITERS}  |X|={X.norm(dim=1).mean():.3f} |Y|={Y.norm(dim=1).mean():.3f}')
# scoring
Yn=Y/Y.norm(dim=1,keepdim=True).clamp(min=1e-6)
for split in ['val','test']:
    f=feeds(split); Mm=f['M'].to_numpy(); H=f['H'].to_numpy(); nf=len(f)
    u=ulut[np.repeat(f['user_id'].to_numpy(),Mm)]; ci=ilut[np.asarray(arr(split,'cand_item_id'))]
    ok=(u>=0)&(ci>=0)
    ut=torch.from_numpy(np.maximum(u,0)).to(dev); ct=torch.from_numpy(np.maximum(ci,0)).to(dev)
    out={'cov':ok}
    s=(X[ut]*Y[ct]).sum(1).cpu().numpy(); out['dot']=np.where(ok,s,np.nan).astype(np.float32)
    hi=ilut[np.asarray(arr(split,'hist_item_id'))]; hp=np.asarray(arr(split,'hist_hpos')); hf=np.repeat(np.arange(nf),H)
    cf=torch.from_numpy(np.repeat(np.arange(nf),Mm)).to(dev)
    for K in [5,20,256]:
        mh=(hi>=0)&(hp<K)
        prof=torch.zeros(nf,FACT,device=dev).index_add_(0,torch.from_numpy(hf[mh]).to(dev),Yn[torch.from_numpy(hi[mh]).to(dev)])
        prof=prof/prof.norm(dim=1,keepdim=True).clamp(min=1e-6)
        s=(prof[cf]*Yn[ct]).sum(1).cpu().numpy(); out[f'lastk{K}']=np.where(ci>=0,s,np.nan).astype(np.float32)
    np.savez(f'{OUT}/{TAG}_{split}.npz',**out)
    print(split,'coverage user&item',ok.mean(),'item',(ci>=0).mean(),flush=True)
need=np.unique(np.concatenate([ilut[np.asarray(arr(sp_,nm_))] for sp_ in ['val','test'] for nm_ in ['cand_item_id','hist_item_id']]))
need=need[need>=0]
np.savez(f'{OUT}/{TAG}_emb_sub.npz', item_id=keep[need], ein=Y[torch.from_numpy(need).to(dev)].cpu().numpy())
Vv=Val('val'); RP=Vv.rand_pf()
d=dict(np.load(f'{OUT}/{TAG}_val.npz'))
for k in ['dot','lastk5','lastk20','lastk256']:
    s=d[k]; x=np.where(np.isnan(s),np.nanmin(s)-1,s)
    a,dd,se=Vv.cmp(x,RP); print(f'{TAG} {k:9s} ndcg={a:.4f} lift={dd:+.4f}+-{se:.4f}',flush=True)
tlog(t0,'done')
