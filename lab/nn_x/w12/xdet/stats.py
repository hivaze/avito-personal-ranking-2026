import sys; sys.path.insert(0,'<repo>/src')
import numpy as np, polars as pl
from scipy.stats import rankdata
D='<repo>/data/flat/train/'
pos=np.load(D+'cand_pos.npy'); fd=pl.read_parquet(D+'feeds.parquet'); print(fd.columns)
aux=sum(np.load(f'out/aux_s{s}.npy').astype(np.float32) for s in (101,102,103))/3
old=np.load('<repo>/eda/labels_position/out/train_prandom.npz')
M=np.load(D+'cand_fidx.npy'); M=np.bincount(M); n=len(M); off=np.concatenate([[0],np.cumsum(M)[:-1]])
fid=np.repeat(np.arange(n),M)
# within-feed ranks via lexsort
def wrank(x):
    o=np.lexsort((x,fid)); r=np.empty(len(x),np.float32); r[o]=(np.arange(len(x))-off[fid[o]]).astype(np.float32); return r
ra=wrank(-aux); rp=wrank(pos.astype(np.float32))
# spearman per feed (no ties handling; aux ties negligible)
ma=(M-1)/2.0; d=(ra-ma[fid])*(rp-ma[fid]); num=np.bincount(fid,d,n); var=M*(M.astype(np.float64)**2-1)/12.0
rho=np.where(M>1,num/np.maximum(var,1e-9),0); z=rho*np.sqrt(np.maximum(M-1,1))
mx=np.maximum.reduceat(pos,off).astype(np.int64); gap=1-M/(mx+1)
np.savez('out/stats.npz',rho=rho.astype(np.float32),z=z.astype(np.float32),M=M,gap=gap.astype(np.float32),maxpos=mx)
p=old['p_random']; 
for lo,hi in [(2,5),(5,10),(10,20),(20,35),(35,50),(50,100),(100,200),(200,301)]:
    m=(M>=lo)&(M<hi); zz=z[m]
    kr=m&(p>0.9)&(M>=50); kp=m&(p<0.05)&(M>=50)
    print(f'M[{lo},{hi}) n={m.sum():8d} rho mean {rho[m].mean():.3f} | z q05 {np.quantile(zz,.05):6.2f} q25 {np.quantile(zz,.25):6.2f} med {np.median(zz):6.2f} | P(z<0) {np.mean(zz<0):.4f} -> pi_ub {2*np.mean(zz<0):.3f} | gap0 share {np.mean(gap[m]==0):.3f}'
          + (f' | old-rand z mean {z[kr].mean():.2f} sd {z[kr].std():.2f}; old-prod z mean {z[kp].mean():.2f} sd {z[kp].std():.2f}' if kr.sum()>100 else ''))
