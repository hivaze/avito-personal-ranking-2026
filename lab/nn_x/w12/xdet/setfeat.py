"""Feed-level SET-composition features (no position, no labels, no popularity/time): mean/std/min/max of time-invariant candidate columns."""
import numpy as np, sys
R='<repo>/data/'
COLS=[('flat','cand_f1'),('flat','cand_f2'),('flat','cand_f3'),('flat','cand_f4'),('flat','cand_f5'),('feats','cos_u'),('feats','cos_hall'),('feats','cos_h10'),('feats','cat_share_all'),('feats','ctx_nd_max'),('feats','centrality'),('feats','cos_het1')]
for sp in ('val','test','train'):
    fidx=np.load(f'{R}flat/{sp}/cand_fidx.npy'); M=np.bincount(fidx); n=len(M); off=np.concatenate([[0],np.cumsum(M)[:-1]])
    assert (np.diff(fidx)>=0).all()
    F=[M.astype(np.float32)]; names=['M']
    for d,c in COLS:
        x=np.load(f'{R}{d}/{sp}/{c}.npy').astype(np.float64); x=np.nan_to_num(x)
        assert len(x)==len(fidx),(c,len(x))
        mu=np.bincount(fidx,x,n)/M; sd=np.sqrt(np.maximum(np.bincount(fidx,x*x,n)/M-mu*mu,0))
        F+=[mu,sd,np.minimum.reduceat(x,off),np.maximum.reduceat(x,off)]; names+=[f'{c}_{a}' for a in ('mu','sd','min','max')]
    cat=np.load(f'{R}flat/{sp}/cand_category_id.npy').astype(np.int64); key=np.unique(fidx*64+cat); F.append(np.bincount(key//64,minlength=n)/M); names.append('ncat_per_M')
    F.append(np.bincount(key//64,minlength=n).astype(np.float32)); names.append('ncat')
    np.save(f'out/setfeat_{sp}.npy',np.stack(F,1).astype(np.float32)); print(sp,n,len(names),flush=True)
open('out/setfeat_names.txt','w').write('\n'.join(names))
