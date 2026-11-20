"""F: teacher soft-target diagnostics. Label mass / entropy / optimal temperature of softmax(teacher/T):
 holdout feeds (teacher out-of-sample) vs train exploration feeds and production feeds (teacher in-sample), and the cross-fitted teacher on the same feeds.
 Labels are used for DIAGNOSTICS only (nothing here is a model input). CPU only."""
import sys; sys.path.insert(0,'<repo>/src')
import numpy as np, polars as pl
from common import FLAT
from metrics import ndcg_per_feed
from pathlib import Path
R=Path('<repo>'); K=R/'nn_x/w12/kd/out'
f=pl.read_parquet(FLAT/'train/feeds.parquet',columns=['M','cand_off']); M=f['M'].to_numpy().astype(np.int64); off=f['cand_off'].to_numpy().astype(np.int64); n=len(M)
y=np.load(FLAT/'train/cand_target.npy',mmap_mode='r')
pr=np.load(R/'eda/labels_position/out/train_prandom.npz'); p=np.nan_to_num(pr['p_random'].astype(np.float32),nan=0.0); p[M<50]=0
hold=np.load('hold_idx.npy'); ish=np.zeros(n,bool); ish[hold]=True
rng=np.random.default_rng(7)
G={'holdout (teacher OOS)':hold,
   'expl train (in-sample), M>=50 p>0.9':np.sort(rng.choice(np.where((p>0.9)&(M>=50)&~ish)[0],30000,replace=False)),
   'prod train M>=50 (in-sample)':np.sort(rng.choice(np.where((p<=0.5)&(M>=50))[0],60000,replace=False)),
   'prod train M<50 (in-sample)':np.sort(rng.choice(np.where((p<=0.5)&(M<50))[0],60000,replace=False))}
T=np.array([0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.25,1.5,2.0])
def seg(fe):
    rows=np.concatenate([np.arange(o,o+m) for o,m in zip(off[fe],M[fe])]); Ms=M[fe]; o2=np.concatenate([[0],np.cumsum(Ms)[:-1]]); fid=np.repeat(np.arange(len(fe)),Ms)
    return rows,Ms,o2,fid
def soft(t,fid,nf,Tm):
    z=t/Tm; mx=np.maximum.reduceat(z,np.r_[0,np.flatnonzero(np.diff(fid))+1]); e=np.exp(z-mx[fid]); return e/np.bincount(fid,weights=e,minlength=nf)[fid]
for nm,fe in G.items():
    rows,Ms,o2,fid=seg(fe); yy=np.asarray(y[rows],np.float64); nf=len(fe); npos=np.bincount(fid,weights=yy,minlength=nf); yt=yy/npos[fid]
    print(f'== {nm}: {nf} feeds, mean M {Ms.mean():.1f}, mean npos {npos.mean():.2f}, uniform CE {np.mean(np.log(Ms)):.4f}')
    for tn in ('mean11','mean25','w8g1','oof'):
        t=np.asarray(np.load(K/f'teacher_{tn}.npy',mmap_mode='r')[rows],np.float64)
        ce=[]; 
        for Tm in T:
            q=soft(t,fid,nf,Tm); ce.append(-np.bincount(fid,weights=yt*np.log(np.maximum(q,1e-30)),minlength=nf).mean())
        q=soft(t,fid,nf,1.0); mass=np.bincount(fid,weights=q*yt,minlength=nf)  # teacher prob on a (uniformly drawn) positive, per positive
        massall=np.bincount(fid,weights=q*yy,minlength=nf)   # total teacher mass on the positives
        ent=-np.bincount(fid,weights=q*np.log(np.maximum(q,1e-30)),minlength=nf)
        sdw=np.sqrt(np.bincount(fid,weights=(t-np.bincount(fid,weights=t)[fid]/Ms[fid])**2)/Ms)
        nd=ndcg_per_feed(t.astype(np.float32),yy.astype(np.float32),o2,Ms).mean()
        k=int(np.argmin(ce)); 
        print(f'   {tn:7s} NDCG {nd:.5f} | CE@T=1 {ce[5]:.4f} best T {T[k]:.2f} (CE {ce[k]:.4f}) | teacher mass on positives {massall.mean():.4f} (base rate {np.mean(npos/Ms):.4f}) | entropy {ent.mean():.3f} (log M {np.mean(np.log(Ms)):.3f}) | within-feed logit sd {sdw.mean():.3f}')
        if tn=='mean11': q11=q
        if tn=='oof':
            kl=np.bincount(fid,weights=q11*(np.log(np.maximum(q11,1e-30))-np.log(np.maximum(q,1e-30))),minlength=nf)
            print(f'   KL(softmax mean11 || softmax oof) per feed: mean {kl.mean():.4f} median {np.median(kl):.4f}; extra mass of mean11 on positives vs oof: {(np.bincount(fid,weights=(q11-q)*yy,minlength=nf)).mean():+.4f}')
