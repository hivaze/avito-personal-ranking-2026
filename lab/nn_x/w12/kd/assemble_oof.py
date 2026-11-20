"""Cross-fitted teacher: out/teacher_xf<K>.npy (SUM of 3 fold-K-excluded models) -> out/teacher_oof.npy; sanity NDCG on exploration train feeds."""
import sys; sys.path.insert(0,'<repo>/src')
from common import *
from metrics import ndcg_per_feed
import polars as pl
f=pl.read_parquet(FLAT/'train/feeds.parquet',columns=['user_id','M','cand_off'])
uid=f['user_id'].to_numpy().astype(np.int64); M=f['M'].to_numpy().astype(np.int64); off=f['cand_off'].to_numpy().astype(np.int64); n=len(uid)
xf=np.where(uid==0,np.arange(n)%5,(uid*2654435761%4294967296)%5)
cf=np.repeat(xf,M); assert len(cf)==246745836
oof=np.zeros(len(cf),np.float32)
for k in range(5):
    t=np.load(f'out/teacher_xf{k}.npy',mmap_mode='r'); nk=int(open(f'out/teacher_xf{k}.n').read()); assert nk==3
    m=cf==k; oof[m]=np.asarray(t[m])/nk; print('fold',k,int(m.sum()),flush=True)
np.save('out/teacher_oof.npy',oof)
y=np.load(FLAT/'train/cand_target.npy'); pr=np.load(ROOT/'eda/labels_position/out/train_prandom.npz')['p_random']; pr=np.nan_to_num(pr,nan=0.0)
sel=np.where((pr>0.9)&(M>=50))[0]; hold=np.sort(np.random.default_rng(12345).choice(sel,30000,replace=False)); ex=np.setdiff1d(sel,hold)
ex=np.sort(np.random.default_rng(1).choice(ex,30000,replace=False))
def nd(s,fe):
    rows=np.concatenate([np.arange(o,o+m) for o,m in zip(off[fe],M[fe])]); o2=np.concatenate([[0],np.cumsum(M[fe])[:-1]])
    return ndcg_per_feed(np.asarray(s[rows],np.float32),y[rows].astype(np.float32),o2,M[fe]).mean()
ins=np.load('out/teacher_mean11.npy',mmap_mode='r')
print(f'exploration train feeds (30K, not holdout): OOF teacher NDCG {nd(oof,ex):.5f} | in-sample teacher_mean11 {nd(ins,ex):.5f}')
print(f'holdout feeds (unseen by all): OOF-file {nd(oof,hold):.5f} | teacher_mean11 {nd(ins,hold):.5f}')
