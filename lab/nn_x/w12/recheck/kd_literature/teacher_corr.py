"""How different are the stored teachers? rank-z correlation on the 30K holdout feeds and on 30K production train feeds."""
import sys; sys.path.insert(0,'<repo>/src')
import numpy as np, polars as pl
from stack_val import feed_rank_z
ROOT='<repo>'; KD=f'{ROOT}/nn_x/w12/kd/out'
f=pl.read_parquet(f'{ROOT}/data/flat/train/feeds.parquet',columns=['M','cand_off']); M=f['M'].to_numpy().astype(np.int64); off=f['cand_off'].to_numpy().astype(np.int64)
pr=np.nan_to_num(np.load(f'{ROOT}/eda/labels_position/out/train_prandom.npz')['p_random'].astype(np.float32),nan=0.0); pr[M<50]=0.0
sel=np.where((pr>0.9)&(M>=50))[0]; hold=np.sort(np.random.default_rng(12345).choice(sel,30000,replace=False))
prod=np.sort(np.random.default_rng(7).choice(np.where(pr<0.1)[0],30000,replace=False))
names=['mean11','mean25','w8g1','oof','mix25w8']
for nm,fe in (('holdout',hold),('production train (in-sample for all but oof)',prod)):
    Ms=M[fe]; rows=np.concatenate([np.arange(o,o+m) for o,m in zip(off[fe],Ms)]); o2=np.concatenate([[0],np.cumsum(Ms)[:-1]])
    Z=[feed_rank_z(o2,Ms,np.asarray(np.load(f'{KD}/teacher_{k}.npy',mmap_mode='r')[rows]).astype(np.float32))[:,1] for k in names]
    C=np.corrcoef(Z); print(nm); print('          '+' '.join(f'{k:>8s}' for k in names))
    for i,k in enumerate(names): print(f'{k:>9s} '+' '.join(f'{C[i,j]:8.4f}' for j in range(len(names))))
