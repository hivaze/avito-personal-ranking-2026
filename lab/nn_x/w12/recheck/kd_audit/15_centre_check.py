"""teacher_mean11 / mean25 = per-feed-centred mean of the member sums? (first 200K feeds + 200K random M<50 feeds). CPU only."""
import numpy as np, polars as pl
K='<repo>/nn_x/w12/kd/out/'
f=pl.read_parquet('<repo>/data/flat/train/feeds.parquet',columns=['M','cand_off']); M=f['M'].to_numpy().astype(np.int64); off=f['cand_off'].to_numpy().astype(np.int64)
nf=200000; n=int(off[nf]); fid=np.repeat(np.arange(nf),M[:nf])
L=lambda t: np.asarray(np.load(K+f'teacher_{t}.npy',mmap_mode='r')[:n],np.float64)
cen=lambda x: x-np.bincount(fid,weights=x)[fid]/M[:nf][fid]
s11=sum(L(t) for t in 'abcdef'); m11=L('mean11'); print('mean11 vs centred (a..f)/11: max|diff|',np.abs(cen(s11/11)-m11).max(),'| per-feed mean of mean11: max|.|',np.abs(np.bincount(fid,weights=m11)/M[:nf]).max())
s25=s11+L('g1')+L('g2')+L('p1')+L('p2'); m25=L('mean25'); print('mean25 vs centred sum/25: max|diff|',np.abs(cen(s25/25)-m25).max())
# is it the mean of centred logits (not z-scored)? within-feed sd of members vs teacher
sd=lambda x: np.sqrt(np.bincount(fid,weights=cen(x)**2)/M[:nf]).mean()
print('within-feed sd: mean11',f'{sd(m11):.3f}','single member (teacher_f)',f'{sd(L("f")):.3f}','w8g1',f'{sd(L("w8g1")):.3f}','oof',f'{sd(L("oof")):.3f}')
