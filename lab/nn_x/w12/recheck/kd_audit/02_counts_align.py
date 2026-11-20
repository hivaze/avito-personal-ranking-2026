"""A: zero-positive / multi-positive feeds in train (all, production, exploration pool, holdout), val.
B: teacher files vs the members' own saved holdout predictions (row alignment of the 246.7M-row file).
C: teacher_mean11 / mean25 / w8g1 / mix25w8 assembly identity on sampled rows.  CPU only."""
import sys, os; sys.path.insert(0,'<repo>/src')
import numpy as np, polars as pl
from common import feed_meta, FLAT
from metrics import ndcg_per_feed
from pathlib import Path
R=Path('<repo>'); K=R/'nn_x/w12/kd/out'; W=R/'nn_x/w5/preds'
f=pl.read_parquet(FLAT/'train/feeds.parquet',columns=['user_id','M','cand_off','timestamp'])
M=f['M'].to_numpy().astype(np.int64); off=f['cand_off'].to_numpy().astype(np.int64); n=len(M)
y=np.load(FLAT/'train/cand_target.npy',mmap_mode='r')
print('train feeds',n,'cands',len(y),'offset check',bool((off==np.concatenate([[0],np.cumsum(M)[:-1]])).all()))
npos=np.add.reduceat(np.asarray(y,dtype=np.int32),off)
pr=np.load(R/'eda/labels_position/out/train_prandom.npz'); assert (pr['fidx']==np.arange(n)).all()
p=np.nan_to_num(pr['p_random'].astype(np.float32),nan=0.0); p[M<50]=0
sel=np.where((p>0.9)&(M>=50))[0]; hold=np.sort(np.random.default_rng(12345).choice(sel,30000,replace=False))
ish=np.zeros(n,bool); ish[hold]=True
pool=(p>0.5)&~ish
print('A: npos distribution all train:',{k:int((npos==k).sum()) for k in range(0,6)},'>=6',int((npos>=6).sum()))
print('   zero-positive feeds: all',int((npos==0).sum()),f'({(npos==0).mean():.4%}) | production(p<=0.5)',int(((npos==0)&(p<=0.5)).sum()),'| pool',int(((npos==0)&pool).sum()),'| holdout',int((npos[hold]==0).sum()))
print('   multi-positive feeds share',f'{(npos>1).mean():.4f}','mean npos',f'{npos.mean():.4f}','| pool feeds',int(pool.sum()),'pool mean npos',f'{npos[pool].mean():.3f}','prod mean npos',f'{npos[p<=0.5].mean():.3f}')
print('   M<50 feeds',int((M<50).sum()),'M>=50',int((M>=50).sum()),'| mean M',M.mean())
fv,Mv,ov=feed_meta('val'); yv=np.load(FLAT/'val/cand_target.npy'); nv=np.add.reduceat(yv.astype(np.int32),ov)
print('   val: feeds',len(Mv),'zero-positive',int((nv==0).sum()),'multi-pos share',f'{(nv>1).mean():.4f}','mean npos',f'{nv.mean():.3f}')
np.save('npos_train.npy',npos.astype(np.int16)); np.save('hold_idx.npy',hold)
# B
Mh,oh=np.load(W/'holdout/holdout_meta.npy'); Mh=Mh.astype(np.int64); oh=oh.astype(np.int64); yh=np.load(W/'holdout/holdout_y.npy')
assert (Mh==M[hold]).all(), 'holdout meta M mismatch'
rows=np.concatenate([np.arange(o,o+m) for o,m in zip(off[hold],M[hold])])
assert (np.asarray(y[rows]).astype(np.float32)==yh.astype(np.float32)).all(), 'holdout y mismatch'
print('B: holdout identity OK: 30000 feeds,',len(rows),'rows; holdout_y == train target on rows')
def cen(s): # per-feed centring
    fid=np.repeat(np.arange(len(Mh)),Mh); return s-np.bincount(fid,weights=s)[fid]/Mh[fid]
eh=lambda s: ndcg_per_feed(np.asarray(s,np.float32),yh,oh,Mh)
def chk(tag,files,nm):
    t=np.asarray(np.load(K/f'teacher_{tag}.npy',mmap_mode='r')[rows],np.float64)
    hs=[np.load(x).astype(np.float64) for x in files if Path(x).exists()]
    if len(hs)!=len(files): print('  ',tag,'missing holdout files',len(hs),len(files)); 
    if not hs: return
    m=np.sum(hs,0); a,b=cen(t),cen(m)
    print(f'  teacher_{tag} (n={nm}) vs sum of {len(hs)} member holdout files: corr(centred) {np.corrcoef(a,b)[0,1]:.6f} max|diff| {np.abs(a-b).max():.3f} rms {np.sqrt(((a-b)**2).mean()):.4f} (sd {a.std():.3f}) | NDCG teacher part {eh(t).mean():.5f} members {eh(m).mean():.5f}')
r6=R/'nn_x/r6/preds/holdout'; ab=R/'nn_x/abl/preds/holdout'
chk('a',[r6/f'nn_r6_cd_m02_r{r}_holdout.npy' for r in (101,102)],2)
chk('d',[r6/'nn_r6_cd_m02_r107_holdout.npy',W/'holdout/nn_w5_e10_base_s953_holdout.npy'],2)
chk('e',[W/f'holdout/nn_w5_e10_base_s{s}_holdout.npy' for s in (954,951)],2)
chk('f',[W/'holdout/nn_w5_e10_base_s952_holdout.npy'],1)
chk('p1',[ab/f'nn_abl_e10_pop1_s{s}_holdout.npy' for s in (951,952,953,954)],4)
chk('p2',[ab/f'nn_abl_e10_pop1_s{s}_holdout.npy' for s in (955,956,957,958)],4)
chk('h1',[W/f'holdout/nn_w5_e10_kd05w192_s{s}_holdout.npy' for s in (951,952,953,954)],4)
chk('h2',[W/f'holdout/nn_w5_e10_kd05t25w192_s{s}_holdout.npy' for s in (951,952,953,954)],4)
for k in range(5): chk(f'xf{k}',[W/f'holdout/nn_w5_e10_xf{k}_s{s}_holdout.npy' for s in (951,952,953)],3)
# C assembly on sampled rows
rs=np.sort(np.random.default_rng(0).choice(len(y),2_000_000,replace=False)); rs=np.concatenate([rs,rows])
L=lambda t: np.asarray(np.load(K/f'teacher_{t}.npy',mmap_mode='r')[rs],np.float64)
s11=sum(L(t) for t in 'abcdef'); m11=L('mean11'); print('C: mean11 == (a..f)/11: max|diff|',np.abs(s11/11-m11).max())
s25=s11+L('g1')+L('g2')+L('p1')+L('p2'); m25=L('mean25'); print('   mean25 == (a..f+g1+g2+p1+p2)/25: max|diff|',np.abs(s25/25-m25).max())
w8=(L('h1')+L('h2'))/8; m8=L('w8g1'); print('   w8g1 == (h1+h2)/8: max|diff|',np.abs(w8-m8).max())
mx=L('mix25w8'); print('   mix25w8 == 0.5 mean25 + 0.5 w8g1: max|diff|',np.abs(0.5*m25+0.5*m8-mx).max())
nr=len(rows)
print('   holdout NDCG: mean11',f'{eh(m11[-nr:]).mean():.5f}','mean25',f'{eh(m25[-nr:]).mean():.5f}','w8g1',f'{eh(m8[-nr:]).mean():.5f}','mix',f'{eh(mx[-nr:]).mean():.5f}','oof',f'{eh(L("oof")[-nr:]).mean():.5f}')
for t in ('mean11','mean25','w8g1','mix25w8','oof'):
    x=L(t)[:2_000_000]; print(f'   {t}: nan {int(np.isnan(x).sum())} mean {x.mean():.2f} sd {x.std():.3f} min {x.min():.1f} max {x.max():.1f}')
