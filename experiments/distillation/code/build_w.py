# Step 3: the first files made of students only. W8 = the 8 width-192 students (4 per teacher), S16 = all 16
# students; each file is 0.5 z(fine-tuned half) + 0.5 z(train-only half), member-equal within a half.
# Run as (docstring): build_w.py [build]. Without 'build' it prints the val read-out only.
# It wrote preds/ens_<TAG>_{ft,to}_{val,test}.npy and subs/cand_<TAG>_r2.parquet.
# W8 scored 0.394486 on the public board against 0.393454 for the best click-label file; S16 scored 0.394078.
# Kept as it ran: paths and names refer to the lab tree (see lab/ and docs/glossary.md).
# Original: lab/nn_x/w12/kd/build_w.py
"""Round-2 student candidates. W8 = 8 d192 students (kd05w192 + kd05t25w192); S16 = all 16 tf students; W8C5 = 0.5 z(C3) + 0.5 z(W8) per branch.
final = 0.5 z(FT) + 0.5 z(TO). Usage: build_w.py [build]"""
import sys; sys.path.insert(0,'<repo>/src')
from common import *
from metrics import ndcg_per_feed
from stack_val import feed_rank_z
from make_sub import make_sub
from pathlib import Path
build='build' in sys.argv
P=Path('<repo>/preds'); S=Path('<repo>/subs'); sd=(951,952,953,954)
G={a:[f'w5_e10_{a}_s{s}' for s in sd] for a in ('kd05','kd05t25','kd05w192','kd05t25w192')}
fv,Mv,ov=feed_meta('val'); ft_,Mt,ot=feed_meta('test'); yv=np.load(FLAT/'val/cand_target.npy')
Z={'val':lambda x: feed_rank_z(ov,Mv,np.asarray(x,np.float32))[:,1],'test':lambda x: feed_rank_z(ot,Mt,np.asarray(x,np.float32))[:,1]}
ev=lambda s: ndcg_per_feed(np.asarray(s,np.float32),yv,ov,Mv); se=lambda d: d.std()/np.sqrt(len(d)); f=lambda d: f'{d.mean():+.5f}+-{se(d):.5f}'
B={}
for sp,ftn,ton in (('val','n2cpv3s3_val_oof','trainonly_val'),('test','n2cpv3s3_test','trainonly_test')):
    z=Z[sp]; c_ft=np.load(P/f'ens_Q2a8s3_ft_{sp}.npy').astype(np.float64); c_to=np.load(P/f'ens_Q2a8s3_to_{sp}.npy').astype(np.float64)
    pool=lambda arms: (np.mean([z(np.load(P/f'nn_{m}_{ftn}.npy')) for a in arms for m in G[a]],0), np.mean([z(np.load(P/f'nn_{m}_{ton}.npy')) for a in arms for m in G[a]],0))
    w8=pool(['kd05w192','kd05t25w192']); s16=pool(list(G))
    B[sp]=dict(C3=(c_ft,c_to),W8=w8,S16=s16,W8C5=(0.5*z(c_ft)+0.5*z(w8[0]),0.5*z(c_to)+0.5*z(w8[1])),S16C5=(0.5*z(c_ft)+0.5*z(s16[0]),0.5*z(c_to)+0.5*z(s16[1])),
               A=pool(['kd05w192']),Bq=pool(['kd05t25w192']),T=pool(['kd05t25']))
fin=lambda sp,k: 0.5*Z[sp](B[sp][k][0])+0.5*Z[sp](B[sp][k][1])
e0=ev(fin('val','C3')); print(f'C3 val {e0.mean():.5f} | FT {ev(B["val"]["C3"][0]).mean():.5f} TO {ev(B["val"]["C3"][1]).mean():.5f}')
t0=fin('test','C3')
for k in ('A','Bq','T','W8','S16','W8C5','S16C5'):
    e=ev(fin('val',k)); d=e-e0
    print(f'{k:6s} val {e.mean():.5f} delta {f(d)} | FT {ev(B["val"][k][0]).mean():.5f} TO {ev(B["val"][k][1]).mean():.5f} | M<50 {f(d[Mv<50])} M>=50 {f(d[Mv>=50])} | test corr C3 {np.corrcoef(fin("test",k),t0)[0,1]:.4f}')
if build:
    for k in ('W8','S16','W8C5'):
        tag=f'{k}kd'; out=S/f'cand_{tag}_r2.parquet'; assert not out.exists() and not list(P.glob(f'ens_{tag}_*'))
        for sp in ('val','test'):
            np.save(P/f'ens_{tag}_ft_{sp}.npy',B[sp][k][0].astype(np.float32)); np.save(P/f'ens_{tag}_to_{sp}.npy',B[sp][k][1].astype(np.float32))
        make_sub(fin('test',k).astype(np.float32),str(out)); print('written',out)
