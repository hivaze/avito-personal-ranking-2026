"""KD candidates on C3 (Q2a8s3). K1 = member-equal C3(28) + kd05 x4; K2 = 0.7 z(C3) + 0.3 z(kd05 pool) per branch; final 0.5 z(FT) + 0.5 z(TO).
Usage: build_k.py [arm=kd05] [build]  -> prints val read; with 'build' writes preds/ens_<TAG>_{ft,to}_{val,test}.npy and subs/cand_<TAG>.parquet"""
import sys; sys.path.insert(0,'<repo>/src')
from common import *
from metrics import ndcg_per_feed
from stack_val import feed_rank_z
from make_sub import make_sub
from pathlib import Path
arm=sys.argv[1] if len(sys.argv)>1 else 'kd05'; build='build' in sys.argv
P=Path('<repo>/preds'); S=Path('<repo>/subs'); ms=[f'w5_e10_{arm}_s{s}' for s in (951,952,953,954)]
fv,Mv,ov=feed_meta('val'); ft_,Mt,ot=feed_meta('test'); yv=np.load(FLAT/'val/cand_target.npy')
Z={'val':lambda x: feed_rank_z(ov,Mv,np.asarray(x,np.float32))[:,1],'test':lambda x: feed_rank_z(ot,Mt,np.asarray(x,np.float32))[:,1]}
ev=lambda s: ndcg_per_feed(np.asarray(s,np.float32),yv,ov,Mv); se=lambda d: d.std()/np.sqrt(len(d)); f=lambda d: f'{d.mean():+.5f}+-{se(d):.5f}'
B={}
for sp,ftn,ton in (('val','n2cpv3s3_val_oof','trainonly_val'),('test','n2cpv3s3_test','trainonly_test')):
    z=Z[sp]; c_ft=np.load(P/f'ens_Q2a8s3_ft_{sp}.npy').astype(np.float64); c_to=np.load(P/f'ens_Q2a8s3_to_{sp}.npy').astype(np.float64)
    k_ft=np.mean([z(np.load(P/f'nn_{m}_{ftn}.npy')) for m in ms],0); k_to=np.mean([z(np.load(P/f'nn_{m}_{ton}.npy')) for m in ms],0)
    B[sp]=dict(C3=(c_ft,c_to),K1=((28*c_ft+4*k_ft)/32,(28*c_to+4*k_to)/32),K2=(0.7*z(c_ft)+0.3*z(k_ft),0.7*z(c_to)+0.3*z(k_to)),K5=(0.5*z(c_ft)+0.5*z(k_ft),0.5*z(c_to)+0.5*z(k_to)),KD=(k_ft,k_to))
fin=lambda sp,k: 0.5*Z[sp](B[sp][k][0])+0.5*Z[sp](B[sp][k][1])
e0=ev(fin('val','C3')); print(f'C3 val {e0.mean():.5f} | FT branch {ev(B["val"]["C3"][0]).mean():.5f} TO branch {ev(B["val"]["C3"][1]).mean():.5f}')
import polars as pl; day=(pl.read_parquet(FLAT/'val/feeds.parquet')['timestamp'].to_numpy()+3*3600)//86400; d0=np.unique(day)[np.argsort(-np.bincount(np.unique(day,return_inverse=True)[1]))[:2]]
for k in ('KD','K1','K2','K5'):
    e=ev(fin('val',k)); d=e-e0
    print(f'{k} ({arm}): val {e.mean():.5f} delta {f(d)} | FT branch {ev(B["val"][k][0]).mean():.5f} TO branch {ev(B["val"][k][1]).mean():.5f} | M<50 {f(d[Mv<50])} M>=50 {f(d[Mv>=50])} | days '+' '.join(f(d[day==x]) for x in sorted(d0))+f' | test corr to C3 {np.corrcoef(fin("test",k),fin("test","C3"))[0,1]:.4f}')
if build:
    for k in ('KD','K2','K5'):
        tag=f'{k}{arm}'; out=S/f'cand_{tag}_C3kd.parquet'; assert not out.exists() and not list(P.glob(f'ens_{tag}_*'))
        for sp in ('val','test'):
            np.save(P/f'ens_{tag}_ft_{sp}.npy',B[sp][k][0].astype(np.float32)); np.save(P/f'ens_{tag}_to_{sp}.npy',B[sp][k][1].astype(np.float32))
        make_sub((0.5*Z['test'](np.load(P/f'ens_{tag}_ft_test.npy'))+0.5*Z['test'](np.load(P/f'ens_{tag}_to_test.npy'))).astype(np.float32),str(out)); print('written',out)
