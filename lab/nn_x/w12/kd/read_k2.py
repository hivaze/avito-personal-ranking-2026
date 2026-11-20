import sys; sys.path.insert(0,'<repo>/src')
from common import *
from metrics import ndcg_per_feed
from stack_val import feed_rank_z
from pathlib import Path
P=Path('<repo>/preds'); fv,Mv,ov=feed_meta('val'); yv=np.load(FLAT/'val/cand_target.npy')
z=lambda x: feed_rank_z(ov,Mv,np.asarray(x,np.float32))[:,1]; ev=lambda s: ndcg_per_feed(np.asarray(s,np.float32),yv,ov,Mv); se=lambda d: d.std()/np.sqrt(len(d)); f=lambda d: f'{d.mean():+.5f}+-{se(d):.5f}'
def pool(ms): return 0.5*z(np.mean([z(np.load(P/f'nn_{m}_n2cpv3s3_val_oof.npy')) for m in ms],0))+0.5*z(np.mean([z(np.load(P/f'nn_{m}_trainonly_val.npy')) for m in ms],0))
c3=0.5*z(np.load(P/'ens_Q2a8s3_ft_val.npy'))+0.5*z(np.load(P/'ens_Q2a8s3_to_val.npy')); e0=ev(c3)
k5=[f'w5_e10_kd05_s{s}' for s in range(951,955)]; k8=[f'w5_e10_kd08_s{s}' for s in range(951,955)]
for nm,ms in (('kd05x4',k5),('kd08x4',k8),('kd05+kd08 x8',k5+k8)):
    p=pool(ms); e=ev(p); print(f'{nm}: val {e.mean():.5f} vs C3 {f(e-e0)} | corr to C3 {np.corrcoef(p,c3)[0,1]:.4f}')
    for w in (0.3,0.5):
        q=(1-w)*z(c3)+w*z(p); d=ev(q)-e0; print(f'   {1-w:.1f} C3 + {w:.1f} {nm}: {ev(q).mean():.5f} {f(d)} | M<50 {f(d[Mv<50])} M>=50 {f(d[Mv>=50])}')
