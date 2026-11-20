"""W8 TO pool (8 wide students) vs its teacher pool tf11 and vs C3 TO, val + holdout, paired; FT vs TO branch of the final files."""
import sys; sys.path.insert(0,'<repo>/src')
from pathlib import Path
import numpy as np
from common import FLAT, feed_meta
from metrics import ndcg_per_feed
from stack_val import feed_rank_z
W=Path('<repo>/nn_x/w5/preds'); R6=Path('<repo>/nn_x/r6/preds'); P=Path('<repo>/preds')
fv,Mv,ov=feed_meta('val'); yv=np.load(FLAT/'val/cand_target.npy')
Mh,oh=np.load(W/'holdout/holdout_meta.npy'); Mh=Mh.astype(np.int64); oh=oh.astype(np.int64); yh=np.load(W/'holdout/holdout_y.npy')
ev=lambda s: ndcg_per_feed(np.asarray(s,np.float32),yv,ov,Mv); eh=lambda s: ndcg_per_feed(np.asarray(s,np.float32),yh,oh,Mh)
zv=lambda x: feed_rank_z(ov,Mv,np.asarray(x,np.float32))[:,1]; zh=lambda x: feed_rank_z(oh,Mh,np.asarray(x,np.float32))[:,1]
se=lambda d: d.std()/np.sqrt(len(d)); f=lambda d: f'{d.mean():+.5f}+-{se(d):.5f}'
ld=lambda v,D=W: (zv(np.load(D/f'nn_{v}_trainonly_val.npy')), zh(np.load(D/f'holdout/nn_{v}_holdout.npy')))
pool=lambda ms: (np.mean([m[0] for m in ms],0), np.mean([m[1] for m in ms],0))
S=[951,952,953,954]
T=pool([ld(f'r6_cd_m02_r{r}',R6) for r in range(101,108)]+[ld(f'w5_e10_base_s{s}') for s in S])
w8m=[ld(f'w5_e10_{a}_s{s}') for a in ('kd05w192','kd05t25w192') for s in S]; W8=pool(w8m)
print(f'W8 TO pool val {ev(W8[0]).mean():.5f} hold {eh(W8[1]).mean():.5f} | teacher tf11 val {ev(T[0]).mean():.5f} hold {eh(T[1]).mean():.5f}')
print(f'W8 - teacher tf11: val {f(ev(W8[0])-ev(T[0]))} hold {f(eh(W8[1])-eh(T[1]))}')
mem_v=np.mean([ev(m[0]) for m in w8m],0); mem_h=np.mean([eh(m[1]) for m in w8m],0)
print(f'mean single wide student: val {mem_v.mean():.5f} hold {mem_h.mean():.5f}; teacher - single student: val {f(ev(T[0])-mem_v)} hold {f(eh(T[1])-mem_h)}; W8 - single: val {f(ev(W8[0])-mem_v)} hold {f(eh(W8[1])-mem_h)}')
iu=np.triu_indices(8,1); print(f'W8 member-member val corr {np.corrcoef([m[0] for m in w8m])[iu].mean():.4f}; corr(W8 pool, teacher pool) val {np.corrcoef(W8[0],T[0])[0,1]:.4f}')
for nm in ('W8kd','Q2a8s3'):
    ft,to=zv(np.load(P/f'ens_{nm}_ft_val.npy')),zv(np.load(P/f'ens_{nm}_to_val.npy')); e_to=ev(to)
    print(f'{nm}: FT branch {ev(ft).mean():.5f} TO branch {e_to.mean():.5f} FT-TO {f(ev(ft)-e_to)} corr {np.corrcoef(ft,to)[0,1]:.4f} | '+' '.join(f'w_ft {w}: {f(ev((1-w)*to+w*ft)-e_to)}' for w in (0.25,0.5,0.75)))
    for nm2,msk in (('M<50',Mv<50),('M>=50',Mv>=50)):
        d=ev(ft)-e_to; d5=ev(0.5*to+0.5*ft)-e_to; print(f'   {nm2}: FT-TO {d[msk].mean():+.5f}+-{se(d[msk]):.5f}; 0.5/0.5 - TO {d5[msk].mean():+.5f}+-{se(d5[msk]):.5f}')
