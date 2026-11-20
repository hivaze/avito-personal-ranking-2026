"""Post-hoc multi-arm student pools (TO level). val + holdout, vs C3 TO and vs kd05x4."""
import sys; sys.path.insert(0,'<repo>/src')
from common import *
from metrics import ndcg_per_feed
from stack_val import feed_rank_z
from pathlib import Path
W=Path('<repo>/nn_x/w5/preds'); AB=Path('<repo>/nn_x/abl/preds'); P=Path('<repo>/preds')
fv,Mv,ov=feed_meta('val'); yv=np.load(FLAT/'val/cand_target.npy')
Mh,oh=np.load(W/'holdout/holdout_meta.npy'); Mh=Mh.astype(np.int64); oh=oh.astype(np.int64); yh=np.load(W/'holdout/holdout_y.npy')
ev=lambda s: ndcg_per_feed(np.asarray(s,np.float32),yv,ov,Mv); eh=lambda s: ndcg_per_feed(np.asarray(s,np.float32),yh,oh,Mh)
zv=lambda x: feed_rank_z(ov,Mv,np.asarray(x,np.float32))[:,1]; zh=lambda x: feed_rank_z(oh,Mh,np.asarray(x,np.float32))[:,1]
se=lambda d: d.std()/np.sqrt(len(d)); f=lambda d: f'{d.mean():+.5f}+-{se(d):.5f}'
def ld(v):
    D=AB if v.startswith('abl_') else W
    h=D/f'holdout/nn_{v}_holdout.npy'
    return zv(np.load(D/f'nn_{v}_trainonly_val.npy')), (zh(np.load(h)) if h.exists() else None)
S=[951,952,953,954]
arms={a:[ld(f'w5_e10_{a}_s{s}') for s in S] for a in ['kd05','kd05w192','kd05t25','kd05t25w192','kd05g2w192','kd03g2w192']}
arms['pop1kd']=[ld(f'abl_e10_pop1kd05t25_s{s}') for s in S]
arms['pop1']=[ld(f'abl_e10_pop1_s{s}') for s in S]
c3=np.load(P/'ens_Q2a8s3_to_val.npy'); e0=ev(c3); zc=zv(c3)
def pool(names):
    m=[x for n in names for x in arms[n]]
    v=np.mean([x[0] for x in m],0); hs=[x[1] for x in m]
    h=np.mean(hs,0) if all(x is not None for x in hs) else None
    return v,h
ref=pool(['kd05'])
for names in [['kd05w192','kd05t25w192'],['kd05g2w192','kd03g2w192'],['kd05w192','kd05t25w192','kd05g2w192','kd03g2w192'],['kd05w192','kd05t25w192','kd03g2w192']]:
    v,h=pool(names); e=ev(v)
    hs=f'{eh(h).mean():.5f} ({f(eh(h)-eh(ref[1]))} vs kd05)' if h is not None else 'n/a'
    print(f'{"+".join(names):55s} val {e.mean():.5f} vs C3TO {f(e-e0)} | hold {hs} | corr C3 {np.corrcoef(v,c3)[0,1]:.4f} | .5/.5 C3: {f(ev(0.5*zc+0.5*zv(v))-e0)} .7/.3: {f(ev(0.7*zc+0.3*zv(v))-e0)}')
