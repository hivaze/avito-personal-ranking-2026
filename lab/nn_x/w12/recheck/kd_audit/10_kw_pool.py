"""Were the kw_ levers (closed on a 2-seed MEMBER screen) also null at POOL level? 2-member pools vs the kd05w192 2-member pool (same seeds), + diversity,
 + the [W4 + arm2] vs [W4 + 2 more kd05t25w192] style 'adds to the pool' read. Also W8 vs N8 at equal size on the holdout. CPU only."""
import sys; sys.path.insert(0,'<repo>/src')
import numpy as np, itertools
from common import feed_meta, FLAT
from metrics import ndcg_per_feed
from stack_val import feed_rank_z
from pathlib import Path
R=Path('<repo>'); W=R/'nn_x/w5/preds'
fv,Mv,ov=feed_meta('val'); yv=np.load(FLAT/'val/cand_target.npy')
Mh,oh=np.load(W/'holdout/holdout_meta.npy'); Mh=Mh.astype(np.int64); oh=oh.astype(np.int64); yh=np.load(W/'holdout/holdout_y.npy')
ev=lambda s: ndcg_per_feed(np.asarray(s,np.float32),yv,ov,Mv); eh=lambda s: ndcg_per_feed(np.asarray(s,np.float32),yh,oh,Mh)
zv=lambda x: feed_rank_z(ov,Mv,np.asarray(x,np.float32))[:,1]; zh=lambda x: feed_rank_z(oh,Mh,np.asarray(x,np.float32))[:,1]
se=lambda d: d.std()/np.sqrt(len(d)); f=lambda d: f'{d.mean():+.5f}+-{se(d):.5f}'
def ld(a,s,suf=''):
    pv=W/f'nn_w5_e10_{a}_s{s}{suf}_trainonly_val.npy'; ph=W/f'holdout/nn_w5_e10_{a}_s{s}{suf}_holdout.npy'
    return (zv(np.load(pv)),zh(np.load(ph))) if pv.exists() and ph.exists() else None
pool=lambda ms: (np.mean([m[0] for m in ms],0),np.mean([m[1] for m in ms],0))
S4=(951,952,953,954)
ref={s:ld('kd05w192',s) for s in S4}; ref25={s:ld('kd05t25w192',s) for s in S4}; n5={s:ld('kd05',s) for s in S4}; n25={s:ld('kd05t25',s) for s in S4}
w8=pool(list(ref.values())+list(ref25.values())); n8=pool(list(n5.values())+list(n25.values()))
print('W8 vs N8 (equal size 8, TO): val',f(ev(w8[0])-ev(n8[0])),'hold',f(eh(w8[1])-eh(n8[1])))
print('arm: member delta vs kd05w192 (val / hold) | 2-pool delta vs kd05w192 2-pool (val / hold) | arm-arm corr (ref corr) | [W8 + arm x2] - W8 (val / hold) ; control [W8 + 2 more same-recipe] not available')
for suf in ('','@last'):
  print('checkpoint',suf or 'selected')
  for a in ('kw_drop0','kw_e20','kw_hl2','kw_ff4','kw_id','kd05t25w256','kd05t25w192b3','kwsub5','kwsub2','kwT07','kwT15','kd05w224','kd05w192gru','kdmixw192'):
    seeds=[s for s in S4 if ld(a,s,suf) is not None]
    if len(seeds)<2: print(f'  {a:14s} not finished ({len(seeds)} seeds)'); continue
    A=[ld(a,s,suf) for s in seeds]; rf=ref25 if 't25' in a else ref
    B=[ld('kd05t25w192' if 't25' in a else 'kd05w192',s,suf) for s in seeds]
    dv=np.mean([ev(x[0])-ev(b[0]) for x,b in zip(A,B)],0); dh=np.mean([eh(x[1])-eh(b[1]) for x,b in zip(A,B)],0)
    pa,pb=pool(A),pool(B); ca=np.mean([np.corrcoef(A[i][0],A[j][0])[0,1] for i,j in itertools.combinations(range(len(A)),2)]); cb=np.mean([np.corrcoef(B[i][0],B[j][0])[0,1] for i,j in itertools.combinations(range(len(B)),2)])
    cx=np.mean([np.corrcoef(A[i][0],B[j][0])[0,1] for i in range(len(A)) for j in range(len(B)) if i!=j])
    q=pool(list(ref.values())+list(ref25.values())+A)
    print(f'  {a:14s} n={len(seeds)} member {f(dv)} / {f(dh)} | pool {f(ev(pa[0])-ev(pb[0]))} / {f(eh(pa[1])-eh(pb[1]))} | corr {ca:.4f} (ref {cb:.4f}; arm x ref diff-seed {cx:.4f}) | W8+arm - W8: {f(ev(q[0])-ev(w8[0]))} / {f(eh(q[1])-eh(w8[1]))}')
