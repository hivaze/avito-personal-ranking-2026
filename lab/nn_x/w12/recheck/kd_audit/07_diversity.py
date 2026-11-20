"""Student diversity: within-arm vs cross-arm (different teacher) correlation, same-seed coupling with the teacher's base members,
 and equal-size pools: mixed-teacher 4-pools vs same-teacher 4-pools (val + holdout). CPU only."""
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
S=(951,952,953,954)
arms=['base','kd03','kd05','kd08','kd10','kd05t25','kd05w192','kd05t25w192','kx05','kx08','kd05g2w192']
A={a:[(zv(np.load(W/f'nn_w5_e10_{a}_s{s}_trainonly_val.npy')),zh(np.load(W/f'holdout/nn_w5_e10_{a}_s{s}_holdout.npy'))) for s in S] for a in arms}
def cc(x,y,i): return np.corrcoef(x[i],y[i])[0,1]
print('within-arm member corr (val / holdout), member mean, 4-pool, pool lift over member mean')
for a in arms:
    c=[(cc(A[a][i],A[a][j],0),cc(A[a][i],A[a][j],1)) for i,j in itertools.combinations(range(4),2)]; c=np.mean(c,0)
    mv=np.mean([ev(x[0]).mean() for x in A[a]]); mh=np.mean([eh(x[1]).mean() for x in A[a]])
    pv=ev(np.mean([x[0] for x in A[a]],0)).mean(); ph=eh(np.mean([x[1] for x in A[a]],0)).mean()
    print(f'  {a:12s} corr {c[0]:.4f} / {c[1]:.4f} | member {mv:.5f} / {mh:.5f} | pool {pv:.5f} / {ph:.5f} | lift {pv-mv:+.5f} / {ph-mh:+.5f}')
print('cross-arm corr, different seeds vs same seed (val / holdout)')
for a,b in (('kd05w192','kd05t25w192'),('kd05','kd05t25'),('kd05','kd05w192'),('kd05w192','kx05'),('kd05w192','kd05g2w192'),('kd05','base'),('kd05w192','base'),('kd05','kd08')):
    d=np.mean([(cc(A[a][i],A[b][j],0),cc(A[a][i],A[b][j],1)) for i in range(4) for j in range(4) if i!=j],0); s=np.mean([(cc(A[a][i],A[b][i],0),cc(A[a][i],A[b][i],1)) for i in range(4)],0)
    print(f'  {a:12s} x {b:12s} diff-seed {d[0]:.4f} / {d[1]:.4f} | same-seed {s[0]:.4f} / {s[1]:.4f}')
print('equal-size 4-pools: same-teacher vs mixed-teacher (2+2, all 36 combos averaged; paired per feed vs the mean of the two same-teacher 4-pools)')
def pool(ms): return np.mean([m[0] for m in ms],0), np.mean([m[1] for m in ms],0)
for a,b in (('kd05w192','kd05t25w192'),('kd05','kd05t25'),('kd05w192','kx05'),('kd05','kd05w192')):
    pa,pb=pool(A[a]),pool(A[b]); ea,eb=(ev(pa[0])+ev(pb[0]))/2,(eh(pa[1])+eh(pb[1]))/2
    mv=[];mh=[]
    for ia in itertools.combinations(range(4),2):
        for ib in itertools.combinations(range(4),2):
            p=pool([A[a][i] for i in ia]+[A[b][i] for i in ib]); mv.append(ev(p[0])); mh.append(eh(p[1]))
    mv=np.mean(mv,0); mh=np.mean(mh,0)
    # same-arm 2+2 is the same 4-pool; size-matched control: all 2-pools vs 4-pools not needed
    p8=pool(A[a]+A[b])
    print(f'  {a}+{b}: same-teacher 4-pools val {ea.mean():.5f} hold {eb.mean():.5f} | mixed 2+2 val {mv.mean():.5f} hold {mh.mean():.5f} | mixed-same val {f(mv-ea)} hold {f(mh-eb)} | 8-pool val {ev(p8[0]).mean():.5f} hold {eh(p8[1]).mean():.5f} (8 - same4: val {f(ev(p8[0])-ea)} hold {f(eh(p8[1])-eb)})')
# pool size curve within one arm (k=1..4) to extrapolate the pure size effect
print('pool-size curve (mean over subsets), val / holdout')
for a in ('base','kd05','kd05w192','kd05t25w192'):
    out=[]
    for k in (1,2,3,4):
        v=[];h=[]
        for c in itertools.combinations(range(4),k):
            p=pool([A[a][i] for i in c]); v.append(ev(p[0]).mean()); h.append(eh(p[1]).mean())
        out.append(f'k{k} {np.mean(v):.5f}/{np.mean(h):.5f}')
    print(f'  {a:12s} '+'  '.join(out))
