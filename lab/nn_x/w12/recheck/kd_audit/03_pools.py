"""D: recompute headline pool numbers from prediction files. E: selected vs @last (selection on the holdout). FT-vs-TO per member. CPU only."""
import sys; sys.path.insert(0,'<repo>/src')
import numpy as np, json
from common import feed_meta, FLAT
from metrics import ndcg_per_feed
from stack_val import feed_rank_z
from pathlib import Path
R=Path('<repo>'); W=R/'nn_x/w5/preds'; P=R/'preds'; L=R/'nn_x/w5/logs'
fv,Mv,ov=feed_meta('val'); yv=np.load(FLAT/'val/cand_target.npy')
Mh,oh=np.load(W/'holdout/holdout_meta.npy'); Mh=Mh.astype(np.int64); oh=oh.astype(np.int64); yh=np.load(W/'holdout/holdout_y.npy')
ev=lambda s: ndcg_per_feed(np.asarray(s,np.float32),yv,ov,Mv); eh=lambda s: ndcg_per_feed(np.asarray(s,np.float32),yh,oh,Mh)
zv=lambda x: feed_rank_z(ov,Mv,np.asarray(x,np.float32))[:,1]; zh=lambda x: feed_rank_z(oh,Mh,np.asarray(x,np.float32))[:,1]
se=lambda d: d.std()/np.sqrt(len(d)); f=lambda d: f'{d.mean():+.5f}+-{se(d):.5f}'
S=(951,952,953,954)
def ld(v,s=''): return zv(np.load(W/f'nn_{v}{s}_trainonly_val.npy')), zh(np.load(W/f'holdout/nn_{v}{s}_holdout.npy'))
def arm(a,s=''): return [ld(f'w5_e10_{a}_s{x}',s) for x in S]
def pool(ms): return np.mean([m[0] for m in ms],0), np.mean([m[1] for m in ms],0)
A={a:arm(a) for a in ('base','kd05','kd05w192','kd05t25w192','kd08','kd10','kd03','kd05t25','kx05','kx08')}
AL={a:arm(a,'@last') for a in ('base','kd05','kd05w192','kd05t25w192','kd08','kx05')}
print('D: pools (selected checkpoints)')
for nm,ms in (('base x4',A['base']),('kd05 x4',A['kd05']),('kd05w192 x4',A['kd05w192']),('kd05t25w192 x4',A['kd05t25w192']),('W8',A['kd05w192']+A['kd05t25w192']),('kx05 x4',A['kx05'])):
    v,h=pool(ms); print(f'  {nm:16s} val {ev(v).mean():.5f} hold {eh(h).mean():.5f}')
w8=pool(A['kd05w192']+A['kd05t25w192']); k5=pool(A['kd05'])
print('  W8 vs kd05 pool: val',f(ev(w8[0])-ev(k5[0])),'hold',f(eh(w8[1])-eh(k5[1])))
for tag in ('W8kd','S16kd','Q2a8s3'):
    ft=np.load(P/f'ens_{tag}_ft_val.npy'); to=np.load(P/f'ens_{tag}_to_val.npy'); fin=0.5*zv(ft)+0.5*zv(to)
    print(f'  ens_{tag}: FT {ev(ft).mean():.5f} TO {ev(to).mean():.5f} final {ev(fin).mean():.5f} | final-TO {f(ev(fin)-ev(to))} FT-TO {f(ev(ft)-ev(to))}')
    if tag=='W8kd': W8fin,W8to,W8ft=fin,to,ft
    if tag=='Q2a8s3': Cfin=fin
print('  W8 final vs C3 final:',f(ev(W8fin)-ev(Cfin)))
for w in (0.0,0.15,0.25,0.35,0.5,0.65,1.0):
    q=w*zv(W8ft)+(1-w)*zv(W8to); print(f'   W8 w_ft {w:.2f}: val {ev(q).mean():.5f} vs 0.5: {f(ev(q)-ev(W8fin))}')
print('E: selected vs @last (members, mean over 4 seeds; pools)')
for a in ('base','kd05','kd05w192','kd05t25w192','kd08','kx05'):
    mv=np.mean([ev(x[0]) for x in A[a]],0); mh=np.mean([eh(x[1]) for x in A[a]],0); lv=np.mean([ev(x[0]) for x in AL[a]],0); lh=np.mean([eh(x[1]) for x in AL[a]],0)
    pv,ph=pool(A[a]); qv,qh=pool(AL[a])
    steps=[json.load(open(L/f'w5_e10_{a}_s{s}.json'))['best_step'] for s in S]
    print(f'  {a:12s} best steps {steps} | member sel val {mv.mean():.5f} hold {mh.mean():.5f} | @last val {lv.mean():.5f} hold {lh.mean():.5f} | sel-last member val {f(mv-lv)} hold {f(mh-lh)} | pool sel {ev(pv).mean():.5f}/{eh(ph).mean():.5f} last {ev(qv).mean():.5f}/{eh(qh).mean():.5f}')
print('   KD deltas vs base at @last (no selection on the holdout):')
bl=AL['base']
for a in ('kd05','kd05w192','kd05t25w192','kd08','kx05'):
    dv=np.mean([ev(x[0])-ev(b[0]) for x,b in zip(AL[a],bl)],0); dh=np.mean([eh(x[1])-eh(b[1]) for x,b in zip(AL[a],bl)],0)
    dvs=np.mean([ev(x[0])-ev(b[0]) for x,b in zip(A[a],A['base'])],0); dhs=np.mean([eh(x[1])-eh(b[1]) for x,b in zip(A[a],A['base'])],0)
    pa,pb=pool(AL[a]),pool(bl); sa,sb=pool(A[a]),pool(A['base'])
    print(f'   {a:12s} member @last val {f(dv)} hold {f(dh)} (selected: val {f(dvs)} hold {f(dhs)}) | 4-pool @last val {f(ev(pa[0])-ev(pb[0]))} hold {f(eh(pa[1])-eh(pb[1]))} (selected: {f(ev(sa[0])-ev(sb[0]))} / {f(eh(sa[1])-eh(sb[1]))})')
pa=pool(AL['kd05w192']+AL['kd05t25w192']); pk=pool(AL['kd05']); print('   W8 @last pool: val',f'{ev(pa[0]).mean():.5f}','hold',f'{eh(pa[1]).mean():.5f}','| vs kd05 @last pool: val',f(ev(pa[0])-ev(pk[0])),'hold',f(eh(pa[1])-eh(pk[1])))
print('FT vs TO per member (val OOF of n2cpv3s3 vs train-only val)')
for a in ('base','kd05','kd05w192','kd05t25w192','kd05t25'):
    ds=[];ft_l=[];to_l=[]
    for s in S:
        v=f'w5_e10_{a}_s{s}'; pf=P/f'nn_{v}_n2cpv3s3_val_oof.npy'
        if not pf.exists(): pf=W/f'nn_{v}_n2cpv3s3_val_oof.npy'
        if not pf.exists(): continue
        e1=ev(np.load(pf)); e0=ev(np.load(W/f'nn_{v}_trainonly_val.npy')); ds.append(e1-e0); ft_l.append(zv(np.load(pf))); to_l.append(zv(np.load(W/f'nn_{v}_trainonly_val.npy')))
    if ds:
        d=np.mean(ds,0); pf_,pt_=np.mean(ft_l,0),np.mean(to_l,0)
        print(f'  {a:12s} n={len(ds)} member FT-TO {f(d)} | pool FT {ev(pf_).mean():.5f} TO {ev(pt_).mean():.5f} FT-TO {f(ev(pf_)-ev(pt_))} | 0.5/0.5 {ev(0.5*zv(pf_)+0.5*zv(pt_)).mean():.5f} | corr(FT pool, TO pool) {np.corrcoef(pf_,pt_)[0,1]:.4f}')
    else: print('  ',a,'no FT files')
