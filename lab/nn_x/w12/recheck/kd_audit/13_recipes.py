"""Zero-GPU file-recipe reads (val; holdout where a TO-level read exists): asymmetric FT / TO member sets, W8 + pop1-view KD students, pooled FT-TO SE. CPU only."""
import sys; sys.path.insert(0,'<repo>/src')
import numpy as np
from common import feed_meta, FLAT
from metrics import ndcg_per_feed
from stack_val import feed_rank_z
from pathlib import Path
R=Path('<repo>'); P=R/'preds'; W=R/'nn_x/w5/preds'; AB=R/'nn_x/abl/preds'
fv,Mv,ov=feed_meta('val'); yv=np.load(FLAT/'val/cand_target.npy'); ft_,Mt,ot=feed_meta('test')
Mh,oh=np.load(W/'holdout/holdout_meta.npy'); Mh=Mh.astype(np.int64); oh=oh.astype(np.int64); yh=np.load(W/'holdout/holdout_y.npy')
ev=lambda s: ndcg_per_feed(np.asarray(s,np.float32),yv,ov,Mv); eh=lambda s: ndcg_per_feed(np.asarray(s,np.float32),yh,oh,Mh)
zv=lambda x: feed_rank_z(ov,Mv,np.asarray(x,np.float32))[:,1]; zh=lambda x: feed_rank_z(oh,Mh,np.asarray(x,np.float32))[:,1]; zt=lambda x: feed_rank_z(ot,Mt,np.asarray(x,np.float32))[:,1]
se=lambda d: d.std()/np.sqrt(len(d)); f=lambda d: f'{d.mean():+.5f}+-{se(d):.5f}'
L=lambda t,b,sp='val': np.load(P/f'ens_{t}_{b}_{sp}.npy')
w_ft,w_to,c_ft,c_to,s_ft,s_to=L('W8kd','ft'),L('W8kd','to'),L('Q2a8s3','ft'),L('Q2a8s3','to'),L('S16kd','ft'),L('S16kd','to')
fin=lambda a,b: 0.5*zv(a)+0.5*zv(b); e0=ev(fin(w_ft,w_to))
print('reference W8 final',f'{e0.mean():.5f}')
tw_ft,tw_to,tc_ft,tc_to=L('W8kd','ft','test'),L('W8kd','to','test'),L('Q2a8s3','ft','test'),L('Q2a8s3','to','test'); t0=0.5*zt(tw_ft)+0.5*zt(tw_to)
for nm,a,b,ta,tb in (('FT=C3, TO=W8',c_ft,w_to,tc_ft,tw_to),('FT=0.5z(C3)+0.5z(W8), TO=W8',0.5*zv(c_ft)+0.5*zv(w_ft),w_to,0.5*zt(tc_ft)+0.5*zt(tw_ft),tw_to),('FT=(28 C3+8 W8)/36, TO=W8',(28*c_ft+8*w_ft)/36,w_to,(28*tc_ft+8*tw_ft)/36,tw_to),
                     ('FT=S16, TO=W8',s_ft,w_to,L('S16kd','ft','test'),tw_to),('W8C5 (both branches 0.5/0.5)',0.5*zv(c_ft)+0.5*zv(w_ft),0.5*zv(c_to)+0.5*zv(w_to),0.5*zt(tc_ft)+0.5*zt(tw_ft),0.5*zt(tc_to)+0.5*zt(tw_to))):
    e=ev(fin(a,b)); tt=0.5*zt(ta)+0.5*zt(tb); print(f'  {nm:34s} val {e.mean():.5f} vs W8 {f(e-e0)} | FT branch alone {ev(a).mean():.5f} | test corr to W8 {np.corrcoef(tt,t0)[0,1]:.4f}')
# pooled FT-TO at member level: 16 students vs 28 hard-label members
st=[f'w5_e10_{a}_s{s}' for a in ('kd05','kd05t25','kd05w192','kd05t25w192') for s in range(951,955)]
hl=[f'r6_cd_m02_r{r}' for r in range(101,108)]+[f'r6_cd_m02_p70k_r{r}' for r in range(101,108)]+[f'v23_gru_pre2s_s{s}' for s in range(501,507)]+[f'abl_e10_pop1_s{s}' for s in range(951,959)]
d=lambda ms: np.mean([ev(np.load(P/f'nn_{m}_n2cpv3s3_val_oof.npy'))-ev(np.load(P/f'nn_{m}_trainonly_val.npy')) for m in ms],0)
ds,dh=d(st),d(hl); print('member FT-TO pooled: 16 KD students',f(ds),'| 8 wide',f(d(st[8:])),'| 28 hard-label members',f(dh),'| difference (students - hard)',f(ds-dh))
# W8 + pop1-view KD students (TO level, val + holdout)
def ld(v):
    D=AB if v.startswith('abl_') else W; return zv(np.load(D/f'nn_{v}_trainonly_val.npy')),zh(np.load(D/f'holdout/nn_{v}_holdout.npy'))
w8=[ld(f'w5_e10_{a}_s{s}') for a in ('kd05w192','kd05t25w192') for s in range(951,955)]; pk=[ld(f'abl_e10_pop1kd05t25_s{s}') for s in range(951,955)]; n4=[ld(f'w5_e10_kd05t25_s{s}') for s in range(951,955)]
pool=lambda ms: (np.mean([m[0] for m in ms],0),np.mean([m[1] for m in ms],0))
a,b,c=pool(w8),pool(w8+pk),pool(w8+n4)
print('TO level: W8 + pop1kd x4 vs W8: val',f(ev(b[0])-ev(a[0])),'hold',f(eh(b[1])-eh(a[1])),'| control W8 + kd05t25 x4 (same teacher, standard view): val',f(ev(c[0])-ev(a[0])),'hold',f(eh(c[1])-eh(a[1])),'| pop1kd-vs-control: val',f(ev(b[0])-ev(c[0])),'hold',f(eh(b[1])-eh(c[1])))
print('   corr(pop1kd member, wide student) val',f'{np.mean([np.corrcoef(x[0],y_[0])[0,1] for x in pk for y_ in w8]):.4f}','| corr(kd05t25 member, wide student)',f'{np.mean([np.corrcoef(x[0],y_[0])[0,1] for x in n4 for y_ in w8]):.4f}','| pop1kd member val',f'{np.mean([ev(x[0]).mean() for x in pk]):.5f}','hold',f'{np.mean([eh(x[1]).mean() for x in pk]):.5f}')
