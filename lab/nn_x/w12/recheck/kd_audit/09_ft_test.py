"""FT vs TO per member for hard-label families vs KD students (val OOF); W8 vs narrow-8 at equal pool size; test-side sanity of the W8 files. CPU only."""
import sys; sys.path.insert(0,'<repo>/src')
import numpy as np, polars as pl
from common import feed_meta, FLAT
from metrics import ndcg_per_feed
from stack_val import feed_rank_z
from pathlib import Path
R=Path('<repo>'); P=R/'preds'; W=R/'nn_x/w5/preds'
fv,Mv,ov=feed_meta('val'); yv=np.load(FLAT/'val/cand_target.npy'); ft_,Mt,ot=feed_meta('test')
ev=lambda s: ndcg_per_feed(np.asarray(s,np.float32),yv,ov,Mv); zv=lambda x: feed_rank_z(ov,Mv,np.asarray(x,np.float32))[:,1]; zt=lambda x: feed_rank_z(ot,Mt,np.asarray(x,np.float32))[:,1]
se=lambda d: d.std()/np.sqrt(len(d)); f=lambda d: f'{d.mean():+.5f}+-{se(d):.5f}'
fam={'r6_cd_m02':[f'r6_cd_m02_r{r}' for r in range(101,108)],'r6_cd_m02_p70k':[f'r6_cd_m02_p70k_r{r}' for r in range(101,108)],'v23_gru_pre2s':[f'v23_gru_pre2s_s{s}' for s in range(501,507)],
     'abl_e10_pop1':[f'abl_e10_pop1_s{s}' for s in range(951,959)],'kd05':[f'w5_e10_kd05_s{s}' for s in range(951,955)],'kd08':[f'w5_e10_kd08_s{s}' for s in range(951,955)],'kd05t25':[f'w5_e10_kd05t25_s{s}' for s in range(951,955)],
     'kd05w192':[f'w5_e10_kd05w192_s{s}' for s in range(951,955)],'kd05t25w192':[f'w5_e10_kd05t25w192_s{s}' for s in range(951,955)]}
print('family: member TO, member FT, member FT-TO (paired, mean over members) | pool TO, pool FT, pool 0.5/0.5 | corr(FT member, TO member)')
Z={}
for k,ms in fam.items():
    d=[];to=[];ft=[];cs=[]
    for m in ms:
        a=np.load(P/f'nn_{m}_n2cpv3s3_val_oof.npy'); b=np.load(P/f'nn_{m}_trainonly_val.npy'); za,zb=zv(a),zv(b)
        d.append(ev(a)-ev(b)); to.append(zb); ft.append(za); cs.append(np.corrcoef(za,zb)[0,1])
    d=np.mean(d,0); pt,pf=np.mean(to,0),np.mean(ft,0); Z[k]=(pf,pt)
    print(f'  {k:16s} n={len(ms)} member TO {np.mean([ev(x).mean() for x in to]):.5f} FT {np.mean([ev(x).mean() for x in ft]):.5f} FT-TO {f(d)} | pool TO {ev(pt).mean():.5f} FT {ev(pf).mean():.5f} mix {ev(0.5*zv(pf)+0.5*zv(pt)).mean():.5f} | corr {np.mean(cs):.4f}')
w8=[(Z['kd05w192'][i]+Z['kd05t25w192'][i])/2 for i in (0,1)]; n8=[(Z['kd05'][i]+Z['kd05t25'][i])/2 for i in (0,1)]
fin=lambda p: 0.5*zv(p[0])+0.5*zv(p[1])
print('W8 (wide) vs N8 (narrow kd05+kd05t25) at equal pool size 8: TO',f(ev(w8[1])-ev(n8[1])),'| FT',f(ev(w8[0])-ev(n8[0])),'| final',f(ev(fin(w8))-ev(fin(n8))))
# test-side sanity
print('test-side sanity (label-free):')
for tag in ('W8kd','S16kd','Q2a8s3'):
    a,b=np.load(P/f'ens_{tag}_ft_test.npy'),np.load(P/f'ens_{tag}_to_test.npy'); av,bv=np.load(P/f'ens_{tag}_ft_val.npy'),np.load(P/f'ens_{tag}_to_val.npy')
    print(f'  ens_{tag}: test rows {len(a)} (expected {int(Mt.sum())}) nan {int(np.isnan(a).sum()+np.isnan(b).sum())} | corr(FT,TO) test {np.corrcoef(zt(a),zt(b))[0,1]:.4f} val {np.corrcoef(zv(av),zv(bv))[0,1]:.4f}')
a,b=np.load(P/'ens_W8kd_ft_test.npy'),np.load(P/'ens_W8kd_to_test.npy'); c,d_=np.load(P/'ens_Q2a8s3_ft_test.npy'),np.load(P/'ens_Q2a8s3_to_test.npy')
w8t=0.5*zt(a)+0.5*zt(b); c3t=0.5*zt(c)+0.5*zt(d_); print('  corr(W8 final, C3 final) on test',f'{np.corrcoef(w8t,c3t)[0,1]:.4f}','| val',f'{np.corrcoef(fin((np.load(P/"ens_W8kd_ft_val.npy"),np.load(P/"ens_W8kd_to_val.npy"))),fin((np.load(P/"ens_Q2a8s3_ft_val.npy"),np.load(P/"ens_Q2a8s3_to_val.npy"))))[0,1]:.4f}')
# member-level test corr of student FT vs TO (alignment check of the copied files)
for m in ('w5_e10_kd05w192_s951','w5_e10_kd05t25w192_s954'):
    x,y_=np.load(P/f'nn_{m}_n2cpv3s3_test.npy'),np.load(P/f'nn_{m}_trainonly_test.npy'); xv,yv_=np.load(P/f'nn_{m}_n2cpv3s3_val_oof.npy'),np.load(P/f'nn_{m}_trainonly_val.npy')
    print(f'  {m}: corr(FT,TO) test {np.corrcoef(zt(x),zt(y_))[0,1]:.4f} val {np.corrcoef(zv(xv),zv(yv_))[0,1]:.4f}; preds/ copy == w5/preds: {bool((np.load(W/f"nn_{m}_trainonly_test.npy")==y_).all())}')
sub=pl.read_parquet(R/'subs/cand_W8kd_r2.parquet'); print('  sub columns',sub.columns,'rows',len(sub))
