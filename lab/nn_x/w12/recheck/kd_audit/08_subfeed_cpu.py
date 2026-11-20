"""CPU forward on VAL feeds (fp32): (1) is a full-feed teacher logit a valid target for a sub-feed? KL( softmax(full-feed logits restricted to the kept set)
 || softmax(sub-feed logits) ) of the SAME model, with the exact --aug_cand rule (cd_arms.augment_pool_batch), vs the seed-to-seed KL and the
 teacher-to-student KL; (2) domain token: random-like vs production; (3) K9 aux-head logit magnitude (bf16 quantisation step).
 Val labels are used only to mimic 'positives are always kept' and for NDCG diagnostics. No GPU."""
import sys, os
sys.path.insert(0,'<repo>/nn_x/w5')
import numpy as np, torch, torch.nn.functional as F
torch.set_num_threads(12)
from data import SplitData
from model import Ranker
from cd_arms import augment_pool_batch
R='<repo>'; dev=torch.device('cpu')
va=SplitData('val',dev)
def load(v):
    ck=torch.load(f'{R}/nn_x/w5/ckpt/{v}.pt',map_location='cpu',weights_only=False); m=Ranker(**ck['cfg']); m.load_state_dict(ck['state_dict']); m.eval(); return m,ck['cfg']
names=['w5_e10_base_s951','w5_e10_base_s952','w5_e10_kd05w192_s951','w5_e10_kd05w192_s952']
MS={}
for v in names:
    m,cfg=load(v); MS[v]=m
va.attach_xt(cfg['xt_names'])
rng=np.random.default_rng(3); cand=np.where(va.M_np>=20)[0]; feeds=np.sort(rng.choice(cand,2500,replace=False))
def lsm(lg,m): return torch.log_softmax(lg.masked_fill(~m,-1e4),1)
def kl(lp,lq,m): return ((lp.exp()*(lp-lq))*m).sum(1)
acc={k:[] for k in ('kl_sub_base','kl_sub_stud','kl_seed_base','kl_seed_stud','kl_t_s','kl_dom','top1_sub','corr_sub','keepfrac','aux_abs','aux_sd','main_abs','kl_sub2_base','kl_sub2_stud','kl_seed_sub')}
cap={}
for v,m in MS.items():
    m.head[0].register_forward_hook(lambda mod,inp,out,v=v: cap.__setitem__(v,inp[0].detach()))
with torch.no_grad():
    for fb in va.batches(feeds,8192,128,shuffle=False):
        b=va.gather(fb); m0=b['cmask'].clone(); y0=b['y'].clone()
        full={}
        for v,m in MS.items(): full[v]=m(b)            # infer_domain = 1 (random-like), as predict()
        bb=dict(b); bb['dom']=F.one_hot(torch.zeros(len(fb),dtype=torch.long),3).float()
        prod=MS[names[0]](bb)
        aux=MS[names[0]].aux_head(cap[names[0]]).squeeze(-1)
        cen=lambda x,mk: (x-(x*mk).sum(1,keepdim=True)/mk.sum(1,keepdim=True))*mk
        mf=m0.float(); a_c=cen(aux,mf)
        acc['aux_abs'].append(((aux.abs()*mf).sum(1)/mf.sum(1)).numpy()); acc['aux_sd'].append(torch.sqrt((a_c**2).sum(1)/mf.sum(1)).numpy()); acc['main_abs'].append(((full[names[0]].abs()*mf).sum(1)/mf.sum(1)).numpy())
        acc['kl_seed_base'].append(kl(lsm(full[names[0]],m0),lsm(full[names[1]],m0),mf).numpy())
        acc['kl_seed_stud'].append(kl(lsm(full[names[2]],m0),lsm(full[names[3]],m0),mf).numpy())
        tch=(cen(full[names[0]],mf)+cen(full[names[1]],mf))/2
        acc['kl_t_s'].append(kl(lsm(tch,m0),lsm(full[names[2]],m0),mf).numpy())
        acc['kl_dom'].append(kl(lsm(full[names[0]],m0),lsm(prod,m0),mf).numpy())
        for a_c_,tag in ((0.5,'kl_sub'),(0.2,'kl_sub2')):
            torch.manual_seed(int(fb[0])+int(a_c_*100)); bs=dict(b); bs['cmask']=m0.clone(); bs['y']=y0.clone(); augment_pool_batch(bs,a_c_,0.0); ms=bs['cmask']; msf=ms.float()
            sub={}
            for v in (names[0],names[1],names[2]):
                X=MS[v].dense(bs); sub[v]=MS[v](bs,X)
            acc[tag+'_base'].append(kl(lsm(full[names[0]],ms),lsm(sub[names[0]],ms),msf).numpy())
            acc[tag+'_stud'].append(kl(lsm(full[names[2]],ms),lsm(sub[names[2]],ms),msf).numpy())
            if a_c_==0.5:
                acc['kl_seed_sub'].append(kl(lsm(sub[names[0]],ms),lsm(sub[names[1]],ms),msf).numpy())
                acc['keepfrac'].append((msf.sum(1)/mf.sum(1)).numpy())
                fa,fs=full[names[0]].masked_fill(~ms,-1e4),sub[names[0]].masked_fill(~ms,-1e4)
                acc['top1_sub'].append((fa.argmax(1)==fs.argmax(1)).float().numpy())
                x,yv=cen(full[names[0]],msf),cen(sub[names[0]],msf); acc['corr_sub'].append(((x*yv).sum(1)/torch.sqrt((x**2).sum(1)*(yv**2).sum(1)).clamp_min(1e-9)).numpy())
se=lambda d: d.std()/np.sqrt(len(d))
for k,v in acc.items():
    v=np.concatenate(v); print(f'{k:14s} mean {v.mean():.5f} +- {se(v):.5f}  median {np.median(v):.5f}  (n {len(v)})')
