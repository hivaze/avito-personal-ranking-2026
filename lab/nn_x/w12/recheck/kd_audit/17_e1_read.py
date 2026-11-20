"""1-epoch fine-tune recipe (FT_EPOCHS=1: cos decay over one epoch, EMA from 0.5 epoch) vs the standard 4-epoch recipe (EMA-final), same member / folds / seed, CPU proxy."""
import sys, os; sys.path.insert(0,'<repo>/src')
import numpy as np
from sklearn.model_selection import GroupKFold
from common import feed_meta, FLAT
from metrics import ndcg_per_feed
from stack_val import feed_rank_z
member='w5_e10_kd05w192_s951'; folds=[0,1]
fv,Mv,ov=feed_meta('val'); yv=np.load(FLAT/'val/cand_target.npy').astype(np.float32); ov=ov.astype(np.int64)
uid=fv['user_id'].to_numpy().astype(np.int64); g=np.where(uid==0,-1-np.arange(len(uid)),uid)
zp_raw=np.load('<repo>/preds/ens_W8kd_to_val.npy'); zv=lambda x: feed_rank_z(ov,Mv,np.asarray(x,np.float32))[:,1]; zp=zv(zp_raw)
se=lambda d: d.std()/np.sqrt(len(d)); f=lambda d: f'{d.mean():+.5f}+-{se(d):.5f}'
ho=[];A=[];B=[]
for k,(tr_f,ho_f) in enumerate(GroupKFold(n_splits=5).split(np.arange(len(uid)),groups=g)):
    if k not in folds: continue
    a=f'out/ftkd_{member}_f{k}_w0.0_s0.npy'; b=f'out/ftkd_{member}_f{k}_w0.0_s0_e1.npy'
    if not (os.path.exists(a) and os.path.exists(b)): print('fold',k,'incomplete'); continue
    ho.append(np.sort(ho_f)); A.append(np.load(a)); B.append(np.load(b))
ho=np.concatenate(ho); nd=lambda p: ndcg_per_feed(p,yv,ov,Mv)[ho]
a4=np.sum([x[-1] for x in A],0); a1=np.sum([x[1] for x in A],0); to=np.sum([x[0] for x in A],0); b1=np.sum([x[-1] for x in B],0); b1raw=np.sum([x[1] for x in B],0)
print(f'folds {folds}, held-out feeds {len(ho)} | TO {nd(to).mean():.5f} | 4-epoch EMA-final {nd(a4).mean():.5f} | in-run epoch-1 snapshot {nd(a1).mean():.5f} | 1-epoch recipe EMA-final {nd(b1).mean():.5f} (raw end {nd(b1raw).mean():.5f})')
print(' member: 1-epoch recipe - 4-epoch EMA-final',f(nd(b1)-nd(a4)),'| 1-epoch recipe - TO',f(nd(b1)-nd(to)),'| 4-epoch - TO',f(nd(a4)-nd(to)))
m4=nd(0.5*zv(a4)+0.5*zp); m1=nd(0.5*zv(b1)+0.5*zp); mt=nd(zp_raw)
print(' mix 0.5 z(FT member) + 0.5 z(W8 TO pool): 4-epoch',f'{m4.mean():.5f}','1-epoch',f'{m1.mean():.5f}','| 1-epoch - 4-epoch',f(m1-m4),'| vs TO pool alone: 4-epoch',f(m4-mt),'1-epoch',f(m1-mt))
