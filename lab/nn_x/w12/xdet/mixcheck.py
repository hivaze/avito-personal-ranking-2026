"""Info: does adding the 4 XPOOL re-cools to the 4 m02ref re-cools beat adding 4 same-root other-seed m02 re-cools (r6_cd_m02)?"""
import sys; sys.path.insert(0,'<repo>/src')
from common import *
from metrics import ndcg_per_feed
from stack_val import feed_rank_z
from pathlib import Path
W=Path('<repo>/nn_x/w5/preds'); R6=Path('<repo>/nn_x/r6/preds')
fv,Mv,ov=feed_meta('val'); yv=np.load(FLAT/'val/cand_target.npy')
Mh,oh=np.load(W/'holdout/holdout_meta.npy'); Mh=Mh.astype(np.int64); oh=oh.astype(np.int64); yh=np.load(W/'holdout/holdout_y.npy')
ev=lambda s: ndcg_per_feed(np.asarray(s,np.float32),yv,ov,Mv); eh=lambda s: ndcg_per_feed(np.asarray(s,np.float32),yh,oh,Mh)
zv=lambda x: feed_rank_z(ov,Mv,np.asarray(x,np.float32))[:,1]; zh=lambda x: feed_rank_z(oh,Mh,np.asarray(x,np.float32))[:,1]
def ld(d,v): return zv(np.load(d/f'nn_{v}_trainonly_val.npy')), zh(np.load(d/f'holdout/nn_{v}_holdout.npy'))
roots=(101,102,103,104)
ref=[ld(W,f'w5_cd_m02ref_r{r}') for r in roots]; ctl=[ld(R6,f'r6_cd_m02_r{r}') for r in roots]
def pool(ms): return ev(np.mean([m[0] for m in ms],0)), eh(np.mean([m[1] for m in ms],0))
bv,bh=pool(ref+ctl); rv,rh=pool(ref); se=lambda d: d.std()/np.sqrt(len(d))
print(f'ref4 val {rv.mean():.5f} hold {rh.mean():.5f} | ref4+ctl4 val {bv.mean():.5f} hold {bh.mean():.5f}')
for a in ('xp2','xp2m05','xp2m10'):
    arm=[ld(W,f'w5_cd_{a}_r{r}') for r in roots]; v,h=pool(ref+arm); v12,h12=pool(ref+ctl+arm)
    print(f'{a}: ref4+arm4 vs ref4+ctl4: val {(v-bv).mean():+.5f}+-{se(v-bv):.5f} (M<50 {(v-bv)[Mv<50].mean():+.5f}) hold {(h-bh).mean():+.5f}+-{se(h-bh):.5f} | 12 (ref+ctl+arm) vs 8: val {(v12-bv).mean():+.5f}+-{se(v12-bv):.5f} hold {(h12-bh).mean():+.5f}+-{se(h12-bh):.5f}')
