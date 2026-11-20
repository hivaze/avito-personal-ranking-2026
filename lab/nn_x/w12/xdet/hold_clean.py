"""Info: G3-type holdout contrasts on the clean (v2>0.5) vs contaminated (v2<0.1) part of the 30K holdout."""
import sys; sys.path.insert(0,'<repo>/src')
import numpy as np
from metrics import ndcg_per_feed
from stack_val import feed_rank_z
W='<repo>/nn_x/w5/preds/holdout/'
old=np.nan_to_num(np.load('<repo>/eda/labels_position/out/train_prandom.npz')['p_random']); new=np.load('<repo>/eda/labels_position/out/train_prandom_v2.npz')['p_random']
M=np.load('<repo>/data/feats/train/M.npy') if False else np.load('<repo>/nn_x/w12/xdet/out/stats.npz')['M']
cand=np.where((old>0.9)&(M>=50))[0]; hold=np.sort(np.random.default_rng(12345).choice(cand,30000,replace=False))
Mh,oh=np.load(W+'holdout_meta.npy'); Mh=Mh.astype(np.int64); oh=oh.astype(np.int64); yh=np.load(W+'holdout_y.npy'); assert (Mh==M[hold]).all()
pv=new[hold]; print('holdout v2: >0.5 %d, <0.1 %d'%((pv>0.5).sum(),(pv<0.1).sum()))
e=lambda s: ndcg_per_feed(np.asarray(s,np.float32),yh,oh,Mh)
import glob
for v in ['w5_e10_base_s951','w5_e10_base_s952','w5_cd_m02ref_r101','w5_cd_m02ref_r102']:
    s=e(np.load(W+f'nn_{v}_holdout.npy')); print(f'{v}: all {s.mean():.5f} clean {s[pv>0.5].mean():.5f} contaminated {s[pv<0.1].mean():.5f}')
