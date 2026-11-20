# Second detector, step 2. A cross-fitted LightGBM classifier separates val + test feeds (all shown in random
# order) from train feeds, using the 51 summaries of setfeat.py. A production feed holds the top of a ranked
# list and a random-order feed a random subset of it, so the make-up of the candidate set gives the feed away.
# Run from lab/nn_x/w12/xdet with 12 CPU threads. The command line is not on record.
# Produced out/adv_train.npy and out/adv_pos.npy (out-of-fold scores) and printed the AUC per feed-size bin;
# the printed table was not saved, its numbers are in the status log of 2026-11-14 02:21 UTC.
# Kept as it ran: paths and names refer to the lab tree (see lab/ and docs/glossary.md).
# Original: lab/nn_x/w12/xdet/adv.py
#
import numpy as np, lightgbm as lgb
from sklearn.metrics import roc_auc_score
Xt=np.load('out/setfeat_train.npy'); Xp=np.concatenate([np.load('out/setfeat_val.npy'),np.load('out/setfeat_test.npy')])
names=open('out/setfeat_names.txt').read().split('\n'); n=len(Xt); rng=np.random.default_rng(0)
half=rng.integers(0,2,n); pfold=rng.integers(0,2,len(Xp)); score=np.zeros(n,np.float32); sp=np.zeros(len(Xp),np.float32)
for h in (0,1):
    neg=rng.choice(np.where(half==h)[0],400000,replace=False); pos=np.where(pfold==h)[0]
    X=np.concatenate([Xt[neg],Xp[pos]]); y=np.r_[np.zeros(len(neg)),np.ones(len(pos))]
    m=lgb.LGBMClassifier(n_estimators=600,learning_rate=0.05,num_leaves=63,subsample=0.8,subsample_freq=1,colsample_bytree=0.7,n_jobs=12,verbose=-1,min_child_samples=50).fit(X,y)
    score[half!=h]=m.predict_proba(Xt[half!=h],raw_score=True); sp[pfold!=h]=m.predict_proba(Xp[pfold!=h],raw_score=True)
    if h==0:
        imp=sorted(zip(m.booster_.feature_importance('gain'),names),reverse=True)[:12]; print([(b,int(a)) for a,b in imp])
np.save('out/adv_train.npy',score); np.save('out/adv_pos.npy',sp)
M=Xt[:,0]; Mp=Xp[:,0]; p=np.load('<repo>/eda/labels_position/out/train_prandom.npz')['p_random']; z=np.load('out/stats2.npz')['zce10']
for lo,hi in [(5,8),(8,12),(12,20),(20,35),(35,50),(50,100),(100,200),(200,301)]:
    m=(M>=lo)&(M<hi); mp=(Mp>=lo)&(Mp<hi); sub=rng.choice(np.where(m)[0],100000)
    a=roc_auc_score(np.r_[np.zeros(len(sub)),np.ones(mp.sum())],np.r_[score[sub],sp[mp]])
    line=f'M[{lo},{hi}) AUC val/test vs train(OOF) {a:.3f} | train: adv score mean | z<-1: {score[m&(z<-1)].mean():.2f}, z in[1,2]: {score[m&(z>1)&(z<2)].mean():.2f}, z>3: {score[m&(z>3)].mean():.2f}; posfeeds {sp[mp].mean():.2f}'
    kr=m&(p>0.9); kp=m&(p<0.05)
    if lo>=50: line+=f' | AUC oldrand vs oldprod {roc_auc_score(np.r_[np.ones(kr.sum()),np.zeros(kp.sum())],np.r_[score[kr],score[kp]]):.3f}'
    print(line,flush=True)
