import numpy as np, polars as pl, lightgbm as lgb, sys, time
from sklearn.model_selection import GroupKFold
from common import *
df=pl.read_parquet(f'{OUT}/features_val.parquet'); y=df['target'].to_numpy(); fidx=df['fidx'].to_numpy()
V=Val('val')
base=['f1','f2','f3','f4','f5','cat','M']
allf=[c for c in df.columns if c not in ('fidx','target') and 'nf0.15' not in c]
tr,te=next(GroupKFold(5).split(y,y,fidx))
g=lambda idx: np.bincount(fidx[idx])[np.unique(fidx[idx])]
tem=np.zeros(V.n,bool); tem[np.unique(fidx[te])]=True
for obj,trunc in [('rank_xendcg',None),('lambdarank',30)]:
    P=dict(objective=obj, label_gain=[0,1], learning_rate=0.05, num_leaves=31, min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
       lambda_l2=10.0, verbose=-1, num_threads=16, seed=0)
    if trunc: P['lambdarank_truncation_level']=trunc
    for nm,feats in [('base',base),('all',allf)]:
        t0=time.time()
        dtr=lgb.Dataset(df[tr].select(feats).to_numpy(),y[tr],group=g(tr),feature_name=feats,categorical_feature=['cat'])
        b=lgb.train(P,dtr,600)
        X=df[te].select(feats).to_numpy(); res={}
        for k in [100,200,300,400,600]:
            s=np.zeros(len(y)); s[te]=b.predict(X,num_iteration=k)
            res[k]=round(V.per_feed(s)[tem].mean(),4)
        print(obj,trunc,nm,res,f'{time.time()-t0:.0f}s',flush=True)
