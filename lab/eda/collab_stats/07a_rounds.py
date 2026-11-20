import numpy as np, polars as pl, lightgbm as lgb, sys
from sklearn.model_selection import GroupKFold
from common import *
df=pl.read_parquet(f'{OUT}/features_val.parquet'); y=df['target'].to_numpy(); fidx=df['fidx'].to_numpy()
base=['f1','f2','f3','f4','f5','cat','M']
allf=[c for c in df.columns if c not in ('fidx','target') and 'nf0.15' not in c]
P=dict(objective='lambdarank', metric='ndcg', eval_at=[1000], label_gain=[0,1], lambdarank_truncation_level=300,
       learning_rate=0.05, num_leaves=31, min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
       lambda_l2=10.0, verbose=-1, num_threads=20, seed=0)
tr,te=next(GroupKFold(5).split(y,y,fidx))
for nm,feats in [('base',base),('all',allf)]:
    g=lambda idx: np.bincount(fidx[idx])[np.unique(fidx[idx])]
    dtr=lgb.Dataset(df[tr].select(feats).to_numpy(),y[tr],group=g(tr),feature_name=feats,categorical_feature=['cat'])
    dte=lgb.Dataset(df[te].select(feats).to_numpy(),y[te],group=g(te),reference=dtr)
    ev={}; lgb.train(P,dtr,1200,valid_sets=[dte],callbacks=[lgb.record_evaluation(ev)])
    s=ev['valid_0']['ndcg@1000']
    print(nm, {k:round(s[k-1],4) for k in [50,100,200,300,400,600,800,1000,1200]}, 'best', int(np.argmax(s))+1, round(max(s),4), flush=True)
