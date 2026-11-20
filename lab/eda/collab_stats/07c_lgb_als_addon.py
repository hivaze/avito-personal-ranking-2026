"""Supplementary ablation adding GPU-ALS features (same folds / params / negative subsampling as 07_lgb_ablation.py)."""
import numpy as np, polars as pl, time, sys, json
import lightgbm as lgb
from joblib import Parallel, delayed
from sklearn.model_selection import GroupKFold
from common import *
t0=time.time()
df=pl.read_parquet(f'{OUT}/features_val.parquet')
pr=np.load(f'{OUT}/cf_i2v_d128_e10_nf0_profiles_val.npz')
df=df.with_columns([pl.Series(f'cf_i2vp_{k}',pr[k]) for k in ['samecat_exp10','exp2','et1_exp5']])
A=np.load(f'{OUT}/alsgpu_f128_i12_evt_a5_val.npz'); AP=np.load(f'{OUT}/cf_alsgpu_f128_i12_evt_a5_profiles_val.npz')
df=df.with_columns([pl.Series('cf_als_dot',A['dot']),pl.Series('cf_als_lastk20',A['lastk20']),pl.Series('cf_als_exp5',AP['exp5']),pl.Series('cf_als_samecat',AP['samecat_exp10'])])
V=Val('val'); y=df['target'].to_numpy(); fidx=df['fidx'].to_numpy()
I2V='i2v_d128_e10_nf0'
G={
 'base':['f1','f2','f3','f4','f5','cat','M'],
 'item_te':['ite_clicks','ite_n_imp','ite_coec_Ef_a20','ite_ctr_a20','ite_mean_pos'],
 'pop_train':['pop_trhist_all','pop_trhist_et1','pop_trhist_et2','pop_trhist_et3','pop_trhist_et4','pop_age_last_seen_h','pop_age_first_seen_h'],
 'pop_transductive':['tr_cand_feed_cnt','tr_cand_dist_users','tr_hist_all','tr_hist_et1'],
 'user_cat_train':['usr_ctr','usr_nfeeds','uc_ctr','uc_clicks','uc_impr','uc_coec'],
 'own_hist_cat':['oh_cat_cnt','oh_cat_cnt_et1','oh_cat_cnt_last20','oh_cat_share','oh_H'],
 'cf_i2v':[f'cf_{I2V}_{k}' for k in ['cos_k5','maxcos_k5','cos_k20','cos_k256','io_k5']]+['cf_i2vp_samecat_exp10','cf_i2vp_exp2','cf_i2vp_et1_exp5'],
 'repeat':['rep_n_prev','rep_pos_prev'],
 'future_hist':['rep_future_hist'],
 'cf_als':['cf_als_dot','cf_als_lastk20','cf_als_exp5','cf_als_samecat'],
}
folds=list(GroupKFold(n_splits=5).split(np.zeros(len(y)),y,fidx))
rng=np.random.RandomState(0); keep_neg=rng.rand(len(y))<0.30
NR=200
P=dict(objective='rank_xendcg', learning_rate=0.05, num_leaves=31, min_data_in_leaf=100, max_bin=63, feature_fraction=0.8,
       bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1, num_threads=10, seed=0)
def run(name, feats):
    oof=np.zeros(len(y)); imp=np.zeros(len(feats))
    for tr,te in folds:
        tr=tr[(y[tr]==1)|keep_neg[tr]]
        gtr=np.bincount(fidx[tr])[np.unique(fidx[tr])]
        ds=lgb.Dataset(df[tr].select(feats).to_numpy(),y[tr],group=gtr,feature_name=feats,categorical_feature=['cat'] if 'cat' in feats else 'auto')
        b=lgb.train(P,ds,num_boost_round=NR)
        oof[te]=b.predict(df[te].select(feats).to_numpy()); imp+=b.feature_importance('gain')
    return name, oof, dict(zip(feats, imp/imp.sum()))
base=G['base']
full=base+sum([G[g] for g in G if g!='base'],[])
configs=[('base+cf_als',base+G['cf_als']),('ALL+als',full),('ALL+als-cf_i2v',[c for c in full if c not in G['cf_i2v']]),
         ('base+cf_als+cf_i2v',base+G['cf_als']+G['cf_i2v']),('base+cf_als+pop_train+future_hist',base+G['cf_als']+G['pop_train']+G['future_hist']),
         ('ALL+als-own_hist_cat-user_cat_train',[c for c in full if c not in G['own_hist_cat']+G['user_cat_train']]),
         ('ALL+als-item_te',[c for c in full if c not in G['item_te']]),('ALL+als-pop_train-pop_transductive',[c for c in full if c not in G['pop_train']+G['pop_transductive']])]
out=Parallel(n_jobs=4)(delayed(run)(n,f) for n,f in configs)
R={n:(o,imp) for n,o,imp in out}
pf={n:V.per_feed(o) for n,(o,_) in R.items()}
pf['base']=V.per_feed(np.load(f'{OUT}/oof_base.npy')); pf['ALL(no als)']=V.per_feed(np.load(f'{OUT}/oof_ALL.npy'))
pf['f1_only(rt)']=V.pf_rt(df['f1'].to_numpy())
def cmp(a,b):
    d=pf[a]-pf[b]; return d.mean(), d.std(ddof=1)/np.sqrt(len(d))
for n in ['f1_only(rt)','base','ALL(no als)']+[c for c,_ in configs]:
    s=f'{n:40s} NDCG={pf[n].mean():.4f}'
    for ref in ['base','ALL(no als)','ALL+als']:
        if ref in pf and n!=ref: d,se=cmp(n,ref); s+=f' | vs {ref}: {d:+.4f}+-{se:.4f}'
    print(s, flush=True)
print('\nTop-30 features by gain share in ALL+als:')
for k,v in sorted(R['ALL+als'][1].items(), key=lambda x:-x[1])[:30]: print(f'  {k:40s} {v:.4f}')
np.save(f'{OUT}/oof_ALL_als.npy', R['ALL+als'][0])
# tie check on predictions
o=R['ALL+als'][0]; print('pessimistic-tie NDCG', V.ndcg(o), 'with f1 tiebreak', V.ndcg(tiebreak(o, df['f1'].to_numpy())))
tlog(t0,'done')
