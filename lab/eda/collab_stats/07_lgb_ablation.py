"""LightGBM (rank_xendcg) 5-fold GroupKFold over val feeds; feature-group ablation with paired per-feed comparisons.
Training folds use all positives + 30% of negatives (speed; CPU heavily shared). OOF predictions scored with the exact metric."""
import numpy as np, polars as pl, time, sys, json, os, glob
import lightgbm as lgb
from joblib import Parallel, delayed
from sklearn.model_selection import GroupKFold
from common import *
t0=time.time()
df=pl.read_parquet(f'{OUT}/features_val.parquet')
pr=np.load(f'{OUT}/cf_i2v_d128_e10_nf0_profiles_val.npz')
df=df.with_columns([pl.Series(f'cf_i2vp_{k}',pr[k]) for k in ['samecat_exp10','exp2','et1_exp5']])
V=Val('val'); y=df['target'].to_numpy(); fidx=df['fidx'].to_numpy()
I2V='i2v_d128_e10_nf0'
als_files=sorted(glob.glob(f'{OUT}/als_*_val.npz'))
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
}
if als_files:
    d=np.load(als_files[-1]); tag=als_files[-1].split('/')[-1].replace('_val.npz','')
    for k in ['dot','lastk5','lastk20']: df=df.with_columns(pl.Series(f'cf_als_{k}',d[k].astype(np.float32)))
    G['cf_als']=['cf_als_dot','cf_als_lastk5','cf_als_lastk20']; print('ALS from', tag)
for k,v in G.items(): assert all(c in df.columns for c in v), (k,[c for c in v if c not in df.columns])
folds=list(GroupKFold(n_splits=5).split(np.zeros(len(y)),y,fidx))
rng=np.random.RandomState(0); keep_neg=rng.rand(len(y))<0.30
NR=int(sys.argv[1]) if len(sys.argv)>1 else 200
P=dict(objective='rank_xendcg', learning_rate=0.05, num_leaves=31, min_data_in_leaf=100, max_bin=63, feature_fraction=0.8,
       bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1, num_threads=10, seed=0)
def run(name, feats):
    oof=np.zeros(len(y)); imp=np.zeros(len(feats))
    for tr,te in folds:
        tr=tr[(y[tr]==1)|keep_neg[tr]]
        gtr=np.bincount(fidx[tr])[np.unique(fidx[tr])]
        ds=lgb.Dataset(df[tr].select(feats).to_numpy(),y[tr],group=gtr,feature_name=feats,
                       categorical_feature=['cat'] if 'cat' in feats else 'auto')
        b=lgb.train(P,ds,num_boost_round=NR)
        oof[te]=b.predict(df[te].select(feats).to_numpy()); imp+=b.feature_importance('gain')
    return name, oof, dict(zip(feats, imp/imp.sum()))
base=G['base']; allg=[g for g in G if g!='base']
full=base+sum([G[g] for g in allg],[])
configs=[('base',base),('ALL',full)]+[(f'base+{g}',base+G[g]) for g in allg]+[(f'ALL-{g}',[c for c in full if c not in G[g]]) for g in allg]
configs+=[('base+cf_i2v+pop_train',base+G['cf_i2v']+G['pop_train'])]
out=Parallel(n_jobs=4)(delayed(run)(n,f) for n,f in configs)
R={n:(o,imp) for n,o,imp in out}
pf={n:V.per_feed(o) for n,(o,_) in R.items()}
pf['f1_only(rt)']=V.pf_rt(df['f1'].to_numpy())
def cmp(a,b):
    d=pf[a]-pf[b]; return d.mean(), d.std(ddof=1)/np.sqrt(len(d))
print(f'\nrounds={NR} i2v={I2V} als={als_files[-1] if als_files else None}')
print(f'{"f1_only(rt)":32s} NDCG={pf["f1_only(rt)"].mean():.4f}')
for n,_ in configs:
    ref='ALL' if n.startswith('ALL-') else 'base'
    d,se=cmp(n,ref) if n not in ('base','ALL') else (0,0)
    dB,seB=cmp(n,'base') if n!='base' else (0,0)
    dF,seF=cmp(n,'f1_only(rt)')
    print(f'{n:32s} NDCG={pf[n].mean():.4f}  vs {ref}: {d:+.4f}+-{se:.4f} | vs base {dB:+.4f}+-{seB:.4f} | vs f1-only {dF:+.4f}+-{seF:.4f}', flush=True)
print('\nTop-30 features by gain share in ALL:')
for k,v in sorted(R['ALL'][1].items(), key=lambda x:-x[1])[:30]: print(f'  {k:40s} {v:.4f}')
np.save(f'{OUT}/oof_ALL.npy', R['ALL'][0]); np.save(f'{OUT}/oof_base.npy', R['base'][0])
json.dump({n:float(pf[n].mean()) for n in pf}, open(f'{OUT}/lgb_ablation_r{NR}.json','w'), indent=1)
tlog(t0,'done')
