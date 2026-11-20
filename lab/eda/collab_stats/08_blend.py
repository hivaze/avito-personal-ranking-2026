"""Cheap complementarity check: within-feed rank-normalized blends f1 + w*feature (no model fitting)."""
import numpy as np, polars as pl
from common import *
V=Val('val')
df=pl.read_parquet(f'{OUT}/features_val.parquet')
pr=np.load(f'{OUT}/cf_i2v_d128_e10_nf0_profiles_val.npz')
def frank(x):
    """within-feed percentile rank in [0,1]; NaN -> 0 (bottom)"""
    x=np.asarray(x,dtype=np.float64); x=np.where(np.isnan(x),-1e18,x)
    order=np.lexsort((x,V.fid)); r=np.empty(len(x)); r[order]=np.arange(len(x))
    r=r-np.repeat(V.off,V.M)
    return r/np.maximum(np.repeat(V.M,V.M)-1,1)
f1=df['f1'].to_numpy(); F1=frank(f1)
base=V.pf_rt(F1)
print('f1 alone', base.mean())
feats={'i2v_samecat':pr['samecat_exp10'],'i2v_last5':pr['last5'],'oh_cat_cnt_last20':df['oh_cat_cnt_last20'].to_numpy(),
       'pop_trhist_et1':df['pop_trhist_et1'].to_numpy(),'ite_clicks':df['ite_clicks'].to_numpy(),'ite_coec_Ef_a20':df['ite_coec_Ef_a20'].to_numpy(),
       'uc_impr':df['uc_impr'].to_numpy(),'tr_hist_all':df['tr_hist_all'].to_numpy(),'f3':df['f3'].to_numpy()}
best={}
for k,x in feats.items():
    X=frank(x); row=[]
    for w in [0.1,0.25,0.5,1.0]:
        pf=V.pf_rt(F1+w*X); d=pf-base; row.append((w,pf.mean(),d.mean(),d.std(ddof=1)/np.sqrt(len(d))))
    b=max(row,key=lambda r:r[1]); best[k]=b
    print(f'{k:20s} '+' '.join(f'w={w}:{m:.4f}' for w,m,_,_ in row)+f' | best w={b[0]} lift over f1 {b[2]:+.4f}+-{b[3]:.4f}',flush=True)
# combined: f1 + f3 + i2v_samecat + oh + pop
X=F1+0.25*frank(feats['f3'])
p0=V.pf_rt(X); print('f1+0.25 f3', p0.mean())
X2=X+0.5*frank(feats['i2v_samecat']); p2=V.pf_rt(X2); print('+0.5 i2v_samecat', p2.mean(), f'{(p2-p0).mean():+.4f}')
X3=X2+0.25*frank(feats['pop_trhist_et1']); p3=V.pf_rt(X3); print('+0.25 pop_trhist_et1', p3.mean(), f'{(p3-p2).mean():+.4f}')
X4=X3+0.25*frank(feats['oh_cat_cnt_last20']); p4=V.pf_rt(X4); print('+0.25 oh_cat_cnt_last20', p4.mean(), f'{(p4-p3).mean():+.4f}')
