"""Within-feed rank correlation of collaborative features with f1..f5 (explains redundancy)."""
import numpy as np, polars as pl
from common import *
V=Val('val'); df=pl.read_parquet(f'{OUT}/features_val.parquet')
pr=np.load(f'{OUT}/cf_i2v_d128_e10_nf0_profiles_val.npz')
def frank(x):
    x=np.asarray(x,dtype=np.float64); x=np.where(np.isnan(x),-1e18,x)
    rnd=np.random.RandomState(0).rand(len(x))
    order=np.lexsort((rnd,x,V.fid)); r=np.empty(len(x)); r[order]=np.arange(len(x))
    r=r-np.repeat(V.off,V.M); return r/np.maximum(np.repeat(V.M,V.M)-1,1)-0.5
R={c:frank(df[c].to_numpy()) for c in ['f1','f2','f3','f4','f5']}
feats={'i2v_samecat':pr['samecat_exp10'],'i2v_last5':pr['last5'],'oh_cat_cnt_last20':df['oh_cat_cnt_last20'].to_numpy(),'pop_trhist_et1':df['pop_trhist_et1'].to_numpy(),
       'ite_clicks':df['ite_clicks'].to_numpy(),'ite_n_imp':df['ite_n_imp'].to_numpy(),'ite_coec_Ef_a20':df['ite_coec_Ef_a20'].to_numpy(),'uc_impr':df['uc_impr'].to_numpy(),'tr_hist_all':df['tr_hist_all'].to_numpy()}
print(f'{"feature":20s} '+' '.join(f'{c:>7s}' for c in R))
for k,x in feats.items():
    X=frank(x); print(f'{k:20s} '+' '.join(f'{np.corrcoef(X,R[c])[0,1]:7.3f}' for c in R))
