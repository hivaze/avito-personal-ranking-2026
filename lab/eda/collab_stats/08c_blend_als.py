import numpy as np, polars as pl
from common import *
V=Val('val'); df=pl.read_parquet(f'{OUT}/features_val.parquet')
A=np.load(f'{OUT}/cf_alsgpu_f128_i12_evt_a5_profiles_val.npz'); I=np.load(f'{OUT}/cf_i2v_d128_e10_nf0_profiles_val.npz')
def frank(x):
    x=np.asarray(x,dtype=np.float64); x=np.where(np.isnan(x),-1e18,x)
    order=np.lexsort((x,V.fid)); r=np.empty(len(x)); r[order]=np.arange(len(x)); r=r-np.repeat(V.off,V.M)
    return r/np.maximum(np.repeat(V.M,V.M)-1,1)
F1=frank(df['f1'].to_numpy()); base=V.pf_rt(F1); print('f1', base.mean())
for nm,x in [('als_exp5',A['exp5']),('als_last10',A['last10']),('als_samecat',A['samecat_exp10'])]:
    X=frank(x)
    for w in [0.25,0.5,0.75,1.0]:
        pf=V.pf_rt(F1+w*X); d=pf-base; print(f'f1 + {w}*{nm}: {pf.mean():.4f}  lift {d.mean():+.4f}+-{d.std(ddof=1)/np.sqrt(len(d)):.4f}', flush=True)
X=F1+0.5*frank(A['exp5']); p1=V.pf_rt(X)
X2=X+0.25*frank(I['last5']); p2=V.pf_rt(X2); print('f1+.5als+.25i2v', p2.mean(), f'{(p2-p1).mean():+.4f}')
X3=X+0.25*frank(df['pop_trhist_et1'].to_numpy()); p3=V.pf_rt(X3); print('f1+.5als+.25pop_et1', p3.mean(), f'{(p3-p1).mean():+.4f}+-{(p3-p1).std(ddof=1)/np.sqrt(V.n):.4f}')
# spearman within feed with f1..f5
def fr2(x): return frank(x)-0.5
for c in ['f1','f3','f4','f5']: print('corr(als_exp5,',c,')', np.corrcoef(fr2(A['exp5']),fr2(df[c].to_numpy()))[0,1].round(3), ' corr(i2v_last5)', np.corrcoef(fr2(I['last5']),fr2(df[c].to_numpy()))[0,1].round(3))
