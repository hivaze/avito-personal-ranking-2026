"""Paired per-feed comparisons among item-TE variants (position debiasing, smoothing) and popularity variants."""
import numpy as np, time
from common import *
V=Val('val')
rel_items=np.load(f'{OUT}/rel_items.npy'); lut=np.full(NITEM,-1,np.int32); lut[rel_items]=np.arange(len(rel_items),dtype=np.int32)
r=lut[np.asarray(arr('val','cand_item_id'))]
A=dict(np.load(f'{OUT}/item_day_aggs.npz')); W={k:v.sum(1) for k,v in A.items()}
n,c=W['n'][r],W['c'][r]; p0=W['c'].sum()/W['n'].sum()
HC=np.load(f'{OUT}/train_hist_counts.npy')
S={'ctr_bayes_a100':(c+100*p0)/(n+100),'coec_Eg_a20':(c+20)/(W['Eg'][r]+20),'coec_Ef_a20':(c+20)/(W['Ef'][r]+20),'coec_Eu_a20':(c+20)/(W['Eu'][r]+20),
   'clicks':c,'n_imp':n,'log1p_clicks+log1p_hist_et1':np.log1p(c)+np.log1p(HC[r,1]),'hist_et1':HC[r,1],
   'coec_Ef_a20*log1p(n)':((c+20)/(W['Ef'][r]+20))*np.log1p(n)}
PF={k:V.pf_rt(s,seeds=range(5)) for k,s in S.items()}
def p(a,b):
    d=PF[a]-PF[b]; print(f'{a:30s} - {b:22s}: {d.mean():+.4f} +- {d.std(ddof=1)/np.sqrt(len(d)):.4f}')
p('coec_Eg_a20','ctr_bayes_a100'); p('coec_Ef_a20','ctr_bayes_a100'); p('coec_Ef_a20','coec_Eu_a20'); p('coec_Ef_a20','coec_Eg_a20')
p('clicks','coec_Ef_a20'); p('clicks','n_imp'); p('hist_et1','clicks'); p('log1p_clicks+log1p_hist_et1','hist_et1'); p('coec_Ef_a20*log1p(n)','coec_Ef_a20')
for k in PF: print(f'{k:30s} {PF[k].mean():.4f}')
# lift by coverage: feeds with high share of covered candidates
cov=np.bincount(V.fid,weights=(n>0))/V.M
RP=V.rand_pf()
for lo,hi in [(0,.4),(.4,.6),(.6,.8),(.8,1.01)]:
    m=(cov>=lo)&(cov<hi); print(f'feeds with train-coverage in [{lo},{hi}): {m.sum():5d}  coec_Ef lift {PF["coec_Ef_a20"][m].mean()-RP[m].mean():+.4f}  clicks lift {PF["clicks"][m].mean()-RP[m].mean():+.4f}')
