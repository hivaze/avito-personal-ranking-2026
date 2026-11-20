import numpy as np, polars as pl, time
from common import *
t0=time.time()
V = Val('val')
rel_items = np.load(f'{OUT}/rel_items.npy')
lut = np.full(NITEM, -1, np.int32); lut[rel_items]=np.arange(len(rel_items),dtype=np.int32)
A = dict(np.load(f'{OUT}/item_day_aggs.npz'))
vi = np.asarray(arr('val','cand_item_id')); r = lut[vi]
f1 = np.asarray(arr('val','cand_f1')).astype(np.float64)
rnd = np.random.RandomState(0).rand(len(vi)); RP = V.rand_pf(); F1P = V.pf_rt(f1)
print("random(20 seeds avg)", RP.mean(), "f1 (rt)", F1P.mean())
days_all = np.arange(14)
def window(days):
    return {k: v[:, days].sum(1) for k,v in A.items()}
def scores(W, alpha_ctr=20, alpha_coec=2):
    n,c = W['n'][r], W['c'][r]
    p0 = W['c'].sum()/W['n'].sum()
    out = {
      'n_imp': n, 'clicks': c,
      'ctr_raw': np.where(n>0, c/np.maximum(n,1), -1),
      f'ctr_bayes_a{alpha_ctr}': (c+alpha_ctr*p0)/(n+alpha_ctr),
    }
    for e in ['Eg','Ef','Eu']:
        E = W[e][r]
        out[f'coec_{e}_a{alpha_coec}'] = (c+alpha_coec)/(E+alpha_coec)
    out['mean_pos'] = -np.where(n>0, W['spos'][r]/np.maximum(n,1), 400)
    return out
rows=[]
W = window(days_all)
cov = (W['n'][r]>0).mean()
print(f'coverage (val cand rows with >=1 train impression): {cov:.4f}; >=10: {(W["n"][r]>=10).mean():.4f}; with >=1 click: {(W["c"][r]>0).mean():.4f}')
for a_ctr, a_coec in [(5,0.5),(20,2),(100,5),(500,20)]:
    S = scores(W, a_ctr, a_coec)
    for k,s in S.items():
        if a_ctr!=20 and k in ('n_imp','clicks','ctr_raw','mean_pos'): continue
        a,d,se = V.cmp(s, RP)
        rows.append((k, a, d, se)); print(f'{k:22s} ndcg={a:.4f}  vs random: {d:+.4f} +- {se:.4f}  pess-ties={V.ndcg(s):.4f}', flush=True)
# windows (recency)
print('--- windows for coec_Ef a2 and ctr_bayes a20 ---')
for nm,days in [('all14',days_all),('first9 (Sep6-14; gap>=7d)',np.arange(9)),('last5 days',np.arange(9,14)),('last3',np.arange(11,14)),('first5',np.arange(5))]:
    Wd = window(days); S=scores(Wd,20,2)
    cov=(Wd['n'][r]>0).mean()
    print(f'{nm:28s} cov={cov:.3f} coec_Ef={V.pf_rt(S["coec_Ef_a2"]).mean():.4f} coec_Eu={V.pf_rt(S["coec_Eu_a2"]).mean():.4f} ctr_bayes={V.pf_rt(S["ctr_bayes_a20"]).mean():.4f} n_imp={V.pf_rt(S["n_imp"]).mean():.4f}')
# exponential decay weighting
for hl in [1,2,4]:
    wts = 0.5**((13-np.arange(14))/hl)
    Wd = {k:(v*wts[None,:]).sum(1) for k,v in A.items()}
    S=scores(Wd,20,2)
    print(f'decay hl={hl}d coec_Ef={V.pf_rt(S["coec_Ef_a2"]).mean():.4f} ctr_bayes={V.pf_rt(S["ctr_bayes_a20"]).mean():.4f}')
# val day-by-day decay of train-based coec
ts = V.f['timestamp'].to_numpy(); vday = (ts-ts.min())//86400
S=scores(W,20,2); pf = V.pf_rt(S['coec_Ef_a2']); pr = RP; pf1=F1P
for d in np.unique(vday):
    m=vday==d; print(f'val day {d}: feeds {m.sum()} coec_Ef {pf[m].mean():.4f} random {pr[m].mean():.4f} f1 {pf1[m].mean():.4f} lift_over_rand {pf[m].mean()-pr[m].mean():+.4f}')
# category CTR
cn=np.load(f'{OUT}/cat_n.npy'); cc=np.load(f'{OUT}/cat_c.npy'); cEf=np.load(f'{OUT}/cat_Ef.npy'); cEu=np.load(f'{OUT}/cat_Eu.npy')
vc = np.asarray(arr('val','cand_category_id')).astype(int)
for nm,s in [('cat_ctr',cc/np.maximum(cn,1)),('cat_coec_Ef',cc/np.maximum(cEf,1e-9)),('cat_coec_Eu',cc/np.maximum(cEu,1e-9)),('cat_impr',cn)]:
    print(f'{nm:15s} ndcg={V.pf_rt(s[vc]).mean():.4f} (pess {V.ndcg(s[vc]):.4f})')
# category CTR on val itself (in-sample, upper bound-ish)
vt = V.target
vcn=np.bincount(vc,minlength=54); vcc=np.bincount(vc,weights=vt,minlength=54)
print('cat_ctr_val_insample', V.pf_rt((vcc/np.maximum(vcn,1))[vc]).mean())
np.save(f'{OUT}/val_item_coef.npy', S['coec_Ef_a2'])
