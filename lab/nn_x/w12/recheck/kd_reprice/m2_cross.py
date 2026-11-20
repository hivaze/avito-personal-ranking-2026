"""M2: cross-family z-corr (different seeds only; same-seed pairs separately) on val and holdout + N=2 pair pools.
For each pair of families (A,X): mean NDCG of cross pairs (different seeds) vs within-A pairs and within-X pairs (paired per feed)."""
from lib import *
import itertools, json
S4=(951,952,953,954); S2=(951,952)
F={ 'base':fam('base'),'kd05':fam('kd05'),'kd03':fam('kd03'),'kd08':fam('kd08'),'kd10':fam('kd10'),'kd05t25':fam('kd05t25'),
 'kd05w192':fam('kd05w192'),'kd05t25w192':fam('kd05t25w192'),'kd05g2w192':fam('kd05g2w192'),'kd03g2w192':fam('kd03g2w192'),'kx05':fam('kx05'),'kx08':fam('kx08'),
 'kd05t25w256':fam('kd05t25w256',S2),'kd05t25w192b3':fam('kd05t25w192b3',S2),'kw_drop0':fam('kw_drop0',S2),'kw_e20':fam('kw_e20',S2),'kw_hl2':fam('kw_hl2',S2),'kw_ff4':fam('kw_ff4',S2),'kw_id':fam('kw_id',S2),
 'pop1kd':fam('pop1kd05t25',S4,'abl_e10_'),'pop1':fam('pop1',S4,'abl_e10_'),'e20':fam('e20'),
 'gruF10':[f'v23_gru_pre2s_s{s}' for s in range(501,507)],'d192F12':[f'v26_d192_pre2s_s{s}' for s in range(701,708)],'lr1F9':[f'v22_pre2s_lr1_s{s}' for s in range(401,407)],'F7':[f'v17_pre2s_s{s}' for s in range(101,108)]}
seed=lambda v: v.rsplit('_s',1)[1]
names=list(F)
Z={n:[ld(v) for v in F[n]] for n in names}
for n in names:
    assert all(z[0] is not None and z[1] is not None for z in Z[n]), n
def cc(A,X,k):
    same=[];diff=[]
    for a in F[A]:
        for x in F[X]:
            if a==x: continue
            if A==X and a>=x: continue
            c=np.corrcoef(ld(a)[k],ld(x)[k])[0,1]
            (same if (seed(a)==seed(x) and A!=X) else diff).append(c)
    return (np.mean(diff) if diff else np.nan, np.mean(same) if same else np.nan)
ref=['kd05w192','kd05t25w192','kd05','base']
out={}
print('cross corr (different seeds) val/hold, and same-seed cross corr val; rows = family, cols = reference family')
print(f'{"family":15s} '+' '.join(f'{r:>27s}' for r in ref))
for n in names:
    row=[]
    for r in ref:
        dv,sv=cc(n,r,0); dh,sh=cc(n,r,1); out[f'{n}|{r}']=dict(val=dv,hold=dh,same_val=sv,same_hold=sh)
        row.append(f'{dv:.4f}/{dh:.4f} (ss {sv:.4f})')
    print(f'{n:15s} '+' '.join(f'{x:>27s}' for x in row))
json.dump(out,open(OUT/'m2_cross.json','w'),indent=1)
