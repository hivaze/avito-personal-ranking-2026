"""M1: member table of every e10 arm (TO level): mean member val/hold, paired delta vs reference arm on the same seeds,
within-arm z-corr (val), N-pool val/hold. Reference = kd05w192 for d192 arms and kd05 for d128 arms; base for hard-label."""
from lib import *
import itertools, json
S4=(951,952,953,954); S2=(951,952)
arms={ 'base':S4,'kd03':S4,'kd05':S4,'kd08':S4,'kd10':S4,'kd05t25':S4,'kd08t25':(951,952,953),'kd05w192':S4,'kd05t25w192':S4,
 'kd05t25w256':S2,'kd05t25w192b3':S2,'kd05g2w192':S4,'kd03g2w192':S4,'kx05':S4,'kx08':S4,'kw_drop0':S2,'kw_e20':S2,'kw_id':S2,'kw_hl2':S2,'kw_ff4':S2,
 'e20':S4,'e15':S2,'bs4':S2,'bs4lr4':S2,'ap1':S2,'pseudo':S2,'trunc':S4}
ref={'kd05t25w192':'kd05w192','kd05t25w256':'kd05t25w192','kd05t25w192b3':'kd05t25w192','kd05g2w192':'kd05w192','kd03g2w192':'kd05w192','kx05':'kd05w192','kx08':'kd05w192',
     'kw_drop0':'kd05w192','kw_e20':'kd05w192','kw_id':'kd05w192','kw_hl2':'kd05w192','kw_ff4':'kd05w192','kd05w192':'kd05','kd05t25':'kd05','kd03':'kd05','kd08':'kd05','kd10':'kd05','kd08t25':'kd08'}
res={}
print(f'{"arm":16s} n  member val / hold     | delta vs ref (val | hold)                         | corr_w | N-pool val / hold')
for a,sd in arms.items():
    vs=fam(a,sd); Z=[ld(v) for v in vs]
    if any(z[0] is None for z in Z): print(a,'missing'); continue
    mv=np.mean([ev(z[0]).mean() for z in Z]); mh=np.mean([eh(z[1]).mean() for z in Z])
    r=ref.get(a,'base'); Rz=[ld(v) for v in fam(r,sd)]
    dv=np.mean([ev(z[0])-ev(q[0]) for z,q in zip(Z,Rz)],0); dh=np.mean([eh(z[1])-eh(q[1]) for z,q in zip(Z,Rz)],0)
    c=np.mean([np.corrcoef(Z[i][0],Z[j][0])[0,1] for i,j in itertools.combinations(range(len(Z)),2)])
    ch=np.mean([np.corrcoef(Z[i][1],Z[j][1])[0,1] for i,j in itertools.combinations(range(len(Z)),2)])
    pv,ph=pool(vs)
    res[a]=dict(n=len(sd),mv=mv,mh=mh,ref=r,dv=dv.mean(),dv_se=se(dv),dh=dh.mean(),dh_se=se(dh),corr_val=c,corr_hold=ch,pool_val=ev(pv).mean(),pool_hold=eh(ph).mean())
    print(f'{a:16s} {len(sd)}  {mv:.5f} / {mh:.5f} | vs {r:12s} {f(dv)} | {f(dh)} | {c:.4f}/{ch:.4f} | {ev(pv).mean():.5f} / {eh(ph).mean():.5f}')
json.dump(res,open(OUT/'m1_members.json','w'),indent=1)
