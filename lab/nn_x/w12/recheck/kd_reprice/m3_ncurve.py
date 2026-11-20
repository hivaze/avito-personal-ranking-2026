"""M3a: pool-size law. NDCG(N) for random N-subsets of a family (val, holdout); fit Q_inf - a/N.
M3b: Q_inf (mean-function quality) estimator for 2-seed arms: Qhat = 2*pool2 - mean(member), paired vs the same statistic of kd05w192
     on the SAME seeds; calibration of its run-to-run noise from 2-seed halves of 4-seed arms."""
from lib import *
import itertools, json
rng=np.random.default_rng(0)
S4=(951,952,953,954); S2=(951,952)
wide=fam('kd05w192',(951,952,953,954,955))+fam('kd05t25w192',(951,952,953,954,955,957))
nar=fam('kd05')+fam('kd05t25')
base=fam('base',(951,952,953,954,955,956))
out={}
def curve(name,vs,maxsub=120):
    Z=[ld(v) for v in vs]; n=len(vs); res={}
    for N in range(1,n+1):
        subs=list(itertools.combinations(range(n),N))
        if len(subs)>maxsub: subs=[subs[i] for i in rng.choice(len(subs),maxsub,replace=False)]
        a=np.mean([ev(np.mean([Z[i][0] for i in s],0)).mean() for s in subs]); b=np.mean([eh(np.mean([Z[i][1] for i in s],0)).mean() for s in subs])
        res[N]=(a,b)
    Ns=np.array(list(res)); A=np.stack([np.ones(len(Ns)),-1/Ns],1)
    qv,av=np.linalg.lstsq(A,np.array([res[N][0] for N in Ns]),rcond=None)[0]; qh,ah=np.linalg.lstsq(A,np.array([res[N][1] for N in Ns]),rcond=None)[0]
    print(f'== {name} ({n} members)'); 
    for N in Ns: print(f'  N={N:2d} val {res[N][0]:.5f} (fit {qv-av/N:.5f}) hold {res[N][1]:.5f} (fit {qh-ah/N:.5f})')
    print(f'  fit: val Q_inf {qv:.5f} a {av:.5f} | hold Q_inf {qh:.5f} a {ah:.5f} | predicted N=8 {qv-av/8:.5f}/{qh-ah/8:.5f}  N=16 {qv-av/16:.5f}/{qh-ah/16:.5f}  N=32 {qv-av/32:.5f}/{qh-ah/32:.5f}')
    out[name]=dict(curve={int(N):res[N] for N in Ns},qv=qv,av=av,qh=qh,ah=ah)
curve('wide students (kd05w192 + kd05t25w192)',wide)
curve('narrow students (kd05 + kd05t25)',nar)
curve('hard-label base',base)
# --- M3b
def qhat(vs):
    """per-feed Qhat = 2*NDCG(pool of the 2) - mean member NDCG, (val, hold)"""
    Z=[ld(v) for v in vs]; assert len(Z)==2
    return (2*ev((Z[0][0]+Z[1][0])/2)-(ev(Z[0][0])+ev(Z[1][0]))/2, 2*eh((Z[0][1]+Z[1][1])/2)-(eh(Z[0][1])+eh(Z[1][1]))/2)
print('\n== M3b calibration: Qhat(2 seeds) of 4-seed arms, the 6 seed pairs; sd over pairs = run noise of the 2-seed estimator')
cal={}
for a in ('kd05w192','kd05t25w192','kd05','kd05t25','kd05g2w192','kx05','base'):
    q=[qhat(fam(a,p)) for p in itertools.combinations(S4,2)]
    v=[x[0].mean() for x in q]; h=[x[1].mean() for x in q]; cal[a]=(np.mean(v),np.std(v,ddof=1),np.mean(h),np.std(h,ddof=1))
    print(f'  {a:14s} Qhat val {np.mean(v):.5f} sd {np.std(v,ddof=1):.5f} | hold {np.mean(h):.5f} sd {np.std(h,ddof=1):.5f}')
sdv=np.mean([cal[a][1] for a in cal]); sdh=np.mean([cal[a][3] for a in cal]); print(f'  mean sd of one 2-seed Qhat: val {sdv:.5f} hold {sdh:.5f} -> sd of a paired arm-vs-ref difference (indep. seeds noise) ~ val {sdv*2**.5:.5f} hold {sdh*2**.5:.5f}')
print('\n== M3b: 2-seed arms vs kd05w192 (same seeds 951,952): member delta | 2-pool delta | Qhat delta (per-feed SE; add run noise above)')
r=fam('kd05w192',S2); qr=qhat(r); Zr=[ld(v) for v in r]; pr=(ev((Zr[0][0]+Zr[1][0])/2),eh((Zr[0][1]+Zr[1][1])/2)); mr=((ev(Zr[0][0])+ev(Zr[1][0]))/2,(eh(Zr[0][1])+eh(Zr[1][1]))/2)
for a in ('kw_drop0','kw_e20','kw_hl2','kw_ff4','kw_id','kd05t25w256','kd05t25w192b3','kd05t25w192','kd05g2w192','kd03g2w192','kx05','kx08','kd05'):
    x=fam(a,S2); qa=qhat(x); Z=[ld(v) for v in x]; pa=(ev((Z[0][0]+Z[1][0])/2),eh((Z[0][1]+Z[1][1])/2)); ma=((ev(Z[0][0])+ev(Z[1][0]))/2,(eh(Z[0][1])+eh(Z[1][1]))/2)
    print(f'  {a:14s} member {f(ma[0]-mr[0])} / {f(ma[1]-mr[1])} | 2-pool {f(pa[0]-pr[0])} / {f(pa[1]-pr[1])} | Qhat {f(qa[0]-qr[0])} / {f(qa[1]-qr[1])}')
    out[f'q_{a}']=dict(member=((ma[0]-mr[0]).mean(),(ma[1]-mr[1]).mean()),pool2=((pa[0]-pr[0]).mean(),(pa[1]-pr[1]).mean()),qhat=((qa[0]-qr[0]).mean(),(qa[1]-qr[1]).mean()))
json.dump(out,open(OUT/'m3_ncurve.json','w'),indent=1,default=float)
