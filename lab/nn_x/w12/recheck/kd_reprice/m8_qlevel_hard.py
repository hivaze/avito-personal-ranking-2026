"""M8: pool-level (mean-function) re-read of hard-label closures that were judged at member level.
 (a) 10-epoch arms with 2-4 seeds vs w5_e10_base on the same seeds: member delta, N-pool delta, Qhat delta (N=2 pairs averaged).
 (b) cooldown (CD harness) arms, roots 101-104, vs the m02 reference on the same roots: member delta, 4-pool delta.
 (c) old 6-7 seed families vs F7: N-curve Q_inf."""
from lib import *
import itertools, json
out={}
def members(vs): return [ld(v) for v in vs]
def stats(A,R,label):
    ZA=members(A); ZR=members(R)
    if any(z[0] is None or z[1] is None for z in ZA+ZR): print(f'{label:26s} MISSING'); return
    n=len(A)
    dm=(np.mean([ev(a[0])-ev(r[0]) for a,r in zip(ZA,ZR)],0),np.mean([eh(a[1])-eh(r[1]) for a,r in zip(ZA,ZR)],0))
    pa=(ev(np.mean([z[0] for z in ZA],0)),eh(np.mean([z[1] for z in ZA],0))); pr=(ev(np.mean([z[0] for z in ZR],0)),eh(np.mean([z[1] for z in ZR],0)))
    dp=(pa[0]-pr[0],pa[1]-pr[1])
    # Q extrapolation with the 1/N law: Q = m + (pool_N - m) * N/(N-1)
    k=n/(n-1); dq=(dm[0]+(dp[0]-dm[0])*k, dm[1]+(dp[1]-dm[1])*k)
    cw=lambda Z,i: np.mean([np.corrcoef(Z[a][i],Z[b][i])[0,1] for a,b in itertools.combinations(range(len(Z)),2)])
    print(f'{label:26s} {n} | member {f(dm[0])} / {f(dm[1])} | {n}-pool {f(dp[0])} / {f(dp[1])} | Q(1/N law) {f(dq[0])} / {f(dq[1])} | corr {cw(ZA,0):.4f} vs {cw(ZR,0):.4f}')
    out[label]=dict(n=n,member=(dm[0].mean(),dm[1].mean()),pool=(dp[0].mean(),se(dp[0]),dp[1].mean(),se(dp[1])),q=(dq[0].mean(),dq[1].mean()))
S4=(951,952,953,954); S2=(951,952)
print('(a) 10-epoch hard-label arms vs w5_e10_base (same seeds); val / holdout')
for a,sd,pre in (('e20',S4,'w5_e10_'),('trunc',S4,'w5_e10_'),('trunc2',S4,'w5_e10_'),('e15',S2,'w5_e10_'),('ap1',S2,'w5_e10_'),('pseudo',S2,'w5_e10_'),('bs4',S2,'w5_e10_'),('bs4lr4',S2,'w5_e10_'),
                 ('retr32',S2,'abl_e10_'),('retr32u',S2,'abl_e10_'),('xrel_more',S2,'abl_e10_'),('blocks3',S2,'abl_e10_'),('latedrop',S2,'abl_e10_'),('xdrop_pre2',S2,'abl_e10_'),('pop1',S4,'abl_e10_'),('pop2',S4,'abl_e10_'),('xdrop_pop',S4,'abl_e10_')):
    stats(fam(a,sd,pre),fam('base',sd),a)
print('(b) cooldown arms (CD harness, roots 101-104) vs m02 reference on the same roots')
R4=range(101,105)
for a in ('rdrop','sam','hzcd','m01','m04','aa0','ap1','auxf','cbs05','cbs2','cbs025','fpcd','truncp','truncp2','xp2','xp2m05','xp2m10'):
    stats([f'w5_cd_{a}_r{i}' for i in R4],[f'w5_cd_m02ref_r{i}' for i in R4],'w5_cd_'+a)
for a in ('d0','d20','hcap','aug','rec33','rho70','pp02','s2c2','e3','cf4_p60k','aa5','w10','m0','c0','m02_p70k'):
    stats([f'r6_cd_{a}_r{i}' for i in R4],[f'r6_cd_m02_r{i}' for i in R4],'r6_cd_'+a)
print('(c) old families (6 seeds each, first 6 of F7) vs F7: equal-N pools')
F7=[f'v17_pre2s_s{s}' for s in range(101,107)]
for n,vs in (('gruF10',[f'v23_gru_pre2s_s{s}' for s in range(501,507)]),('lr1F9',[f'v22_pre2s_lr1_s{s}' for s in range(401,407)]),('d192F12',[f'v26_d192_pre2s_s{s}' for s in range(701,707)]),
             ('a4u_e10',[f'r6_a4u_e10_s{s}' for s in range(101,107)]),('cset_e10',[f'r6_cset_e10_s{s}' for s in range(101,107)])):
    stats(vs,F7,n)
stats([f'r6_cd_m02_a4u_r{i}' for i in range(101,107)],[f'r6_cd_m02_r{i}' for i in range(101,107)],'m02_a4u vs F7x m02')
stats([f'r6_cd_m02_r{i}' for i in range(101,107)],F7,'F7x m02 vs F7')
json.dump(out,open(OUT/'m8_qlevel_hard.json','w'),indent=1,default=float)
