"""M10: structural diversity at FIXED pool size. For families A, X with n members each:
 G_mix = mean NDCG of pools made of n/2 members of A + n/2 of X  -  0.5 * (NDCG of the pure-A n-pool average + pure-X n-pool average),
 with pure pools also of size n (all n/2+n/2 splits; pure = random n-subsets or the full family). Positive G_mix = the two families'
 mean functions differ in a useful way (seed noise cancels: every pool has n members). rho_sys = cross / sqrt(within_A * within_X)."""
from lib import *
import itertools, json
rng=np.random.default_rng(2)
S4=(951,952,953,954)
def gmix(A,X,n,label,nsub=24):
    ZA=[ld(v) for v in A]; ZX=[ld(v) for v in X]; h=n//2
    def pools(Z1,Z2,k1,k2):
        c1=list(itertools.combinations(range(len(Z1)),k1)); c2=list(itertools.combinations(range(len(Z2)),k2)) if k2 else [()]
        allc=[(a,b) for a in c1 for b in c2]
        if len(allc)>nsub: allc=[allc[i] for i in rng.choice(len(allc),nsub,replace=False)]
        v=np.mean([ev(np.mean([Z1[i][0] for i in a]+[Z2[j][0] for j in b],0)) for a,b in allc],0)
        hh=np.mean([eh(np.mean([Z1[i][1] for i in a]+[Z2[j][1] for j in b],0)) for a,b in allc],0)
        return v,hh
    pa=pools(ZA,ZA,n,0); px=pools(ZX,ZX,n,0); pm=pools(ZA,ZX,h,h)
    g=(pm[0]-0.5*(pa[0]+px[0]), pm[1]-0.5*(pa[1]+px[1])); dA=(pm[0]-pa[0],pm[1]-pa[1])
    w=lambda Z,i: np.mean([np.corrcoef(Z[a][i],Z[b][i])[0,1] for a,b in itertools.combinations(range(len(Z)),2)])
    cx=lambda i: np.mean([np.corrcoef(a[i],x[i])[0,1] for ia,a in enumerate(ZA) for ix,x in enumerate(ZX) if A[ia].rsplit('_',1)[1]!=X[ix].rsplit('_',1)[1]])
    rs=[cx(i)/np.sqrt(w(ZA,i)*w(ZX,i)) for i in (0,1)]
    print(f'{label:34s} n={n} | pure A {pa[0].mean():.5f}/{pa[1].mean():.5f} pure X {px[0].mean():.5f}/{px[1].mean():.5f} mix {pm[0].mean():.5f}/{pm[1].mean():.5f} | G_mix {f(g[0])} / {f(g[1])} | mix - pureA {f(dA[0])} / {f(dA[1])} | rho_sys {rs[0]:.4f}/{rs[1]:.4f}',flush=True)
    return dict(g=(g[0].mean(),se(g[0]),g[1].mean(),se(g[1])),dA=(dA[0].mean(),se(dA[0]),dA[1].mean(),se(dA[1])),rho=rs)
out={}
F7=[f'v17_pre2s_s{s}' for s in range(101,107)]
print('students (val / holdout)')
out['wide11x'] =gmix(fam('kd05w192'),fam('kd05t25w192'),4,'kd05w192 x kd05t25w192 (teacher)')
out['wide_oof']=gmix(fam('kd05w192'),fam('kx05'),4,'kd05w192 x kx05 (OOF teacher)')
out['wide_g2'] =gmix(fam('kd05w192'),fam('kd05g2w192'),4,'kd05w192 x kd05g2w192 (gen 2)')
out['wide_nar']=gmix(fam('kd05w192'),fam('kd05'),4,'kd05w192 x kd05 (width)')
out['kd05_03'] =gmix(fam('kd05'),fam('kd03'),4,'kd05 x kd03 (kd weight)')
out['kd05_08'] =gmix(fam('kd05'),fam('kd08'),4,'kd05 x kd08 (kd weight)')
out['tf_pop1kd']=gmix(fam('kd05t25'),fam('pop1kd05t25',S4,'abl_e10_'),4,'kd05t25 x pop1kd (view, students)')
out['wide_pop1kd']=gmix(fam('kd05t25w192'),fam('pop1kd05t25',S4,'abl_e10_'),4,'kd05t25w192 x pop1kd (view+width)')
out['wide_base']=gmix(fam('kd05w192'),fam('base'),4,'kd05w192 x hard base')
print('hard-label families')
out['base_pop1']=gmix(fam('base'),fam('pop1',S4,'abl_e10_'),4,'base x pop1 (view, hard)')
out['base_e20']=gmix(fam('base'),fam('e20'),4,'base x e20 (hard)')
out['F7_gru']=gmix(F7,[f'v23_gru_pre2s_s{s}' for s in range(501,507)],6,'F7 x gruF10 (arch, hard)')
out['F7_lr1']=gmix(F7,[f'v22_pre2s_lr1_s{s}' for s in range(401,407)],6,'F7 x lr1F9 (lr, hard)')
out['F7_d192']=gmix(F7,[f'v26_d192_pre2s_s{s}' for s in range(701,707)],6,'F7 x d192F12 (width, hard)')
out['F7_a4u']=gmix(F7,[f'r6_a4u_e10_s{s}' for s in range(101,107)],6,'F7 x a4u (retrieval, hard)')
out['F7_cset']=gmix(F7,[f'r6_cset_e10_s{s}' for s in range(101,107)],6,'F7 x cset (cand retrieval, hard)')
out['m02_p70k']=gmix([f'r6_cd_m02_r{i}' for i in range(101,107)],[f'r6_cd_m02_p70k_r{i}' for i in range(101,107)],6,'F7x m02 x F7x70 (snapshot, hard)')
json.dump(out,open(OUT/'m10_gmix.json','w'),indent=1,default=float)
