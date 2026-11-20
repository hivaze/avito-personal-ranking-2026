"""M4: which families ADD to a saturated wide-student pool? Reference A = z-mean of the 11 finished wide students
(kd05w192 s951-955 + kd05t25w192 s951-955,957; N-law says its noise term a/N = 0.0002, so mixing gains > ~0.0001 are systematic).
For each family X (z-mean of its members): rho_sys(A,X) = c(A,X)/sqrt(c(A,A) c(X,X)) from member-level cross / within correlations,
and the paired per-feed delta of (1-w) zA + w zX vs zA on val and holdout for w in {0.15, 0.3, 0.5}; plus member-equal union.
"""
from lib import *
import itertools, json, glob
S4=(951,952,953,954); S2=(951,952)
wide=fam('kd05w192',(951,952,953,954,955))+fam('kd05t25w192',(951,952,953,954,955,957))
def r6(n,ids,p='r'): return [f'r6_{n}_{p}{i}' for i in ids]
F={ 'narrow8 (kd05+kd05t25)':fam('kd05')+fam('kd05t25'),'kd03':fam('kd03'),'kd08':fam('kd08'),'kd10':fam('kd10'),
 'kd05g2w192':fam('kd05g2w192'),'kd03g2w192':fam('kd03g2w192'),'kx05':fam('kx05'),'kx08':fam('kx08'),
 'kw_drop0':fam('kw_drop0',S2),'kw_e20':fam('kw_e20',S2),'kw_hl2':fam('kw_hl2',S2),'kw_ff4':fam('kw_ff4',S2),'kw_id':fam('kw_id',S2),'w256':fam('kd05t25w256',S2),'w192b3':fam('kd05t25w192b3',S2),
 'pop1kd (student, pop1 view)':fam('pop1kd05t25',S4,'abl_e10_'),
 'HARD base6':fam('base',(951,952,953,954,955,956)),'HARD e20':fam('e20'),'HARD e15':fam('e15',S2),'HARD bs4':fam('bs4',S2),'HARD bs4lr4':fam('bs4lr4',S2),'HARD ap1':fam('ap1',S2),'HARD pseudo':fam('pseudo',S2),'HARD trunc':fam('trunc'),
 'HARD pop1 x8':fam('pop1',tuple(range(951,959)),'abl_e10_'),'HARD pop2':fam('pop2',S4,'abl_e10_'),'HARD popfree':fam('xdrop_pop',S4,'abl_e10_'),
 'HARD retr32':fam('retr32',S2,'abl_e10_'),'HARD retr32u':fam('retr32u',S2,'abl_e10_'),'HARD xrel_more':fam('xrel_more',S2,'abl_e10_'),'HARD blocks3':fam('blocks3',S2,'abl_e10_'),'HARD latedrop':fam('latedrop',S2,'abl_e10_'),'HARD xdrop_pre2':fam('xdrop_pre2',S2,'abl_e10_'),
 'HARD gruF10':[f'v23_gru_pre2s_s{s}' for s in range(501,507)],'HARD F7':[f'v17_pre2s_s{s}' for s in range(101,108)],'HARD lr1F9':[f'v22_pre2s_lr1_s{s}' for s in range(401,407)],'HARD d192F12':[f'v26_d192_pre2s_s{s}' for s in range(701,708)],
 'HARD F7x m02':r6('cd_m02',range(101,108)),'HARD F7x70':r6('cd_m02_p70k',range(101,108)),'HARD a4u_e10':r6('a4u_e10',range(101,107),'s'),'HARD cset_e10':r6('cset_e10',range(101,107),'s'),'HARD m02_a4u':r6('cd_m02_a4u',range(101,107)),
 'HARD cd_rdrop':[f'w5_cd_rdrop_r{i}' for i in range(101,105)],'HARD cd_sam':[f'w5_cd_sam_r{i}' for i in range(101,105)],'HARD cd_hzcd':[f'w5_cd_hzcd_r{i}' for i in range(101,105)],'HARD cd_m04':[f'w5_cd_m04_r{i}' for i in range(101,105)],
 'HARD cd_d0':r6('cd_d0',range(101,105)),'HARD cd_d20':r6('cd_d20',range(101,105)),'HARD cd_hcap':r6('cd_hcap',range(101,105)),'HARD cd_aug':r6('cd_aug',range(101,105)),'HARD cd_rec33':r6('cd_rec33',range(101,105)),'HARD cd_m0':r6('cd_m0',range(101,108)),
}
def within(vs,k):
    return np.mean([np.corrcoef(ld(a)[k],ld(b)[k])[0,1] for a,b in itertools.combinations(vs,2)])
def cross(A,X,k):
    return np.mean([np.corrcoef(ld(a)[k],ld(x)[k])[0,1] for a in A for x in X if a.rsplit('_',1)[1]!=x.rsplit('_',1)[1] or a.split('_s')[0]==x.split('_s')[0]])
zA=pool(wide); eA=(ev(zA[0]),eh(zA[1])); cAA=(within(wide,0),within(wide,1))
mA=(np.mean([ev(ld(v)[0]).mean() for v in wide]),np.mean([eh(ld(v)[1]).mean() for v in wide]))
print(f'reference wide11: pool val {eA[0].mean():.5f} hold {eA[1].mean():.5f}; member {mA[0]:.5f}/{mA[1]:.5f}; within corr {cAA[0]:.4f}/{cAA[1]:.4f}')
print(f'{"family":28s} n  member d(val/hold) vs wide | pool X val/hold | rho_sys v/h   | w=0.15 val | hold         | w=0.30 val | hold         | w=0.50 val | hold')
out={}
for n,vs in F.items():
    Z=[ld(v) for v in vs]
    if any(z[0] is None or z[1] is None for z in Z): print(f'{n:28s} MISSING {[v for v,z in zip(vs,Z) if z[0] is None or z[1] is None][:2]}'); continue
    zX=pool(vs); mX=(np.mean([ev(z[0]).mean() for z in Z]),np.mean([eh(z[1]).mean() for z in Z]))
    cXX=(within(vs,0),within(vs,1)); cAX=(cross(wide,vs,0),cross(wide,vs,1)); rs=tuple(cAX[k]/np.sqrt(cAA[k]*cXX[k]) for k in (0,1))
    row=[]; d={}
    for w in (0.15,0.3,0.5):
        dv=ev((1-w)*zv(zA[0])+w*zv(zX[0]))-eA[0]; dh=eh((1-w)*zh(zA[1])+w*zh(zX[1]))-eA[1]; row.append(f'{f(dv)} | {f(dh)}'); d[w]=(dv.mean(),se(dv),dh.mean(),se(dh))
    out[n]=dict(n=len(vs),member=mX,poolX=(ev(zX[0]).mean(),eh(zX[1]).mean()),within=cXX,cross=cAX,rho_sys=rs,mix=d)
    print(f'{n:28s} {len(vs):2d} {mX[0]-mA[0]:+.5f}/{mX[1]-mA[1]:+.5f} | {ev(zX[0]).mean():.5f}/{eh(zX[1]).mean():.5f} | {rs[0]:.4f}/{rs[1]:.4f} | '+' | '.join(row))
json.dump(out,open(OUT/'m4_mix.json','w'),indent=1,default=float)
