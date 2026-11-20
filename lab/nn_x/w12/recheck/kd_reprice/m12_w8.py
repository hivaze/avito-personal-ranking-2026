"""M12: (a) W8 selected-checkpoint vs @last (is holdout selection still doing anything for students?);
 (b) W8 + pop1-view students at TO level (val, holdout, slices, label-free test correlation) and the hard-label analogue (G3 -> Q2a / Q2a8 TO pools)
 whose public deltas are known (+0.000385 / +0.00057); (c) forecast inputs for W16 and GRU from the N-law."""
from lib import *
S4=(951,952,953,954)
W8=fam('kd05w192')+fam('kd05t25w192'); PK=fam('pop1kd05t25',S4,'abl_e10_')
z8=pool(W8); e8=(ev(z8[0]),eh(z8[1]))
print(f'(a) W8 TO selected: val {e8[0].mean():.5f} hold {e8[1].mean():.5f}')
L=[ld(v,'@last') for v in W8]; zl=(np.mean([x[0] for x in L],0),np.mean([x[1] for x in L],0))
print(f'    W8 TO @last   : val {ev(zl[0]).mean():.5f} hold {eh(zl[1]).mean():.5f} | @last - selected: val {f(ev(zl[0])-e8[0])} hold {f(eh(zl[1])-e8[1])} | corr {np.corrcoef(zl[0],z8[0])[0,1]:.5f}')
dm=np.mean([ev(ld(v,"@last")[0])-ev(ld(v)[0]) for v in W8],0); dh=np.mean([eh(ld(v,"@last")[1])-eh(ld(v)[1]) for v in W8],0)
print(f'    member @last - selected: val {f(dm)} hold {f(dh)}')
B=fam('base'); Lb=[ld(v,'@last') for v in B]
dm=np.mean([ev(ld(v,"@last")[0])-ev(ld(v)[0]) for v in B],0); dh=np.mean([eh(ld(v,"@last")[1])-eh(ld(v)[1]) for v in B],0)
print(f'    hard base member @last - selected: val {f(dm)} hold {f(dh)}')
print('(b) W8 + pop1kd x4 (TO level)')
zp=pool(PK)
day=(fv['timestamp'].to_numpy()//86400)
for lab,zz in (('member-equal 8+4 (w=1/3)',(np.mean([ld(v)[0] for v in W8+PK],0),np.mean([ld(v)[1] for v in W8+PK],0))),
               ('w=0.20',(0.8*zv(z8[0])+0.2*zv(zp[0]),0.8*zh(z8[1])+0.2*zh(zp[1]))),('w=0.15',(0.85*zv(z8[0])+0.15*zv(zp[0]),0.85*zh(z8[1])+0.15*zh(zp[1]))),('w=0.10',(0.9*zv(z8[0])+0.1*zv(zp[0]),0.9*zh(z8[1])+0.1*zh(zp[1])))):
    d=ev(zz[0])-e8[0]; dhh=eh(zz[1])-e8[1]
    print(f'    {lab:26s} val {f(d)} (M<50 {f(d[Mv<50])} | M>=50 {f(d[Mv>=50])} | Tue {d[day==20718].mean():+.5f} Thu {d[day==20720].mean():+.5f}) hold {f(dhh)} | corr to W8 {np.corrcoef(zz[0],z8[0])[0,1]:.4f}')
print('    hard-label analogue at TO level (val only): G3 TO -> +pop1 x4 (Q2a, public +0.000385), -> +pop1 x8 (Q2a8, public +0.00057)')
g3=np.load(P/'ens_G3_to_val.npy'); e0=ev(g3)
for t in ('Q2apop1','Q2a8pop1'):
    x=np.load(P/f'ens_{t}_to_val.npy'); d=ev(x)-e0
    ft=ev(0.5*zv(np.load(P/f'ens_{t}_ft_val.npy'))+0.5*zv(x))-ev(0.5*zv(np.load(P/'ens_G3_ft_val.npy'))+0.5*zv(g3))
    print(f'    {t:10s} TO val {f(d)} (M<50 {d[Mv<50].mean():+.5f} M>=50 {d[Mv>=50].mean():+.5f}) | final (FT+TO) val {f(ft)} | corr TO {np.corrcoef(zv(x),zv(g3))[0,1]:.4f}')
