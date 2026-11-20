"""M16: T1 file recipe at the FINAL level (0.5 z(FT) + 0.5 z(TO)), val only (the holdout has no FT branch).
 Variants: (a) pop1kd TO admixed into the TO branch only (FT branch = W8 FT, zero GPU); (b) + hard pop1 x8 FT pool admixed into the FT branch
 (zero GPU proxy for the missing pop1kd FT). Reference = W8 final (0.40165). Label-free: test correlation to the W8 final."""
from lib import *
S4=(951,952,953,954)
ft=np.load(P/'ens_W8kd_ft_val.npy'); to=np.load(P/'ens_W8kd_to_val.npy'); e0=ev(0.5*zv(ft)+0.5*zv(to)); print(f'W8 final val {e0.mean():.5f}')
pk=np.mean([ld(v)[0] for v in fam('pop1kd05t25',S4,'abl_e10_')],0)
hp_ft=np.mean([ldft(v) for v in fam('pop1',tuple(range(951,959)),'abl_e10_')],0)
day=(fv['timestamp'].to_numpy()//86400)
for w in (0.10,0.15,0.20,0.30):
    toa=(1-w)*zv(to)+w*zv(pk)
    a=ev(0.5*zv(ft)+0.5*zv(toa))-e0
    b=ev(0.5*zv((1-w)*zv(ft)+w*zv(hp_ft))+0.5*zv(toa))-e0
    print(f' w={w:.2f}: (a) TO-branch only {f(a)} (M<50 {a[Mv<50].mean():+.5f} M>=50 {a[Mv>=50].mean():+.5f}; Tue {a[day==20718].mean():+.5f} Thu {a[day==20720].mean():+.5f}) | (b) + hard-pop1 FT in the FT branch {f(b)} (M<50 {b[Mv<50].mean():+.5f} M>=50 {b[Mv>=50].mean():+.5f})')
