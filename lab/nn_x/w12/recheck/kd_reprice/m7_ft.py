"""M7: what does the val fine-tune (n2cpv3s3, OOF) do to students vs hard-label members? Member FT-TO deltas, pool FT/TO/blend,
w_ft scan for W8 / S16 / C3 (val OOF only; the holdout cannot see FT), slices by M."""
from lib import *
import json
S4=(951,952,953,954)
G={'base (hard)':fam('base'),'kd05':fam('kd05'),'kd05t25':fam('kd05t25'),'kd05w192':fam('kd05w192'),'kd05t25w192':fam('kd05t25w192'),'kd08':fam('kd08'),
   'pop1 (hard)':fam('pop1',S4,'abl_e10_'),'gruF10 (hard)':[f'v23_gru_pre2s_s{s}' for s in range(501,507)],'F7x m02 (hard)':[f'r6_cd_m02_r{i}' for i in range(101,108)]}
print(f'{"family":18s} n | member TO   FT    | member FT-TO (paired)  M<50 | M>=50 | pool TO   FT    | pool FT-TO          | 0.5/0.5   | blend - TO pool')
out={}
for n,vs in G.items():
    T=[ld(v)[0] for v in vs]; Fz=[ldft(v) for v in vs]
    if any(x is None for x in Fz):
        Fz=[ldft(v,'n2cpv3') for v in vs]
    if any(x is None for x in Fz) or any(x is None for x in T): print(f'{n:18s} FT or TO preds missing'); continue
    eT=[ev(t) for t in T]; eF=[ev(x) for x in Fz]; d=np.mean([a-b for a,b in zip(eF,eT)],0)
    pT=np.mean(T,0); pF=np.mean(Fz,0); ePT=ev(pT); ePF=ev(pF); bl=ev(0.5*zv(pF)+0.5*zv(pT))
    print(f'{n:18s} {len(vs)} | {np.mean([e.mean() for e in eT]):.5f} {np.mean([e.mean() for e in eF]):.5f} | {f(d)} {d[Mv<50].mean():+.5f} | {d[Mv>=50].mean():+.5f} | {ePT.mean():.5f} {ePF.mean():.5f} | {f(ePF-ePT)} | {bl.mean():.5f} | {f(bl-ePT)}')
    out[n]=dict(member_ft_minus_to=d.mean(),se=se(d),pool_to=ePT.mean(),pool_ft=ePF.mean(),blend=bl.mean())
print('\n w_ft scan: final = w z(FT pool) + (1-w) z(TO pool)')
for tag in ('W8kd','S16kd','Q2a8s3','G3'):
    ft=np.load(P/f'ens_{tag}_ft_val.npy'); to=np.load(P/f'ens_{tag}_to_val.npy'); e0=ev(0.5*zv(ft)+0.5*zv(to))
    row=[]
    for w in (0,0.15,0.3,0.5,0.7,1.0):
        e=ev(w*zv(ft)+(1-w)*zv(to)); row.append(f'w={w:.2f} {e.mean():.5f} ({f(e-e0)}; M<50 {(e-e0)[Mv<50].mean():+.5f} M>=50 {(e-e0)[Mv>=50].mean():+.5f})')
    print(f' {tag:7s} corr(FT,TO) {np.corrcoef(zv(ft),zv(to))[0,1]:.4f} | '+' | '.join(row))
json.dump(out,open(OUT/'m7_ft.json','w'),indent=1,default=float)
