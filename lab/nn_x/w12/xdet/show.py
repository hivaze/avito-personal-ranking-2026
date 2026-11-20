import numpy as np
s=np.load('out/stats.npz'); z,rho,M,gap=s['z'],s['rho'],s['M'],s['gap']
p=np.load('<repo>/eda/labels_position/out/train_prandom.npz')['p_random']
print('n',len(M),'minM',M.min(),'nan p',np.isnan(p).sum())
for lo,hi in [(1,5),(5,8),(8,12),(12,20),(20,35),(35,50),(50,100),(100,200),(200,301)]:
    m=(M>=lo)&(M<hi)
    if m.sum()==0: continue
    zz=z[m]; kr=m&(p>0.9)&(M>=50); kp=m&(p<0.05)&(M>=50)
    print(f'M[{lo},{hi}) n={m.sum():8d} rho {rho[m].mean():.3f} | z q05 {np.quantile(zz,.05):6.2f} q25 {np.quantile(zz,.25):6.2f} med {np.median(zz):6.2f} q75 {np.quantile(zz,.75):6.2f}| P(z<0) {np.mean(zz<0):.4f} pi_ub {2*np.mean(zz<0):.3f} | gap0 {np.mean(gap[m]<=0):.3f} gap0|z<0 {np.mean(gap[m&(z<0)]<=0):.3f} gap0|z>2 {np.mean(gap[m&(z>2)]<=0):.3f}'
          + (f' | oldrand z {z[kr].mean():.2f}±{z[kr].std():.2f}; oldprod z {z[kp].mean():.2f}±{z[kp].std():.2f}' if kr.sum()>100 else ''))
