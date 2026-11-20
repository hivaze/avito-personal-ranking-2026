"""R9: label-free test-side check of the W8P20 recipe (0.8 z(W8) + 0.2 z(pop1kd) per branch): corr to W8, by public/private and by day."""
from lo_common import *
ts = ft['timestamp'].to_numpy().astype(np.int64); pub = ts < 1790497920; day = (ts+10800)//86400
fid = np.repeat(np.arange(len(Mt)), Mt)
w_ft, w_to = zt(np.load(P/'ens_W8kd_ft_test.npy')), zt(np.load(P/'ens_W8kd_to_test.npy')); w8 = 0.5*w_ft+0.5*w_to
have = [m for m in POP1KD if any((D_/f'nn_{m}_n2cpv3s3_test.npy').exists() for D_ in (P, W, AB))]
p_ft = zt(np.mean([zt(np.load(ftpath(m, 'test'))) for m in have], 0)); p_to = zt(np.mean([zt(np.load(tpath(m))) for m in POP1KD], 0))
x = 0.5*zt(0.8*w_ft+0.2*p_ft)+0.5*zt(0.8*w_to+0.2*p_to)
def pf_corr(a, b):  # per-feed Pearson corr of the two score vectors
    n = Mt.astype(float); sa = np.bincount(fid, a); sb = np.bincount(fid, b); saa = np.bincount(fid, a*a); sbb = np.bincount(fid, b*b); sab = np.bincount(fid, a*b)
    cov = sab/n-sa*sb/n**2; va = saa/n-(sa/n)**2; vb = sbb/n-(sb/n)**2; return cov/np.sqrt(np.maximum(va*vb, 1e-18))
c = pf_corr(x, w8)
print(f' FT members available {len(have)}/4 | global corr W8P20 vs W8 on test {np.corrcoef(x, w8)[0,1]:.5f}; mean per-feed corr public {c[pub].mean():.5f} private {c[~pub].mean():.5f}')
print(' by day: ' + ' '.join(f'{int(d)}: {c[day==d].mean():.5f}' for d in np.unique(day)))
# same on val for reference
vft, vto = zv(np.load(P/'ens_W8kd_ft_val.npy')), zv(np.load(P/'ens_W8kd_to_val.npy'))
pv_ft = zv(np.mean([zv(np.load(ftpath(m, 'val'))) for m in have], 0)); pv_to = zv(poolv(POP1KD))
xv = 0.5*zv(0.8*vft+0.2*pv_ft)+0.5*zv(0.8*vto+0.2*pv_to); print(f' val global corr {np.corrcoef(xv, 0.5*vft+0.5*vto)[0,1]:.5f}')
