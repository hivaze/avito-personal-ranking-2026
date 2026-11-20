"""Joint distribution of proxy z and gap share on the 04b held-out sample (M>=50)."""
from common import *




F = feeds('train'); off = F['cand_off'].to_numpy(); M = F['M'].to_numpy()
d = np.load(f'{OUT}/04b_detect.npz'); sel = d['sel']; z = d['z']; Ms = d['M']
dd, o2, M2 = gather('train', sel, F, ['pos'])
ngap, relgap, gt10 = gap_stats(dd['pos'].astype(np.int64), o2, M2)
gs = ngap / (ngap + M2)
np.savez(f'{OUT}/04f_joint.npz', z=z, gs=gs, M=M2, sel=sel)
m = M2 >= 50
print('gap share quantiles (M>=50):', np.quantile(gs[m], [.05, .1, .25, .5, .75, .9]).round(3))
h, _ = np.histogram(gs[m], np.linspace(0, 0.5, 26)); print('gap share hist', (h / h.sum() * 1000).astype(int).tolist())
zb = [-9, -1, 0, 0.5, 1, 1.5, 2, 3, 99]
gb = [0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.15, 0.2, 1]
H2, _, _ = np.histogram2d(z[m], gs[m], [zb, gb])
print('rows z bins', zb, 'cols gap-share bins', gb)
print((H2 / H2.sum(1, keepdims=True) * 1000).astype(int))
print('row counts', H2.sum(1).astype(int))
# P(z<0) by gap-share bin
for a, b in zip(gb[:-1], gb[1:]):
    mm = m & (gs >= a) & (gs < b)
    print(f'gap share [{a},{b}) n={mm.sum()} P(z<0)={np.mean(z[mm] < 0):.3f} median z={np.median(z[mm]):.2f}')
