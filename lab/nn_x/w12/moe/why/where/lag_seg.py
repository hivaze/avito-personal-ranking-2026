"""why/where W2a by segment: on the holdout, is the arms' lag0 -> priv_v3 loss of edge a composition effect (more feeds whose positives
look popularity-cold under test-lag inputs) or a within-segment effect? Uses out/lag_perfeed.npz (lag_read.py) and the poplag priv_v3
it_hist_all_luo column. CPU, read-only. Output out/lag_seg.txt"""
from common_where import *
m = Meta()
Q = dict(np.load(HERE / 'out/lag_perfeed.npz'))
sh = dict(np.load(HERE / 'out/seg_hold.npz'))
PL = ROOT / 'lab/nn_x/w12/next/poplag/cache'
yh = m.yh; fid = np.repeat(np.arange(len(m.Mh)), m.Mh); pos = yh > 0; npos = np.bincount(fid, weights=pos, minlength=len(m.Mh))
lagpop = np.load(PL / 'lag_priv_v3_it_hist_all_luo.npy')
pp0_lag = np.bincount(fid, weights=(np.nan_to_num(lagpop, nan=0) <= 0) * pos, minlength=len(m.Mh)) / npos
pp0 = sh['pos_pop0']
arm = lambda v, ref: np.mean([Q[f'{v}|{k}'] for k in ARMS], 0) - Q[f'{v}|{ref}']
L = ['=== W2a by segment (holdout, 30,000 feeds; arm = mean of A,B,C 4-pools minus refN; +- = 1 SE paired per feed) ===']
L.append(f'share of feeds whose positives are all popularity-cold (it_hist_all_luo == 0): lag0 {np.mean(pp0 >= 1):.3f}, priv_v3 {np.mean(pp0_lag >= 1):.3f}; any positive cold: lag0 {np.mean(pp0 > 0):.3f}, priv_v3 {np.mean(pp0_lag > 0):.3f}  (val: see seg_read: all-cold 0.237)')
for ref in ('refN', 'ref4'):
    d0, d1 = arm('lag0', ref), arm('priv_v3', ref)
    L.append(f'-- reference {ref}: ALL lag0 {fmt(d0)} priv {fmt(d1)} DiD {fmt(d1 - d0)}')
    for lbl, seg in (('positives all pop>0 at lag0', pp0 <= 0), ('some/all positives pop=0 at lag0', pp0 > 0),
                     ('positives all pop>0 under priv_v3', pp0_lag <= 0), ('some/all positives pop=0 under priv_v3', pp0_lag > 0),
                     ('pop>0 at lag0 AND pop=0 under priv (made cold by lag)', (pp0 <= 0) & (pp0_lag > 0)),
                     ('positives all seen in training', sh['pos_seen'] >= 1), ('no positive seen in training', sh['pos_seen'] <= 0)):
        L.append(f'   {lbl:58s} n {int(seg.sum()):6d} | lag0 {fmt(d0[seg])} | priv_v3 {fmt(d1[seg])} | DiD {fmt((d1 - d0)[seg])}')
# composition check: priv_v3 delta predicted from lag0 segment deltas reweighted to the priv_v3 pop-cold mix
d0 = arm('lag0', 'refN'); d1 = arm('priv_v3', 'refN')
a0, a1 = d0[pp0_lag <= 0].mean(), d0[pp0_lag > 0].mean()
L.append(f'-- composition: lag0 deltas split by the priv_v3 pop-cold status: warm {a0:+.5f}, cold {a1:+.5f}; observed priv_v3 delta {d1.mean():+.5f} '
         f'(lag0 delta on the same feeds {d0.mean():+.5f}): within-feed change carries the DiD when the split by lagged status already shows it at lag0')
txt = '\n'.join(L); print(txt); open(HERE / 'out/lag_seg.txt', 'w').write(txt + '\n')
