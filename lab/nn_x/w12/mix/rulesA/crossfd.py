"""ADDENDUM 5: which pool carries the short-feed difference between full-data members and twins? (post hoc, descriptive)"""
from lib import *
f, M, off = meta('val'); y = labels(); sh = M < 50; lg = ~sh; cache = {}
P_ = {k: np.load(HERE / f'pools_{k}_val.npy') for k in ('C1', 'C3', 'C2', 'C4')}
def rd(tag, pools):
    N = table(pools, y); d4 = N[:, cidx(4, 4, C05)] - N[:, PLAIN]; d3 = N[:, cidx(3, 3, C05)] - N[:, PLAIN]
    print(f'{tag:46s}: a .4 - .2  M<50 {fmt(d4[sh])}  M>=50 {fmt(d4[lg])} all {fmt(d4)} | a .3 - .2  M<50 {fmt(d3[sh])} | plain {N[:, PLAIN].mean():.5f}', flush=True); return d4
r = {}
r['C1'] = rd('C1  FD16 base + FDP (full-data pop-light)', tuple(P_['C1'])); r['X1'] = rd('X1  FD16 base + P1w (held-out pop-light)', (P_['C1'][0], P_['C1'][1], P_['C3'][2], P_['C3'][3]))
r['X2'] = rd('X2  W16 base + FDP (full-data pop-light)', (P_['C3'][0], P_['C3'][1], P_['C1'][2], P_['C1'][3])); r['C3'] = rd('C3  W16 base + P1w (held-out pop-light)', tuple(P_['C3']))
print(f'paired on M<50: pop-light pool effect (FDP minus P1w): with FD16 base {fmt((r["C1"] - r["X1"])[sh])}, with W16 base {fmt((r["X2"] - r["C3"])[sh])} | base pool effect (FD16 minus W16): with FDP {fmt((r["C1"] - r["X2"])[sh])}, with P1w {fmt((r["X1"] - r["C3"])[sh])}')
r['C2'] = rd('C2  FD8 base + FDP', tuple(P_['C2'])); r['X3'] = rd('X3  FD8 base + P1w', (P_['C2'][0], P_['C2'][1], P_['C4'][2], P_['C4'][3])); r['X4'] = rd('X4  W8 base + FDP', (P_['C4'][0], P_['C4'][1], P_['C2'][2], P_['C2'][3])); r['C4'] = rd('C4  W8 base + P1w', tuple(P_['C4']))
print(f'paired on M<50 (8-base): pop-light pool effect: with FD8 base {fmt((r["C2"] - r["X3"])[sh])}, with W8 base {fmt((r["X4"] - r["C4"])[sh])} | base pool effect: with FDP {fmt((r["C2"] - r["X4"])[sh])}, with P1w {fmt((r["X3"] - r["C4"])[sh])}')
