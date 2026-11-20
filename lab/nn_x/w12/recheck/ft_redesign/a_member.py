"""A. Per-member FT (n2cpv3s3 OOF) minus TO, paired per feed, with slices. Students vs hard-label members (identical FT recipe)."""
from lib import *
import datetime
day = (tv.astype(np.int64) + 3 * 3600) // 86400; dd = np.unique(day)  # MSK dates: Tue Sep 22 (7,542 feeds), Thu Sep 24 (6,878), 3 stray feeds
SL = {'all': np.ones(len(Mv), bool), 'M<50': Mv < 50, 'M>=50': Mv >= 50, 'M<20': Mv < 20, 'M20-49': (Mv >= 20) & (Mv < 50), 'M50-99': (Mv >= 50) & (Mv < 100), 'M>=100': Mv >= 100,
      'user0': uv == 0, 'known': uv != 0, 'H0': Hv == 0, 'H1-20': (Hv >= 1) & (Hv <= 20), 'H21-100': (Hv > 20) & (Hv <= 100), 'H101-255': (Hv > 100) & (Hv < 256), 'H256': Hv >= 256}
SL['Tue'] = day == dd[0]; SL['Thu'] = day == dd[2]
npos = np.add.reduceat(yv, ov); SL['npos1'] = npos == 1; SL['npos>=2'] = npos >= 2
print('slice sizes:', {k: int(v.sum()) for k, v in SL.items()})
D = {}; E = {}
for g, ms in GROUPS.items():
    for m in ms:
        e_to = ev(raw(m, 'to', 'val')); e_ft = ev(raw(m, 'ft', 'val')); D[m] = e_ft - e_to; E[m] = (e_to.mean(), e_ft.mean())
np.save(OUT / 'member_ft_minus_to.npy', np.array([D[m] for g in GROUPS for m in GROUPS[g]], np.float32))
print('\n== member level: FT OOF - TO (mean over members of the per-feed delta; +- = paired SE over feeds of the member-mean delta)')
print(f"{'group':15s} n  TO      FT      " + ' '.join(f'{k:>16s}' for k in SL))
for g, ms in list(GROUPS.items()) + [('S:W8', W8), ('S:N8', N8), ('H:C3x28', C3M)]:
    d = np.mean([D[m] for m in ms], 0)
    print(f"{g:15s} {len(ms):2d} {np.mean([E[m][0] for m in ms]):.5f} {np.mean([E[m][1] for m in ms]):.5f} " + ' '.join(f'{f(d[s]):>16s}' for s in SL.values()))
print('\n== per member (students): all | M<50 | M>=50')
for g in ('S:kd05w192', 'S:kd05t25w192', 'S:kd05', 'S:kd05t25', 'S:kd08'):
    for m in GROUPS[g]: print(f' {m:28s} TO {E[m][0]:.5f} FT {E[m][1]:.5f} {f(D[m])} | {f(D[m][Mv<50])} | {f(D[m][Mv>=50])}')
# difference of FT gain: students vs hard-label (paired per feed)
dW8 = np.mean([D[m] for m in W8], 0); dN8 = np.mean([D[m] for m in N8], 0); dH = np.mean([D[m] for m in C3M], 0); dM = np.mean([D[m] for m in GROUPS['H:m02']], 0)
print('\n== FT gain, students minus hard-label (paired per feed):')
for nm, a, b in (('W8 - C3x28', dW8, dH), ('W8 - m02x7', dW8, dM), ('N8 - C3x28', dN8, dH), ('N8 - m02x7', dN8, dM), ('W8 - N8', dW8, dN8)):
    d = a - b; print(f' {nm:12s} all {f(d)} | M<50 {f(d[Mv<50])} | M>=50 {f(d[Mv>=50])}')
