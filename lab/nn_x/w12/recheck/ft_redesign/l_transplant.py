"""L. (1) FT-pool level vs number of members (is the FT branch capped by its common-mode part?);
 (2) shift transplant: z(TO pool W8) + pooled FT shift of ANOTHER family (does cross-family averaging de-noise the shift?)."""
from lib import *
rng = np.random.default_rng(0)
print('(1) W8 students: pool NDCG vs number of members k (mean over 20 random subsets; all 8 for k = 8), FT branch / TO branch / file')
ZF = np.array([z(m, 'ft', 'val') for m in W8 + N8]); ZT = np.array([z(m, 'to', 'val') for m in W8 + N8])
for k in (1, 2, 4, 8, 16):
    r = []
    for _ in range(1 if k == 16 else 20):
        i = rng.choice(16, k, replace=False); pf, pt = zv(ZF[i].mean(0)), zv(ZT[i].mean(0))
        r.append((ev(pf).mean(), ev(pt).mean(), ev(0.5 * pf + 0.5 * pt).mean()))
    r = np.mean(r, 0); print(f'  k {k:2d} (of the 16 students): FT {r[0]:.5f} TO {r[1]:.5f} file {r[2]:.5f} | FT - TO {r[0]-r[1]:+.5f} | file - TO {r[2]-r[1]:+.5f}')
print('\n(2) shift transplant onto the W8 TO pool')
S = {k: zv(pool(ms, 'ft', 'val')) - zv(pool(ms, 'to', 'val')) for k, ms in (('W8', W8), ('N8', N8), ('C3', C3M), ('kd08', GROUPS['S:kd08']))}
a = zv(pool(W8, 'to', 'val')); e0 = ev(a); base1, baseh = ev(a + S['W8']), ev(a + 0.5 * S['W8'])
for nm, s in (('own (W8)', S['W8']), ('N8', S['N8']), ('C3', S['C3']), ('mean(W8, N8)', (S['W8'] + S['N8']) / 2), ('mean(W8, N8, C3, kd08)', (S['W8'] + S['N8'] + S['C3'] + S['kd08']) / 4), ('mean(N8, C3, kd08) [no own]', (S['N8'] + S['C3'] + S['kd08']) / 3)):
    e1, eh = ev(a + s), ev(a + 0.5 * s)
    print(f'  shift {nm:28s} RMS {np.sqrt((s**2).mean()):.4f} | lam 1: vs TO {f(e1 - e0)} (vs own {f(e1 - base1)}) | lam 0.5: vs TO {f(eh - e0)} (vs own {f(eh - baseh)})')
