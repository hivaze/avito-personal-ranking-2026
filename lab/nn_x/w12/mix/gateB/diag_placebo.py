"""Diagnostic (fixed contrasts, nothing optimised): dose curve of the '4-member pool' weight by M side, real compositions vs the four same-recipe pool placebos."""
from lib2 import *
y = labels().astype(np.float32); f, M, off = meta('val'); g = groups(); j5 = IF[0.5]
ws = [0.0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.8]
for side, m in (('M < 50', M < 50), ('M >= 50', M >= 50), ('all', M > 0)):
    print(f'--- {side} (n {m.sum()}): E(w, 0.5) - E(0.25, 0.5) x 1e5 (+- user-clustered SE of the cell mean)')
    print(f'{"run":8s} ' + ' '.join(f'{w:>9.2f}' for w in ws))
    for run in ('FD16P4', 'W16P4', 'PL1', 'PL2', 'PL3', 'PL4'):
        E = np.load(CACHE / f'E_{run}.npy'); ref = E[m][:, IP[0.25], j5]; row = []
        for w in ws:
            mu, se = cluster_se(E[m][:, IP[w], j5] - ref, g[m]); row.append(f'{mu * 1e5:+4.0f}+-{se * 1e5:2.0f}')
        print(f'{run:8s} ' + ' '.join(f'{r:>9s}' for r in row))
    Es = [np.load(CACHE / f'E_PL{i}.npy') for i in (1, 2, 3, 4)]
    row = []
    for w in ws:
        d = np.mean([E[m][:, IP[w], j5] - E[m][:, IP[0.25], j5] for E in Es], 0); mu, se = cluster_se(d, g[m]); row.append(f'{mu * 1e5:+4.0f}+-{se * 1e5:2.0f}')
    print(f'{"PL mean":8s} ' + ' '.join(f'{r:>9s}' for r in row))
