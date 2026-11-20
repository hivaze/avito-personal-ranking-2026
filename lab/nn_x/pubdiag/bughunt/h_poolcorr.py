"""Pool-level divergence from Cd on val vs public vs private (label-free): mean per-feed Pearson of the final
0.5*z(ft)+0.5*z(to) score, and of the ft / to halves. A test-side pipeline difference in the r6 members would show up
as M2/G3 diverging from Cd more on test than on val."""
import sys
import numpy as np
sys.path.insert(0, '<repo>/src')
from common import feed_meta
from stack_val import feed_rank_z
P = '<repo>/preds'; PUB = 1790497920
for sp in ('val', 'test'):
    f, M, off = feed_meta(sp)
    z = lambda x: feed_rank_z(off, M, np.asarray(x, np.float32))[:, 1]
    subs = {'val': np.ones(len(M), bool)} if sp == 'val' else {'public': f['timestamp'].to_numpy() < PUB, 'private': f['timestamp'].to_numpy() >= PUB}
    fin = {}
    for t in ('Cd', 'Pall', 'PF7x', 'G3', 'M2'):
        try:
            a, b = np.load(f'{P}/ens_{t}_ft_{sp}.npy'), np.load(f'{P}/ens_{t}_to_{sp}.npy')
        except FileNotFoundError:
            continue
        fin[t] = (z(0.5 * z(a) + 0.5 * z(b)), z(a), z(b))
    for t in fin:
        if t == 'Cd': continue
        for sn, m in subs.items():
            r = []
            for k in range(3):
                c = np.add.reduceat((fin[t][k] * fin['Cd'][k]).astype(np.float64), off) / M
                r.append(c[m].mean())
            print(f'{t:5s} vs Cd {sn:8s} final {r[0]:.5f} ft {r[1]:.5f} to {r[2]:.5f}')
