"""Descriptive dose table (ADDENDUM 3): constant a at c = .5 by M cell minus a = .2; pool-only levels. Fixed rules, in-sample, no selection."""
from lib import *
f, M, off = meta('val'); y = labels(); n = len(M); ev = lambda s: ndcg_per_feed(np.asarray(s, np.float32), y, off, M)
cells = {'all': np.ones(n, bool), 'M<30': M < 30, 'M30-49': (M >= 30) & (M < 50), 'M50-99': (M >= 50) & (M < 100), 'M>=100': M >= 100, 'M<50': M < 50, 'M>=50': M >= 50}
print('cell sizes: ' + ', '.join(f'{k} {v.sum()}' for k, v in cells.items()))
for c in COMPS:
    N = np.load(HERE / f'N_{c}.npy'); pools = np.load(HERE / f'pools_{c}_val.npy'); bF, bT, pF, pT = pools
    eb = ev(ref_final('val', bF, bT, pF, pT, 0.0, 0.0, 0.5)); ep = ev(ref_final('val', bF, bT, pF, pT, 1.0, 1.0, 0.5))
    print(f'\n=== {c} {COMPS[c]}: pool-only finals (0.5 FT + 0.5 TO): popularity-light pool minus base pool by cell: ' + ' | '.join(f'{k} {fmt((ep - eb)[v])}' for k, v in cells.items()))
    print(f'   const c at a = .2 (minus c = .5), all feeds: ' + ' | '.join(f'c {CG[ic]:.1f} {fmt(N[:, cidx(A02, A02, ic)] - N[:, PLAIN])}' for ic in range(NC)))
    print(f'{"a (c = .5) minus a = .2":24s} | ' + ' | '.join(f'{k:>18s}' for k in cells))
    for ia in range(NA):
        d = N[:, cidx(ia, ia, C05)] - N[:, PLAIN]; print(f'a = {AG[ia]:.1f}                  | ' + ' | '.join(fmt(d[v]) for v in cells.values()))
