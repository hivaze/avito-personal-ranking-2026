"""ADDENDUM 4: descriptive constant-w_ft curve, unweighted and under pmix weights."""
import sys; sys.path.insert(0, '<repo>/src')
from lib import *
import pmix
f, M, off = meta('val'); y = labels(); n = len(M); pr = props('val'); W = pmix.weights(); ev = lambda s: ndcg_per_feed(np.asarray(s, np.float32), y, off, M); big = M >= 50; cold = pr['CS'] >= .45; ar = np.arange(n); Mrow = np.repeat(M, M)
for c in COMPS:
    N = np.load(HERE / f'N_{c}.npy'); bF, bT, pF, pT = np.load(HERE / f'pools_{c}_val.npy')
    for tag, aidx, arow in (('a = .2', np.full(n, A02), 0.2), ('M-rule weights', np.where(big, A04, A02), np.where(Mrow >= 50, 0.4, 0.2).astype(np.float32))):
        base = N[ar, cidx(aidx, aidx, C05)]; curve = {0.0: ev(ref_final('val', bF, bT, pF, pT, arow, arow, 0.0))}
        for ic in range(NC): curve[float(CG[ic])] = N[ar, cidx(aidx, aidx, ic)]
        curve[1.0] = ev(ref_final('val', bF, bT, pF, pT, arow, arow, 1.0))
        print(f'\n=== {c} {COMPS[c]} | {tag} | level at c = .5: val {base.mean():.5f} pmix-public {np.average(base, weights=W["public"]):.5f} pmix-private {np.average(base, weights=W["private"]):.5f}')
        print(f'{"c":4s} | val                 | pmix-public         | pmix-private        | cold cell CS>=.45 (n {cold.sum()}) | rest')
        for cv, v in curve.items():
            d = v - base; print(f'{cv:.1f}  | {fmt(d)} | {fmt(d, W["public"])} | {fmt(d, W["private"])} | {fmt(d[cold])}      | {fmt(d[~cold])}')
