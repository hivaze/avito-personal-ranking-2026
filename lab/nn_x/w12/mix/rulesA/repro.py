"""Step 1: reproduce build_r4.py's levels with the own loader (reference path = project functions) and check the numba kernel against it. Declared bar: DECLARATION.txt block 0."""
import time; t0 = time.time()
from lib import *
f, M, off = meta('val'); y = labels(); Mrow = np.repeat(M, M)
print('labels: unique', np.unique(y), 'rows', len(y), 'feeds', len(M), flush=True)
ev = lambda s: ndcg_per_feed(np.asarray(s, np.float32), y, off, M)
REC = {'C1': (0.40271, 0.40298, '+0.00026+-0.00021'), 'C2': (0.40240, 0.40290, '+0.00051+-0.00021'), 'C3': (0.40214, 0.40285, '+0.00071+-0.00021'), 'C4': (None, None, '+0.00043+-0.00019')}
ok = True; cache = {}
for c, (b, p) in COMPS.items():
    for m in MEMBERS[b] + MEMBERS[p]:
        if (m, 'val') not in cache: cache[(m, 'val')] = member_z(m, 'val')
    pools = pool(MEMBERS[b], 'val', cache) + pool(MEMBERS[p], 'val', cache); bF, bT, pF, pT = pools
    np.save(HERE / f'pools_{c}_val.npy', np.stack(pools).astype(np.float32))
    ep = ev(ref_final('val', bF, bT, pF, pT, 0.2, 0.2, 0.5)); wr = np.where(Mrow >= 50, 0.4, 0.2).astype(np.float32); em = ev(ref_final('val', bF, bT, pF, pT, wr, wr, 0.5))
    N = table(pools, y); np.save(HERE / f'N_{c}.npy', N)
    kp = N[:, PLAIN]; km = np.where(M >= 50, N[:, cidx(A04, A04, C05)], N[:, PLAIN])
    # two more spot rules through both paths
    e3 = ev(ref_final('val', bF, bT, pF, pT, 0.5, 0.1, 0.7)); k3 = N[:, cidx(5, 1, 4)]; e4 = ev(ref_final('val', bF, bT, pF, pT, 0.0, 0.6, 0.3)); k4 = N[:, cidx(0, 6, 0)]
    dk = max(abs(ep.mean() - kp.mean()), abs(em.mean() - km.mean()), abs(e3.mean() - k3.mean()), abs(e4.mean() - k4.mean())); dpf = max(np.abs(ep - kp).mean(), np.abs(em - km).mean(), np.abs(e3 - k3).mean(), np.abs(e4 - k4).mean())
    r = REC[c]; lv = (r[0] is None or round(ep.mean(), 5) == r[0]) and (r[1] is None or round(em.mean(), 5) == r[1]); dl = fmt(em - ep) == r[2]; kk = dk <= 1e-5 and dpf <= 1e-5
    ok &= lv and dl and kk
    print(f'{c} {b}+{p}: plain {ep.mean():.6f} (record {r[0]}) | M-rule {em.mean():.6f} (record {r[1]}) | rule-plain {fmt(em - ep)} (record {r[2]}) | levels {"OK" if lv else "MISMATCH"} delta {"OK" if dl else "MISMATCH"}'
          f' | kernel: plain {kp.mean():.6f} M-rule {km.mean():.6f} max|mean diff| {dk:.2e} mean|per-feed diff| {dpf:.2e} feeds differing {(np.abs(ep - kp) > 1e-9).sum()} {"OK" if kk else "FAIL"} | {time.time() - t0:.0f}s', flush=True)
print('STEP 1', 'PASS' if ok else 'FAIL')
