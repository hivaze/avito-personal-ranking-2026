"""ADDENDUM 6: the declared nested search on the 8-member popularity-light pools (4 wide + 4 d128)."""
import pickle, time; t0 = time.time()
from lib import *
f, M, off = meta('val'); y = labels(); pr = props('val'); gt = gates(pr); parts = partitions(gt); fams = families(); FL = [folds(pr['uid'], s) for s in SEEDS]; names = [x[0] for x in fams]; iM = names.index('C(M50)')
C8 = {'C5': ('FD16', MEMBERS['FDP'] + MEMBERS['FDPd']), 'C6': ('FD8', MEMBERS['FDP'] + MEMBERS['FDPd']), 'C7': ('W16', MEMBERS['P1w'] + MEMBERS['P1d']), 'C8': ('W8', MEMBERS['P1w'] + MEMBERS['P1d'])}
cache = {}; SHOW = ['C(M50)', 'C(Mq)', 'G(M3q)', 'G(M3r)', 'K(Mq,Hq)', 'F(Hq)', 'J(CS45,Hq)', 'A', 'B', 'I(M50)', 'D(M50)', 'E(CS45)']
for c, (b, pop) in C8.items():
    for m in MEMBERS[b] + pop:
        if (m, 'val') not in cache: cache[(m, 'val')] = member_z(m, 'val')
    pools = pool(MEMBERS[b], 'val', cache) + pool(pop, 'val', cache); N = table(pools, y); np.save(HERE / f'N_{c}.npy', N); r = search(N, parts, fams, FL, want_vec=True)
    plain = N[:, PLAIN]; mr = np.where(M >= 50, N[:, cidx(A04, A04, C05)], plain); r['plain_vec'] = plain; r['mrule_vec'] = mr; pickle.dump(r, open(HERE / f'res_{c}.pkl', 'wb'))
    el = np.array([x[5] for x in fams]); top = names[int(np.argmax(np.where(el, r['nested'], -9)))]
    print(f'\n=== {c} {b} + 4 wide + 4 d128: plain {plain.mean():.5f} | fixed M-rule - plain {fmt(mr - plain)} | top nested row {top} | {time.time() - t0:.0f}s')
    print(f'{"family":14s} | in-sample | nested - plain (z)          | nested - fixed MRULE | best rule on all val')
    for k in SHOW + ([top] if top not in SHOW else []):
        fi = names.index(k); d = r['vec'][fi] - plain; m_, s_ = se(d)
        print(f'{k:14s} | {r["ins"][fi] - plain.mean():+.5f}  | {m_:+.5f}+-{s_:.5f} ({m_ / s_:+.1f}) | {fmt(r["vec"][fi] - mr)}   | ' + ' ; '.join(str(decode(x)) for x in r['ins_rule'][fi]))
    for v in ('B-all', 'B-3', 'G'): print(f'VARIANT {v}: nested - plain {fmt(r["var"][v]["vec"] - plain)} | - fixed MRULE {fmt(r["var"][v]["vec"] - mr)}')
