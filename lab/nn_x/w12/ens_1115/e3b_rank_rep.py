"""e3b: replication read of the one scan form that read above its structure in e3 (rank-z pooling inside every sub-pool).
RULE written 2026-11-15T21:2xZ BEFORE this script ran (out/e3b_rule.txt): the operator is parameter-free, so the test is replication on DISJOINT member sets:
  S1 = held-out members only (W16 + P1d + P1w), S2 = all-data members only (FD16 + FDP12), S3 = all-data hz re-cools + popularity-free (FDXHZ + FDPF), S4 = hard-label C3 members (F7x + F7x70 + F10 | pop1 x8).
  REPLICATED iff the paired final-val delta (rank pooling - mean-z pooling, same members) is > 0 on S1 AND S2 and the inverse-variance pooled z over S1..S4 is >= 2;
  also reported: the delta per branch, pmix, the holdout read on S1 (train-only pool; reported only) and which stage carries it (rank only inside pools, with the usual z blends above).
A REPLICATED read is a candidate for a paired public test, not an adopted change (closed row 'rank / RRF fusion' in COVERAGE_MAP: the difference is the fine-tuned branch and the students)."""
from ens_common import *
import pmix
m = Meta(); st = Store(m); WTS = pmix.weights(); Z = lambda x: m.z('val', x)
def pool(kind, names, op):
    if op == 'mean': return st.pool(kind, names)
    sp = SPLIT[kind]; return np.mean([m.rk(sp, st.get(kind, n_)) for n_ in names], 0)
def final(base, pop, op, ops=None):
    ops = ops or dict(ft=op, to=op); br = []
    for k, o in (('ft_val', ops['ft']), ('to_val', ops['to'])):
        b = pool(k, base, o); br.append(0.8 * Z(b) + 0.2 * Z(pool(k, pop, o)) if pop else b)
    return 0.5 * Z(br[0]) + 0.5 * Z(br[1]), br[0], br[1]
SETS = {'S1 held-out W16 + P1d + P1w': (W16, P1d + P1w), 'S2 all-data FD16 + FDP12': (FD16, FDP12), 'S3 all-data FDXHZ + FDPF': (FDXHZ, FDPF), 'S4 hard-label F7x + F7x70 + F10 | pop1 x8': (F7x + F7x70 + F10, POP1H),
        'ref FD16P20 (FD16 + FDP4)': (FD16, FDP4), 'ref W8P20 (W8 + P1d)': (W8, P1d), 'A': (W16 + FD16, P1d + P1w + FDP12 + FDPd + FDP2 + FDPF), 'B': (FD16 + FDXHZ, FDP12 + FDPd + FDP2 + FDPF)}
res = {}; zz = []
for nm, (b, p) in SETS.items():
    fm, ftm, tom = final(b, p, 'mean'); fr, ftr, tor = final(b, p, 'rank'); em = m.ev(fm); d = m.ev(fr) - em; dft = m.ev(ftr) - m.ev(ftm); dto = m.ev(tor) - m.ev(tom)
    dfo = m.ev(final(b, p, None, dict(ft='rank', to='mean'))[0]) - em; dto_only = m.ev(final(b, p, None, dict(ft='mean', to='rank'))[0]) - em
    pp, sp = pmix.wmean_se(d, WTS['public']); pr, sr = pmix.wmean_se(d, WTS['private'])
    r = dict(mean=float(em.mean()), d=float(d.mean()), se=se(d), ft=float(dft.mean()), ft_se=se(dft), to=float(dto.mean()), to_se=se(dto), pub=float(pp), pub_se=float(sp), priv=float(pr), priv_se=float(sr),
             rank_ft_only=(float(dfo.mean()), se(dfo)), rank_to_only=(float(dto_only.mean()), se(dto_only)))
    if all(st.has('hold', x) for x in b + p):
        hm = 0.8 * m.z('hold', pool('hold', b, 'mean')) + 0.2 * m.z('hold', pool('hold', p, 'mean')); hr = 0.8 * m.z('hold', pool('hold', b, 'rank')) + 0.2 * m.z('hold', pool('hold', p, 'rank'))
        dh = m.eh(hr) - m.eh(hm); r['hold'] = (float(dh.mean()), se(dh))
    res[nm] = r
    if nm.startswith('S'): zz.append((r['d'], r['se']))
    print(f'{nm:44s} mean-z {r["mean"]:.5f} | rank - mean: final {r["d"]:+.5f}+-{r["se"]:.5f} (z {r["d"] / r["se"]:+.2f}) FT branch {r["ft"]:+.5f}+-{r["ft_se"]:.5f} TO branch {r["to"]:+.5f}+-{r["to_se"]:.5f} | pmix {r["pub"]:+.5f}+-{r["pub_se"]:.5f} / {r["priv"]:+.5f}+-{r["priv_se"]:.5f}'
          f' | rank in FT only {dfo.mean():+.5f}+-{se(dfo):.5f} in TO only {dto_only.mean():+.5f}+-{se(dto_only):.5f}' + (f' | holdout TO pool {r["hold"][0]:+.5f}+-{r["hold"][1]:.5f}' if 'hold' in r else ''), flush=True)
w = np.array([1 / s ** 2 for _, s in zz]); dd = np.array([d for d, _ in zz]); pz = float((w * dd).sum() / np.sqrt(w.sum())); res['_pooled'] = dict(mean=float((w * dd).sum() / w.sum()), z=pz, note='feeds are shared between sets, so the pooled z overstates independence; member sets are disjoint')
ok = zz[0][0] > 0 and zz[1][0] > 0 and pz >= 2
print(f'pooled over S1..S4: {res["_pooled"]["mean"]:+.5f} z {pz:+.2f} (same feeds in every set: z is an upper bound) -> {"REPLICATED" if ok else "NOT REPLICATED"}'); res['_verdict'] = 'REPLICATED' if ok else 'NOT REPLICATED'
json.dump(res, open(OUT / 'rank_rep.json', 'w'), indent=0)
