"""e3c: paired read of the non-fitted step structures against B, A and the expected FD16P20-type file (mean over the 495 draws of 4 of the 12 wide pop1 students), and the
candidate-building bar of the task (an extra file only if it beats B by more than 2 SE; these are fixed structures, so their nested value equals the in-sample value;
they are 6 of the 11 descriptive step rows, i.e. a scan).  Output out/pairs.json."""
from ens_common import *
import itertools, pmix
m = Meta(); st = Store(m); WTS = pmix.weights(); Z = lambda x: m.z('val', x)
def final(base, pop, w=0.2):
    br = [((1 - w) * Z(st.pool(k, base)) + w * Z(st.pool(k, pop))) if pop else st.pool(k, base) for k in ('ft_val', 'to_val')]; return m.ev(0.5 * Z(br[0]) + 0.5 * Z(br[1]))
A = final(W16 + FD16, P1d + P1w + FDP12 + FDPd + FDP2 + FDPF); B = final(FD16 + FDXHZ, FDP12 + FDPd + FDP2 + FDPF); FD = final(FD16, FDP4)
bft = Z(st.pool('ft_val', FD16)); bto = Z(st.pool('to_val', FD16)); XF = st.mat('ft_val', FDP12).astype(np.float64); XT = st.mat('to_val', FDP12).astype(np.float64); acc = np.zeros(len(m.Mv)); n = 0
for c in itertools.combinations(range(12), 4):
    c = list(c); acc += m.ev(0.5 * Z(0.8 * bft + 0.2 * Z(XF[c].mean(0))) + 0.5 * Z(0.8 * bto + 0.2 * Z(XT[c].mean(0)))); n += 1
EXP = acc / n
S = {'FD16 + FDP12 (28 all-data)': (FD16, FDP12), 'FD16 + all 28 all-data pop (44 all-data)': (FD16, FDP12 + FDPd + FDP2 + FDPF), 'FD16 + FDP12 + FDPF (36 all-data)': (FD16, FDP12 + FDPF),
     'W16 + FD16 + FDP4 (36)': (W16 + FD16, FDP4), 'W16 + FD16 + FDP12 (44)': (W16 + FD16, FDP12), 'W16 + FD16 + all 36 pop = A': (W16 + FD16, P1d + P1w + FDP12 + FDPd + FDP2 + FDPF), 'B': (FD16 + FDXHZ, FDP12 + FDPd + FDP2 + FDPF)}
res = {}
for nm, (b, p) in S.items():
    e = final(b, p); r = {}
    for rn, ref in (('B', B), ('A', A), ('FD16P20', FD), ('E[FD16P20-type]', EXP)):
        d = e - ref; pp, sp = pmix.wmean_se(d, WTS['public']); pr, sr = pmix.wmean_se(d, WTS['private']); r[rn] = dict(d=float(d.mean()), se=se(d), pub=float(pp), pub_se=float(sp), priv=float(pr), priv_se=float(sr))
    res[nm] = dict(val=float(e.mean()), vs=r)
    print(f'{nm:42s} val {e.mean():.5f} | vs B {r["B"]["d"]:+.5f}+-{r["B"]["se"]:.5f} (z {r["B"]["d"] / max(r["B"]["se"], 1e-9):+.2f}) | vs A {r["A"]["d"]:+.5f}+-{r["A"]["se"]:.5f} | vs FD16P20 {r["FD16P20"]["d"]:+.5f}+-{r["FD16P20"]["se"]:.5f} '
          f'| vs expected FD16P20-type {r["E[FD16P20-type]"]["d"]:+.5f}+-{r["E[FD16P20-type]"]["se"]:.5f} pmix {r["E[FD16P20-type]"]["pub"]:+.5f}+-{r["E[FD16P20-type]"]["pub_se"]:.5f} / {r["E[FD16P20-type]"]["priv"]:+.5f}+-{r["E[FD16P20-type]"]["priv_se"]:.5f}')
d = FD - EXP; pp, sp = pmix.wmean_se(d, WTS['public']); pr, sr = pmix.wmean_se(d, WTS['private']); print(f'realised FD16P20 vs expected FD16P20-type: {fmt(d)} pmix {pp:+.5f}+-{sp:.5f} / {pr:+.5f}+-{sr:.5f}')
res['_FD16P20_vs_expected'] = dict(d=float(d.mean()), se=se(d), pub=float(pp), priv=float(pr)); json.dump(res, open(OUT / 'pairs.json', 'w'), indent=0)
