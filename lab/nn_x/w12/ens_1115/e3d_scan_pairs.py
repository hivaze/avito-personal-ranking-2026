"""e3d (derived from e3_forms.py: same pool / branch / final code): every scan form of e3 read PAIRED against its own structure (A or B). Output out/scan_pairs.json.

e3: the largest zero-training ensembles on FINAL val (0.5 z(FT) + 0.5 z(TO); per branch 0.8 z(base pool) + 0.2 z(popularity-light pool)) and on val reweighted to the
public / private mix (src/pmix.py).  References: FD16P20 and W8P20 (stored branch files preds/ens_<tag>_{ft,to}_val.npy).
PRE-DECLARED STRUCTURES (member lists written before any number was read; no fitted quantity):
  A  = base W16 + FD16 (32 wide students) | pop = every popularity-light STUDENT with a fine-tune (P1d 4, P1w 4, FDP12, FDPd 4, FDP2 4, FDPF 8 = 36) | w 0.2
  B  = base FD16 + FDXHZ16 (32, all trained on all data) | pop = FDP12 + FDPd 4 + FDP2 4 + FDPF 8 (28, all data) | w 0.2
SCANS (every form listed here is counted in the report):
  c1 / c2 : pop pool arm-equal / view-equal (pop1, pop2, popularity-free) instead of member-equal, on A and on B
  d1 / d2 / d3 : base' = 0.9 z(base) + 0.1 z(extra): extra = 12 d128 students | 20 hard-label members (F7x, F7x70, F10) | both (0.8 / 0.1 / 0.1), on A and on B
  f  : pooling operator inside each of the four sub-pools of B and A: rank-z mean | softmax-probability mean | trimmed mean 10% and 25% per side | median
Also: step table FD16P20 -> B (one arm at a time), leave-one-arm-out of A and B, the expected level of an FD16P20-type file over the 495 draws of 4 of the 12 wide pop1 students.
Output out/forms.json."""
from ens_common import *
import itertools, time
import pmix
m = Meta(); st = Store(m); WTS = pmix.weights(); t0 = time.time(); big = m.Mv >= 50
inv = json.load(open(OUT / 'inventory.json'))['members']
Z = lambda x: m.z('val', x)
def pool(kind, names, op='mean'):
    if op == 'mean': return st.pool(kind, names)
    X = st.mat(kind, names)
    if op == 'rank': return np.mean([m.rk('val', x) for x in X], 0)
    if op == 'median': return np.median(X, 0)
    if op.startswith('trim'):
        k = int(round(float(op[4:]) / 100 * len(names))); Xs = np.sort(X, 0); return Xs[k:len(names) - k].mean(0)
    if op == 'softmax':
        acc = np.zeros(m.n['val'], np.float64)
        for n_ in names:
            r = np.load(ROOT / inv[n_][kind]['path']).astype(np.float64); mx = np.maximum.reduceat(r, m.ov); e = np.exp(r - np.repeat(mx, m.Mv)); acc += e / np.repeat(np.add.reduceat(e, m.ov), m.Mv)
        return (acc / len(names)).astype(np.float32)
    raise ValueError(op)
def branch(kind, base, pop, w=0.2, op='mean', wb=None, wp=None, extra=()):
    """base / pop: member lists, or lists of member lists with weights wb / wp (arm-weighted pools). extra: [(members, u)] mixed into the base pool as (1 - sum u) z(base) + u z(extra)."""
    def P_(g, ws):
        if g and isinstance(g[0], (list, tuple)):
            ws = ws or [1.0 / len(g)] * len(g); return sum(w_ * Z(pool(kind, list(x), op)) for x, w_ in zip(g, ws))
        return pool(kind, g, op)
    b = P_(base, wb)
    if extra:
        u = sum(u_ for _, u_ in extra); b = (1 - u) * Z(b) + sum(u_ * Z(pool(kind, list(x), op)) for x, u_ in extra)
    if not pop: return b
    return (1 - w) * Z(b) + w * Z(P_(pop, wp))
def final(base, pop, **kw):
    ft = branch('ft_val', base, pop, **kw); to = branch('to_val', base, pop, **kw); return 0.5 * Z(ft) + 0.5 * Z(to), ft, to
ev = m.ev
stored = lambda tag: 0.5 * Z(np.load(P / f'ens_{tag}_ft_val.npy')) + 0.5 * Z(np.load(P / f'ens_{tag}_to_val.npy'))
REF = {'FD16P20': ev(stored('FD16P20kd')), 'W8P20': ev(stored('W8P20kd'))}
def line(e, ref):
    d = e - REF[ref]; pp, sp = pmix.wmean_se(d, WTS['public']); pr, sr = pmix.wmean_se(d, WTS['private'])
    return dict(d=float(d.mean()), se=se(d), pub=float(pp), pub_se=float(sp), priv=float(pr), priv_se=float(sr), short=float(d[~big].mean()), long=float(d[big].mean()))
RES = {}
def _unused_rec(name, sc, kind, note=''):
    f, ft, to = sc if isinstance(sc, tuple) else (sc, None, None)
    e = ev(f); r = dict(kind=kind, note=note, val=float(e.mean()), vs={k: line(e, k) for k in REF})
    if ft is not None: r['ft'] = float(ev(ft).mean()); r['to'] = float(ev(to).mean())
    RES[name] = r; a, b = r['vs']['FD16P20'], r['vs']['W8P20']
    print(f'{name:34s} [{kind}] val {r["val"]:.5f}' + (f' FT {r["ft"]:.5f} TO {r["to"]:.5f}' if ft is not None else '') +
          f' | vs FD16P20 {a["d"]:+.5f}+-{a["se"]:.5f} pmix {a["pub"]:+.5f}+-{a["pub_se"]:.5f} / {a["priv"]:+.5f}+-{a["priv_se"]:.5f} (M<50 {a["short"]:+.5f} M>=50 {a["long"]:+.5f})'
          f' | vs W8P20 {b["d"]:+.5f}+-{b["se"]:.5f} pmix {b["pub"]:+.5f} / {b["priv"]:+.5f}  [{time.time() - t0:.0f}s]', flush=True)
    json.dump(RES, open(OUT / 'forms.json', 'w'), indent=0)
    return e


A_base = W16 + FD16; A_pop = P1d + P1w + FDP12 + FDPd + FDP2 + FDPF; B_base = FD16 + FDXHZ; B_pop = FDP12 + FDPd + FDP2 + FDPF
A_parms = {'e10 pop1 d128': P1d, 'e10 pop1 wide': P1w, 'fd pop1 wide': FDP12, 'fd pop1 d128': FDPd, 'fd pop2 wide': FDP2, 'fd popfree wide': FDPF}; B_parms = {k: v for k, v in A_parms.items() if k.startswith('fd')}
HARD = F7x + F7x70 + F10; OUTP = {}
for tag, base, pop, parms in (('A', A_base, A_pop, A_parms), ('B', B_base, B_pop, B_parms)):
    e0 = ev(final(base, pop)[0])
    views = [[x for k, v in parms.items() if 'pop1' in k for x in v], [x for k, v in parms.items() if 'pop2' in k for x in v], [x for k, v in parms.items() if 'popfree' in k for x in v]]
    forms = {'c1 pop arm-equal': dict(base=base, pop=list(parms.values())), 'c2 pop view-equal': dict(base=base, pop=views), 'd1 + d128 x12 at 0.1': dict(base=base, pop=pop, extra=[(D128, 0.1)]),
             'd2 + hard x20 at 0.1': dict(base=base, pop=pop, extra=[(HARD, 0.1)]), 'd3 + d128 0.1 + hard 0.1': dict(base=base, pop=pop, extra=[(D128, 0.1), (HARD, 0.1)])}
    for op in ('rank', 'softmax', 'trim10', 'trim25', 'median'): forms[f'f {op}'] = dict(base=base, pop=pop, op=op)
    for nm, kw in forms.items():
        b_ = kw.pop('base'); p_ = kw.pop('pop'); d = ev(final(b_, p_, **kw)[0]) - e0; pp, sp = pmix.wmean_se(d, WTS['public']); pr, sr = pmix.wmean_se(d, WTS['private'])
        OUTP[f'{tag}.{nm}'] = dict(d=float(d.mean()), se=se(d), z=float(d.mean() / se(d)), pub=float(pp), pub_se=float(sp), priv=float(pr), priv_se=float(sr))
        print(f'{tag}.{nm:28s} minus {tag}: {fmt(d)} (z {d.mean() / se(d):+.2f}) | pmix {pp:+.5f}+-{sp:.5f} / {pr:+.5f}+-{sr:.5f}', flush=True)
json.dump(OUTP, open(OUT / 'scan_pairs.json', 'w'), indent=0)
