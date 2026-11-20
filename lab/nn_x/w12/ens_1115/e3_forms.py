"""e3: the largest zero-training ensembles on FINAL val (0.5 z(FT) + 0.5 z(TO); per branch 0.8 z(base pool) + 0.2 z(popularity-light pool)) and on val reweighted to the
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
def rec(name, sc, kind, note=''):
    f, ft, to = sc if isinstance(sc, tuple) else (sc, None, None)
    e = ev(f); r = dict(kind=kind, note=note, val=float(e.mean()), vs={k: line(e, k) for k in REF})
    if ft is not None: r['ft'] = float(ev(ft).mean()); r['to'] = float(ev(to).mean())
    RES[name] = r; a, b = r['vs']['FD16P20'], r['vs']['W8P20']
    print(f'{name:34s} [{kind}] val {r["val"]:.5f}' + (f' FT {r["ft"]:.5f} TO {r["to"]:.5f}' if ft is not None else '') +
          f' | vs FD16P20 {a["d"]:+.5f}+-{a["se"]:.5f} pmix {a["pub"]:+.5f}+-{a["pub_se"]:.5f} / {a["priv"]:+.5f}+-{a["priv_se"]:.5f} (M<50 {a["short"]:+.5f} M>=50 {a["long"]:+.5f})'
          f' | vs W8P20 {b["d"]:+.5f}+-{b["se"]:.5f} pmix {b["pub"]:+.5f} / {b["priv"]:+.5f}  [{time.time() - t0:.0f}s]', flush=True)
    json.dump(RES, open(OUT / 'forms.json', 'w'), indent=0)
    return e

# ---- references rebuilt from members (consistency with the stored branch files)
f, ft, to = final(FD16, FDP4); s = np.load(P / 'ens_FD16P20kd_ft_val.npy'); print('FD16P20 rebuilt FT branch max|d| vs stored', float(np.abs(ft - s).max()), 'final val', ev(f).mean(), 'stored', REF['FD16P20'].mean())
f, ft, to = final(W8, P1d); s = np.load(P / 'ens_W8P20kd_ft_val.npy'); print('W8P20 rebuilt FT branch max|d| vs stored', float(np.abs(ft - s).max()), 'final val', ev(f).mean(), 'stored', REF['W8P20'].mean())
rec('FD16P20 (rebuilt)', final(FD16, FDP4), 'reference'); rec('W8P20 (rebuilt)', final(W8, P1d), 'reference')
rec('FD16 (no pop)', final(FD16, []), 'reference'); rec('W16 (no pop)', final(W16, []), 'reference'); rec('FD16P20vf', final(FD16, FDP4 + FDPF[:4]), 'reference')

# ---- pre-declared structures
A_base = W16 + FD16; A_pop = P1d + P1w + FDP12 + FDPd + FDP2 + FDPF
B_base = FD16 + FDXHZ; B_pop = FDP12 + FDPd + FDP2 + FDPF
eA = rec('A: 32 wide + 36 pop students', final(A_base, A_pop), 'STRUCTURE'); eB = rec('B: all-data 32 + 28', final(B_base, B_pop), 'STRUCTURE')
d = eA - eB; print(f'A - B {fmt(d)}')
A_parms = {'e10 pop1 d128': P1d, 'e10 pop1 wide': P1w, 'fd pop1 wide': FDP12, 'fd pop1 d128': FDPd, 'fd pop2 wide': FDP2, 'fd popfree wide': FDPF}
B_parms = {k: v for k, v in A_parms.items() if k.startswith('fd')}
A_barms = {'e10 kd05w192': W16[:8], 'e10 kd05t25w192': W16[8:], 'fd kd05w192': FD16[:8], 'fd kd05t25w192': FD16[8:]}
B_barms = {'fd kd05w192': FD16[:8], 'fd kd05t25w192': FD16[8:], 'fdxhz kd05w192': FDXHZ[:8], 'fdxhz kd05t25w192': FDXHZ[8:]}

# ---- expected level of an FD16P20-type file (4 of the 12 full-data wide pop1 students): the realised file is one draw
bft = Z(st.pool('ft_val', FD16)); bto = Z(st.pool('to_val', FD16)); lv = []; acc = np.zeros(len(m.Mv))
XF = st.mat('ft_val', FDP12).astype(np.float64); XT = st.mat('to_val', FDP12).astype(np.float64)
for c in itertools.combinations(range(12), 4):
    c = list(c); e = ev(0.5 * Z(0.8 * bft + 0.2 * Z(XF[c].mean(0))) + 0.5 * Z(0.8 * bto + 0.2 * Z(XT[c].mean(0)))); lv.append(e.mean()); acc += e
lv = np.array(lv); eEXP = acc / len(lv); REF['E[FD16P20-type]'] = eEXP
print(f'FD16 + 4 of 12 wide pop1 (495 draws): mean {lv.mean():.5f} sd {lv.std():.5f} min {lv.min():.5f} max {lv.max():.5f}; realised FD16P20 {REF["FD16P20"].mean():.5f} = above {int((lv < REF["FD16P20"].mean()).sum())} of 495', flush=True)
RES['_draws4of12'] = dict(mean=float(lv.mean()), sd=float(lv.std()), min=float(lv.min()), max=float(lv.max()), realised_rank=int((lv < REF['FD16P20'].mean()).sum()))
for nm, e in (('A', eA), ('B', eB)):
    d = e - eEXP; pp, sp = pmix.wmean_se(d, WTS['public']); pr, sr = pmix.wmean_se(d, WTS['private'])
    print(f'{nm} vs expected FD16P20-type file: {fmt(d)} pmix {pp:+.5f}+-{sp:.5f} / {pr:+.5f}+-{sr:.5f}'); RES[f'_{nm}_vs_expected'] = dict(d=float(d.mean()), se=se(d), pub=float(pp), pub_se=float(sp), priv=float(pr), priv_se=float(sr))

# ---- step table FD16P20 -> B
steps = [('FD16 + FDP12', FD16, FDP12), ('FD16 + FDP12 + FDPd', FD16, FDP12 + FDPd), ('FD16 + FDP12 + FDPd + FDP2', FD16, FDP12 + FDPd + FDP2), ('FD16 + all 28 pop', FD16, B_pop),
         ('FD16 + FDXHZ, pop FDP4', B_base, FDP4), ('FDXHZ16 + FDP4 (hz replaces)', FDXHZ, FDP4), ('FD16 + FDXHZ, pop FDP12', B_base, FDP12), ('W16 + FD16, pop FDP4', A_base, FDP4),
         ('W16 + FD16, pop P1w + FDP4', A_base, P1w + FDP4), ('W16 + FD16 + FDXHZ + HZL8, pop A', A_base + FDXHZ + HZL8, A_pop), ('W16, pop P1d + P1w', W16, P1d + P1w)]
for nm, b_, p_ in steps: rec('step: ' + nm, final(b_, p_), 'step')

# ---- leave-one-arm-out
for tag, base, pop, barms, parms, eS in (('A', A_base, A_pop, A_barms, A_parms, eA), ('B', B_base, B_pop, B_barms, B_parms, eB)):
    lo = {}
    for an, ms in list(barms.items()) + list(parms.items()):
        b_ = [x for x in base if x not in ms]; p_ = [x for x in pop if x not in ms]; e = ev(final(b_, p_)[0]); d = e - eS; lo[an] = (float(d.mean()), se(d))
        print(f'LOAO {tag} without {an:20s}: {fmt(d)}', flush=True)
    RES[f'_loao_{tag}'] = lo

# ---- scans c, d
for tag, base, pop, parms in (('A', A_base, A_pop, A_parms), ('B', B_base, B_pop, B_parms)):
    rec(f'{tag}.c1 pop arm-equal', final(base, list(parms.values())), 'scan')
    views = [[x for k, v in parms.items() if 'pop1' in k for x in v], [x for k, v in parms.items() if 'pop2' in k for x in v], [x for k, v in parms.items() if 'popfree' in k for x in v]]
    rec(f'{tag}.c2 pop view-equal', final(base, views), 'scan')
    HARD = F7x + F7x70 + F10
    rec(f'{tag}.d1 + d128 x12 at 0.1', final(base, pop, extra=[(D128, 0.1)]), 'scan'); rec(f'{tag}.d2 + hard x20 at 0.1', final(base, pop, extra=[(HARD, 0.1)]), 'scan')
    rec(f'{tag}.d3 + d128 0.1 + hard 0.1', final(base, pop, extra=[(D128, 0.1), (HARD, 0.1)]), 'scan')
# ---- scan f: pooling operators
for tag, base, pop in (('B', B_base, B_pop), ('A', A_base, A_pop)):
    for op in ('rank', 'softmax', 'trim10', 'trim25', 'median'):
        rec(f'{tag}.f {op}', final(base, pop, op=op), 'scan')
print(f'done {time.time() - t0:.0f}s')
