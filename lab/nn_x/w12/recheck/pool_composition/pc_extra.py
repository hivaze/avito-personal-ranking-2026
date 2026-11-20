"""Extra reads:
 a. asymptote estimator A(S) = (n * NDCG(pool S) - mean member NDCG) / (n - 1) per feed (exact under NDCG(n) = a - b/n), paired between arms.
 b. pooled member-level FT - TO delta: wide students vs hard-label members (val).
 c. pooling form: mean of per-feed z (used) vs mean raw logit vs mean within-feed rank, W8, val / holdout.
 d. the three KD-era public files: final val, TO-branch holdout, test corr, public.
 e. pair-level spread of the within-arm correlation (how well 1 pair estimates rho)."""
from pc_common import *
m = Meta(); st = Store(m); G = groups()
ev = lambda s: ndcg_fast(s, m.yv, m.ov, m.Mv); eh = lambda s: ndcg_fast(s, m.yh, m.oh, m.Mh)
zv = lambda x: m.z('val', x); big = m.Mv >= 50
U = lambda *gs: [v for g in gs for v in G[g]]
W8 = U('w192', 't25w192'); N8 = U('kd05', 'kd05t25'); C3 = U('F10', 'F7x', 'F7x70', 'pop1')
res = {}
def A(sp, ms):
    e = ev if sp == 'val' else eh; n = len(ms)
    return (n * e(st.pool(sp, ms)) - np.mean([e(st.get(sp, v)) for v in ms], 0)) / (n - 1)
print('== a. asymptote estimator A(S), paired contrasts (val | hold)')
S4 = lambda arm: [f'w5_e10_{arm}_s{s}' for s in (951, 952, 953, 954)]
sets = {'base x4': S4('base'), 'base x6': G['base'], 'kd05 x4': S4('kd05'), 'kd05t25 x4': S4('kd05t25'), 'kd03 x4': S4('kd03'), 'kd08 x4': S4('kd08'), 'kd10 x4': S4('kd10'), 'N8': N8,
        'w192 x4': S4('kd05w192'), 't25w192 x4': S4('kd05t25w192'), 'W8': W8, 'g2k05 x4': S4('kd05g2w192'), 'g2k03 x4': S4('kd03g2w192'), 'gen2 x8': U('g2k05', 'g2k03'),
        'kx05 x4': S4('kx05'), 'kx08 x4': S4('kx08'), 'F7x x7': G['F7x'], 'F7x70 x7': G['F7x70'], 'F10 x6': G['F10'], 'pop1 x8': G['pop1'], 'pop1kd x4': G['pop1kd'], 'tf11 teacher': U('F7x') + S4('base'),
        'W11': W8 + U('w192new', 't25w192new')}
AV = {k: A('val', v) for k, v in sets.items()}; AH = {k: A('hold', v) for k, v in sets.items()}
for k in sets: print(f'  A({k:13s}) n={len(sets[k]):2d}: val {AV[k].mean():.5f} hold {AH[k].mean():.5f} | pool {ev(st.pool("val", sets[k])).mean():.5f} {eh(st.pool("hold", sets[k])).mean():.5f}')
res['A'] = {k: dict(val=float(AV[k].mean()), hold=float(AH[k].mean())) for k in sets}
res['A_contrasts'] = {}
for a, b in (('kd05 x4', 'base x4'), ('kd05t25 x4', 'base x4'), ('N8', 'base x6'), ('N8', 'tf11 teacher'), ('w192 x4', 'kd05 x4'), ('t25w192 x4', 'kd05t25 x4'), ('W8', 'N8'), ('W8', 'base x6'), ('W8', 'F7x x7'),
             ('gen2 x8', 'W8'), ('g2k05 x4', 'w192 x4'), ('kx05 x4', 'w192 x4'), ('kd03 x4', 'kd05 x4'), ('kd08 x4', 'kd05 x4'), ('kd10 x4', 'kd05 x4'), ('t25w192 x4', 'w192 x4'), ('kd05t25 x4', 'kd05 x4'), ('W11', 'W8')):
    dv = AV[a] - AV[b]; dh = AH[a] - AH[b]; res['A_contrasts'][f'{a} - {b}'] = dict(val=float(dv.mean()), val_se=se(dv), hold=float(dh.mean()), hold_se=se(dh))
    print(f'  A({a}) - A({b}): val {fmt(dv)} | hold {fmt(dh)}')

print('\n== b. member-level FT - TO (val), pooled')
ftv = lambda v: ev(zv(np.load(P/f'nn_{v}_n2cpv3s3_val_oof.npy')))
dW = np.mean([ftv(v) - ev(st.get('val', v)) for v in W8], 0); dN = np.mean([ftv(v) - ev(st.get('val', v)) for v in N8], 0); dC = np.mean([ftv(v) - ev(st.get('val', v)) for v in C3], 0)
for nm, d in (('8 wide students', dW), ('8 narrow students (kd05, kd05t25)', dN), ('28 hard-label C3 members', dC), ('wide - hard', dW - dC), ('narrow - hard', dN - dC), ('all 16 students - hard', (dW + dN) / 2 - dC)):
    print(f'  {nm:34s}: {fmt(d)} | M>=50 {d[big].mean():+.5f}+-{se(d[big]):.5f} M<50 {d[~big].mean():+.5f}+-{se(d[~big]):.5f}')
    res[f'b {nm}'] = dict(d=float(d.mean()), se=se(d), d_big=float(d[big].mean()), d_big_se=se(d[big]), d_small=float(d[~big].mean()), d_small_se=se(d[~big]))

print('\n== c. pooling form for W8 (TO): z-mean (used) vs raw-logit mean vs rank mean')
import numba as nb
from stack_val import feed_rank_z
for sp, e, MM, off in (('val', ev, m.Mv, m.ov), ('hold', eh, m.Mh, m.oh)):
    raw = [np.load(paths(v)[0 if sp == 'val' else 1]).astype(np.float32) for v in W8]
    zm = e(st.pool(sp, W8)); rm = e(np.mean(raw, 0)); rk = e(np.mean([feed_rank_z(off, MM, r)[:, 0] for r in raw], 0))
    s = ''
    if sp == 'val': s = f' | raw-z M<50 {(rm-zm)[~big].mean():+.5f}+-{se((rm-zm)[~big]):.5f}'
    print(f'  {sp}: z-mean {zm.mean():.5f} | raw-logit mean {fmt(rm-zm)} | rank mean {fmt(rk-zm)}{s}')
    res[f'c {sp}'] = dict(raw=float((rm - zm).mean()), raw_se=se(rm - zm), rank=float((rk - zm).mean()), rank_se=se(rk - zm))

print('\n== d. public files: val final / TO holdout / public')
pub = {'C3': 0.393454, 'W8': 0.394486, 'S16': 0.394078}
hold = {'C3': eh(st.pool('hold', C3)), 'W8': eh(st.pool('hold', W8)), 'S16': eh(st.pool('hold', W8 + N8))}
def final_val(tag):
    f = np.load(P/f'ens_{tag}_ft_val.npy'); t = np.load(P/f'ens_{tag}_to_val.npy'); return ev(0.5 * zv(f) + 0.5 * zv(t)), ev(t)
fv = {'C3': final_val('Q2a8s3'), 'W8': final_val('W8kd'), 'S16': final_val('S16kd')}
for k in ('C3', 'S16', 'W8'):
    dv = fv[k][0] - fv['W8'][0]; dt = fv[k][1] - fv['W8'][1]; dh = hold[k] - hold['W8']
    print(f'  {k:4s} final val {fv[k][0].mean():.5f} ({fmt(dv)}; M>=50 {dv[big].mean():+.5f} M<50 {dv[~big].mean():+.5f}) | TO val {fmt(dt)} (M>=50 {dt[big].mean():+.5f}+-{se(dt[big]):.5f}) | TO holdout {hold[k].mean():.5f} ({fmt(dh)}) | public {pub[k]:.6f} ({pub[k]-pub["W8"]:+.6f})')
    res[f'd {k}'] = dict(val=float(fv[k][0].mean()), d_val=float(dv.mean()), d_val_se=se(dv), hold=float(hold[k].mean()), d_hold=float(dh.mean()), d_hold_se=se(dh), public=pub[k], d_public=pub[k] - pub['W8'])
g3 = U('F10', 'F7x', 'F7x70'); dh = eh(st.pool('hold', g3)) - hold['W8']; print(f'  G3 TO holdout {eh(st.pool("hold", g3)).mean():.5f} ({fmt(dh)} vs W8)')

print('\n== e. pair-level spread of within-arm member correlation (val)')
names = json.load(open(OUT/'corr_names.json')); pos = {v: i for i, v in enumerate(names)}; Cv = np.load(OUT/'corr_val.npy')
for g in ('w192', 't25w192', 'kd05', 'kd05t25', 'g2k05', 'g2k03', 'kx05', 'base'):
    ii = [pos[v] for v in G[g]]; x = Cv[np.ix_(ii, ii)][np.triu_indices(len(ii), 1)]
    print(f'  {g:8s} pairs {len(x)}: mean {x.mean():.4f} sd {x.std():.4f} min {x.min():.4f} max {x.max():.4f}')
json.dump(res, open(OUT/'extra.json', 'w'), indent=1)
