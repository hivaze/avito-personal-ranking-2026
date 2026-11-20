"""(2) member-equal pools of group unions: val / holdout (selected and @last), paired vs W8, slices, corr to W8 (val / hold / test),
diversity-adjusted expectation  E = mean member + k (1 - rho_bar)(1 - 1/N)  with k fitted on the homogeneous groups (per witness).
(5) C3 train-only pool added to the student pool. (3) wide/narrow weights. Writes out/unions.json."""
from pc_common import *
import time
m = Meta(); st = Store(m); G = groups()
ev = lambda s: ndcg_fast(s, m.yv, m.ov, m.Mv); eh = lambda s: ndcg_fast(s, m.yh, m.oh, m.Mh)
big = m.Mv >= 50
day = (m.fv['timestamp'].to_numpy() + 10800) // 86400; tue = day < 20720
names = json.load(open(OUT/'corr_names.json')); pos = {v: i for i, v in enumerate(names)}
Cv = np.load(OUT/'corr_val.npy'); Ch = np.load(OUT/'corr_hold.npy'); mem = json.load(open(OUT/'members.json'))['members']
U = lambda *gs: [v for g in gs for v in G[g]]
W8 = U('w192', 't25w192'); NEW = U('w192new', 't25w192new'); N8 = U('kd05', 'kd05t25'); C3 = U('F10', 'F7x', 'F7x70', 'pop1')
ref = {sp: st.pool(sp, W8) for sp in ('val', 'hold', 'test')}; e8v = ev(ref['val']); e8h = eh(ref['hold'])
W8L = [v + '@last' for v in W8]; refL = st.pool('hold', W8L); e8hL = eh(refL); e8vL = ev(st.pool('val', W8L))
t0 = time.time(); eh(ref['hold']); print(f'holdout NDCG eval {time.time()-t0:.2f}s')

# ---- law: k per witness from homogeneous groups (n >= 3) ----
def rho_bar(C, ms):
    ii = [pos[v] for v in ms]; return float(C[np.ix_(ii, ii)][np.triu_indices(len(ii), 1)].mean())
homo = ['kd05', 'kd08', 'kd10', 'kd03', 'kd05t25', 'kd08t25', 'w192', 't25w192', 'g2k05', 'g2k03', 'kx05', 'kx08', 'base', 'F10', 'F7x', 'F7x70', 'pop1', 'pop1kd']
X = {'val': [], 'hold': []}; Y = {'val': [], 'hold': []}
for g in homo:
    ms = G[g]; n = len(ms)
    for sp, C, e in (('val', Cv, ev), ('hold', Ch, eh)):
        gain = e(st.pool(sp, ms)).mean() - np.mean([mem[v][sp] for v in ms])
        X[sp].append((1 - rho_bar(C, ms)) * (1 - 1 / n)); Y[sp].append(gain)
K = {}
for sp in X:
    x = np.array(X[sp]); y = np.array(Y[sp]); K[sp] = float((x * y).sum() / (x * x).sum())
    r2 = 1 - ((y - K[sp] * x) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    print(f'law fit ({sp}): pool gain = k (1-rho)(1-1/N), k = {K[sp]:.4f}, R2 {r2:.3f}, resid sd {np.std(y-K[sp]*x):.5f} over {len(x)} homogeneous groups')
    for g, a, b in zip(homo, x, y): print(f'    {g:9s} x {a:.5f} gain {b:+.5f} pred {K[sp]*a:+.5f}')
def expect(sp, ms):
    C = Cv if sp == 'val' else Ch
    return float(np.mean([mem[v][sp] for v in ms]) + K[sp] * (1 - rho_bar(C, ms)) * (1 - 1 / len(ms)))
def vinf(sp, ms):
    C = Cv if sp == 'val' else Ch
    return float(np.mean([mem[v][sp] for v in ms]) + K[sp] * (1 - rho_bar(C, ms)))

res = {'k': K, 'pools': {}, 'vinf': {}}
print('\nASYMPTOTIC POOL VALUE per arm  V_inf = mean member + k (1 - rho_within)   (val | hold)')
for g in ['kd05', 'kd08', 'kd10', 'kd03', 'kd05t25', 'kd08t25', 'w192', 't25w192', 'g2k05', 'g2k03', 'kx05', 'kx08', 'kw_drop0', 'kw_e20', 'kw_hl2', 'kw_ff4', 'kw_id', 't25w256', 't25w192b3', 'pop1kd', 'base', 'F7x', 'F7x70', 'F10', 'pop1']:
    ms = G[g]; a = vinf('val', ms); b = vinf('hold', ms); res['vinf'][g] = dict(val=a, hold=b, n=len(ms), rho_val=rho_bar(Cv, ms), rho_hold=rho_bar(Ch, ms))
    print(f'  {g:10s} n={len(ms)} member {np.mean([mem[v]["val"] for v in ms]):.5f}/{np.mean([mem[v]["hold"] for v in ms]):.5f} rho {rho_bar(Cv, ms):.4f}/{rho_bar(Ch, ms):.4f} -> V_inf {a:.5f} | {b:.5f}')
a = vinf('val', W8); b = vinf('hold', W8); print(f'  W8 (both d192 arms) V_inf {a:.5f} | {b:.5f}'); res['vinf']['W8'] = dict(val=a, hold=b)

POOLS = {
    'W8': W8, 'w192 x4': U('w192'), 't25w192 x4': U('t25w192'),
    'W11 = W8 + 3 new wide': W8 + NEW,
    'N8 = kd05 + kd05t25': N8, 'S16 = W8 + N8': W8 + N8, 'S20 = S16 + kd08': W8 + N8 + U('kd08'),
    'W8 + kd05': W8 + U('kd05'), 'W8 + kd05t25': W8 + U('kd05t25'), 'W8 + kd08': W8 + U('kd08'), 'W8 + kd03': W8 + U('kd03'), 'W8 + kd10': W8 + U('kd10'),
    'W8 + g2k05': W8 + U('g2k05'), 'W8 + g2k03': W8 + U('g2k03'), 'W16g = W8 + g2k05 + g2k03': W8 + U('g2k05', 'g2k03'),
    'W8 + kx05': W8 + U('kx05'), 'W8 + kx08': W8 + U('kx08'), 'W8 + kx05 + kx08': W8 + U('kx05', 'kx08'),
    'W12 = W8 + g2k05': W8 + U('g2k05'),
    'W16x = W8 + g2k05 + kx05': W8 + U('g2k05', 'kx05'),
    'W20 = W8 + g2k05 + g2k03 + kx05': W8 + U('g2k05', 'g2k03', 'kx05'),
    'W24 = W8 + g2 x8 + kx x8': W8 + U('g2k05', 'g2k03', 'kx05', 'kx08'),
    'W23 = W8 + new3 + g2 x8 + kx05': W8 + NEW + U('g2k05', 'g2k03', 'kx05'),
    'W8 + kw(drop0,e20,hl2,ff4) x8': W8 + U('kw_drop0', 'kw_e20', 'kw_hl2', 'kw_ff4'),
    'W8 + kw_drop0 x2': W8 + U('kw_drop0'), 'W8 + kw_e20 x2': W8 + U('kw_e20'), 'W8 + kw_hl2 x2': W8 + U('kw_hl2'), 'W8 + kw_ff4 x2': W8 + U('kw_ff4'),
    'W8 + t25w256 x2': W8 + U('t25w256'), 'W8 + t25w192b3 x2': W8 + U('t25w192b3'), 'W8 + kd08w192 x1': W8 + U('kd08w192'),
    'ALLWIDE d192 (W8+new+g2+kx+kw4+b3+kd08w192)': W8 + NEW + U('g2k05', 'g2k03', 'kx05', 'kx08', 'kw_drop0', 'kw_e20', 'kw_hl2', 'kw_ff4', 't25w192b3', 'kd08w192'),
    'W8 + pop1kd': W8 + U('pop1kd'), 'W8 + base x6': W8 + U('base'), 'W8 + F7x': W8 + U('F7x'),
    'ALL STUDENTS tf (no kw_id, no pop1kd)': W8 + NEW + N8 + U('kd08', 'kd10', 'kd03', 'kd08t25', 'g2k05', 'g2k03', 'kx05', 'kx08', 'kw_drop0', 'kw_e20', 'kw_hl2', 'kw_ff4', 't25w192b3', 't25w256', 'kd08w192'),
    'C3 TO (28 hard-label)': C3, 'teacher tf11 = F7x + base4': U('F7x') + G['base'][:4],
    'W8 + C3 member-equal (36)': W8 + C3,
}
print(f'\n{"pool":46s}  N | val     hold    | d val vs W8          d hold vs W8        | @last d hold        | val M>=50          M<50              | expect v/h      | rho_bar v | corr W8 v/h/test')
for nm, ms in POOLS.items():
    pv = st.pool('val', ms); ph = st.pool('hold', ms); pt = st.pool('test', ms)
    a = ev(pv); b = eh(ph); dv = a - e8v; dh = b - e8h
    r = dict(n=len(ms), val=float(a.mean()), hold=float(b.mean()), d_val=float(dv.mean()), d_val_se=se(dv), d_hold=float(dh.mean()), d_hold_se=se(dh),
             d_val_big=float(dv[big].mean()), d_val_big_se=se(dv[big]), d_val_small=float(dv[~big].mean()), d_val_small_se=se(dv[~big]),
             d_val_tue=float(dv[tue].mean()), d_val_thu=float(dv[~tue].mean()),
             exp_val=expect('val', ms), exp_hold=expect('hold', ms), rho_val=rho_bar(Cv, ms), rho_hold=rho_bar(Ch, ms),
             corr_val=float(np.corrcoef(pv, ref['val'])[0, 1]), corr_hold=float(np.corrcoef(ph, ref['hold'])[0, 1]), corr_test=float(np.corrcoef(pt, ref['test'])[0, 1]))
    ls = ''
    if all(st.has(v + '@last') for v in ms):
        dL = eh(st.pool('hold', [v + '@last' for v in ms])) - e8hL; dLv = ev(st.pool('val', [v + '@last' for v in ms])) - e8vL
        r.update(d_hold_last=float(dL.mean()), d_hold_last_se=se(dL), d_val_last=float(dLv.mean()), d_val_last_se=se(dLv)); ls = fmt(dL)
    res['pools'][nm] = r
    print(f'{nm:46s} {len(ms):2d} | {a.mean():.5f} {b.mean():.5f} | {fmt(dv)} {fmt(dh)} | {ls:19s} | {fmt(dv[big])} {fmt(dv[~big])} | {r["exp_val"]:.5f} {r["exp_hold"]:.5f} | {r["rho_val"]:.4f} | {r["corr_val"]:.4f} {r["corr_hold"]:.4f} {r["corr_test"]:.4f}')
print(f'W8 @last: val {e8vL.mean():.5f} hold {e8hL.mean():.5f}; selected: val {e8v.mean():.5f} hold {e8h.mean():.5f}')

# ---- (3) weighting wide above narrow ----
print('\n(3) lambda z(wide pool) + (1-lambda) z(other pool): d vs W8 (val | hold | hold @last)')
res['weights'] = {}
z = lambda sp, x: m.z(sp, x)
def mixrow(tag, other, lams=(0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0), last_ok=True):
    ov_ = z('val', st.pool('val', other)); oh_ = z('hold', st.pool('hold', other)); wv = z('val', ref['val']); wh = z('hold', ref['hold'])
    if last_ok: ohL = z('hold', st.pool('hold', [v + '@last' for v in other])); whL = z('hold', refL)
    res['weights'][tag] = {}
    for lam in lams:
        dv = ev(lam * wv + (1 - lam) * ov_) - e8v; dh = eh(lam * wh + (1 - lam) * oh_) - e8h
        s = ''
        if last_ok: dL = eh(lam * whL + (1 - lam) * ohL) - e8hL; s = fmt(dL)
        res['weights'][tag][str(lam)] = dict(d_val=float(dv.mean()), d_val_se=se(dv), d_hold=float(dh.mean()), d_hold_se=se(dh), d_val_big=float(dv[big].mean()), d_val_small=float(dv[~big].mean()))
        print(f'  {tag:34s} lambda {lam:.1f}: val {fmt(dv)} (M>=50 {dv[big].mean():+.5f}, M<50 {dv[~big].mean():+.5f}) | hold {fmt(dh)} | {s}')
mixrow('W8 vs N8 (kd05+kd05t25)', N8)
mixrow('W8 vs N12 (kd05+kd05t25+kd08)', N8 + U('kd08'))
mixrow('W8 vs gen2 x8', U('g2k05', 'g2k03'))
mixrow('W8 vs kx05', U('kx05'))
mixrow('W8 vs kd03', U('kd03'))
mixrow('W8 vs base x6', U('base'))
# ---- (5) C3 train-only pool ----
c3v_file = np.load(P/'ens_Q2a8s3_to_val.npy').astype(np.float32); c3v = st.pool('val', C3)
print(f'\n(5) C3 TO pool: rebuilt-from-members vs preds/ens_Q2a8s3_to_val.npy corr {np.corrcoef(c3v, c3v_file)[0,1]:.6f}; val rebuilt {ev(c3v).mean():.5f} file {ev(c3v_file).mean():.5f}; holdout (28 members rebuilt) {eh(st.pool("hold", C3)).mean():.5f}')
mixrow('W8 vs C3 TO (z-mix)', C3, lams=(0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0), last_ok=False)
mixrow('W8 vs F7x+F7x70 (14)', U('F7x', 'F7x70'), lams=(0.5, 0.7, 0.8, 0.9, 1.0), last_ok=False)
json.dump(res, open(OUT/'unions.json', 'w'), indent=1)
