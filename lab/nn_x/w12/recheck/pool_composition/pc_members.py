"""Member table, group table, correlation structure (val + holdout), selected and @last checkpoints."""
from pc_common import *
m = Meta(); st = Store(m); G = groups()
ev = lambda s: ndcg_fast(s, m.yv, m.ov, m.Mv); eh = lambda s: ndcg_fast(s, m.yh, m.oh, m.Mh)
big = m.Mv >= 50
res = {'members': {}, 'groups': {}}
allm = [v for g in G.values() for v in g]
for v in allm:
    r = dict(val=float(ev(st.get('val', v)).mean()), hold=float(eh(st.get('hold', v)).mean()))
    if st.has(v + '@last'):
        r['val_last'] = float(ev(st.get('val', v + '@last')).mean()); r['hold_last'] = float(eh(st.get('hold', v + '@last')).mean())
    jf = ROOT/f'nn_x/w5/logs/{v}.json'
    if jf.exists():
        j = json.load(open(jf)); r['best_step'] = j.get('best_step'); h = j.get('history', [])
        r['n_evals'] = len(h); r['last_step'] = h[-1]['step'] if h else None
    res['members'][v] = r


def corrmat(sp, names):
    n = m.n[sp]; k = len(names); C = np.zeros((k, k)); ch = 400_000
    for a in range(0, n, ch):
        X = np.stack([np.asarray(st.get(sp, v)[a:a + ch], np.float64) for v in names]); C += X @ X.T
    d = np.sqrt(np.diag(C)); return C / d[:, None] / d[None, :]


Cv = corrmat('val', allm); Ch = corrmat('hold', allm); pos = {v: i for i, v in enumerate(allm)}
np.save(OUT/'corr_val.npy', Cv); np.save(OUT/'corr_hold.npy', Ch); json.dump(allm, open(OUT/'corr_names.json', 'w'))
w8 = G['w192'] + G['t25w192']; w8v = st.pool('val', w8); w8h = st.pool('hold', w8); e8v = ev(w8v); e8h = eh(w8h)
print(f'W8 TO pool: val {e8v.mean():.5f} hold {e8h.mean():.5f}  (val M>=50 {e8v[big].mean():.5f}, M<50 {e8v[~big].mean():.5f}; n M<50 {int((~big).sum())})')
print(f'{"group":11s} n | member val   hold    | @last val  hold   | pool val    hold    | pool-W8 val            hold                 | rho_in v/h    | corr pool~W8 v/h')
for g, ms in G.items():
    ii = [pos[v] for v in ms]; iu = np.triu_indices(len(ms), 1)
    mv = np.mean([res['members'][v]['val'] for v in ms]); mh = np.mean([res['members'][v]['hold'] for v in ms])
    lv = np.mean([res['members'][v].get('val_last', np.nan) for v in ms]); lh = np.mean([res['members'][v].get('hold_last', np.nan) for v in ms])
    pv = st.pool('val', ms); ph = st.pool('hold', ms); dv = ev(pv) - e8v; dh = eh(ph) - e8h
    rv = Cv[np.ix_(ii, ii)][iu].mean() if len(ms) > 1 else np.nan; rh = Ch[np.ix_(ii, ii)][iu].mean() if len(ms) > 1 else np.nan
    cv = np.corrcoef(pv, w8v)[0, 1]; chh = np.corrcoef(ph, w8h)[0, 1]
    res['groups'][g] = dict(n=len(ms), member_val=mv, member_hold=mh, last_val=lv, last_hold=lh, pool_val=float(ev(pv).mean()), pool_hold=float(eh(ph).mean()),
                            d_val=float(dv.mean()), d_val_se=se(dv), d_hold=float(dh.mean()), d_hold_se=se(dh), rho_val=float(rv), rho_hold=float(rh), corr_w8_val=float(cv), corr_w8_hold=float(chh))
    print(f'{g:11s} {len(ms)} | {mv:.5f} {mh:.5f} | {lv:.5f} {lh:.5f} | {ev(pv).mean():.5f} {eh(ph).mean():.5f} | {fmt(dv)} {fmt(dh)} | {rv:.4f} {rh:.4f} | {cv:.4f} {chh:.4f}')
# group-group mean cross correlation (val / holdout), student groups + base
key = ['kd05', 'kd08', 'kd10', 'kd03', 'kd05t25', 'kd08t25', 'w192', 't25w192', 'g2k05', 'g2k03', 'kx05', 'kx08', 'pop1kd', 'base', 'F7x', 'F10', 'pop1']
def cross(C, a, b):
    ia = [pos[v] for v in G[a]]; ib = [pos[v] for v in G[b]]; X = C[np.ix_(ia, ib)]
    if a == b: return X[np.triu_indices(len(ia), 1)].mean()
    return X.mean()
for nm, C in (('val', Cv), ('hold', Ch)):
    print(f'\nmean member-member corr between groups ({nm}); diagonal = within group')
    print(' ' * 9 + ' '.join(f'{k[:7]:>7s}' for k in key))
    for a in key: print(f'{a:9s}' + ' '.join(f'{cross(C, a, b):7.4f}' for b in key))
    res[f'cross_{nm}'] = {a: {b: float(cross(C, a, b)) for b in key} for a in key}
# same-seed cross-arm vs different-seed (is the teacher / width an extra diversity source beyond the seed?)
print('\nsame-seed vs different-seed member corr across arm pairs (val | hold)')
def seedpair(C, a, b):
    same, diff = [], []
    for va in G[a]:
        for vb in G[b]:
            (same if va.split('_s')[-1] == vb.split('_s')[-1] else diff).append(C[pos[va], pos[vb]])
    return np.mean(same), np.mean(diff)
for a, b in (('w192', 't25w192'), ('kd05', 'kd05t25'), ('kd05', 'w192'), ('kd05t25', 't25w192'), ('kd05', 'kd08'), ('w192', 'g2k05'), ('w192', 'kx05'), ('base', 'kd05'), ('base', 'w192'), ('kd05t25', 'pop1kd')):
    sv, dv_ = seedpair(Cv, a, b); sh, dh_ = seedpair(Ch, a, b)
    print(f' {a:8s} x {b:8s}: val same-seed {sv:.4f} diff-seed {dv_:.4f} | hold same {sh:.4f} diff {dh_:.4f} | within {a} {cross(Cv,a,a):.4f}/{cross(Ch,a,a):.4f} within {b} {cross(Cv,b,b):.4f}/{cross(Ch,b,b):.4f}')
json.dump(res, open(OUT/'members.json', 'w'), indent=1)
# checkpoint selection: how many members have best_step == last step
sel = [(v, r.get('best_step'), r.get('last_step')) for v, r in res['members'].items() if r.get('best_step') is not None]
same = sum(1 for v, b, l in sel if b == l); print(f'\nselected == last checkpoint: {same}/{len(sel)} members')
for g in ('w192', 't25w192', 'kd05', 'kd05t25', 'base'):
    print(' ', g, [(res['members'][v].get('best_step'), res['members'][v].get('last_step')) for v in G[g]])
