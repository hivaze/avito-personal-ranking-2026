"""Step 6: LABEL-FREE transfer check. Gate features and gate outputs on val vs public-like vs private-like test feeds (test labels do not exist here; val labels are not loaded)."""
from gate import *
fv, Mv, ov = meta('val'); ft, Mt, ot = meta('test'); qv = pools4('val', None, None, tag='FD16P4'); qt = pools4('test', None, None, tag='FD16P4')
Xv = features('val', qv); Xt = features('test', qt); ts = ft['timestamp'].to_numpy(); pub = ts < BOUND; pri = ~pub; J = json.load(open(HERE / 'out_step3_FD16P4.json')); out = {'features': {}, 'outputs': {}}
def psi(a, b, k=10):
    e = np.unique(np.quantile(a, np.linspace(0, 1, k + 1))); e[0] = -np.inf; e[-1] = np.inf
    pa = np.histogram(a, e)[0] / len(a) + 1e-4; pb = np.histogram(b, e)[0] / len(b) + 1e-4; return float(((pb - pa) * np.log(pb / pa)).sum())
print(f'feeds: val {len(Mv)}, public-like {pub.sum()}, private-like {pri.sum()}')
print('GATE FEATURES (val winsor bounds = [q1%, q99%] of val; "beyond" = share of feeds strictly outside one bound; REJECT rule (i): > 5% beyond one bound on public-like or private-like)')
print(f'{"feat":5s} | {"val mean":>8s} {"pub":>8s} {"priv":>8s} | {"val q1":>8s} {"q50":>8s} {"q99":>8s} | {"min val":>8s} {"min te":>8s} {"max val":>8s} {"max te":>8s} | below q1: val pub priv | above q99: val pub priv | PSI pub priv | rule (i)')
for k, name in enumerate(FEATS):
    v = Xv[:, k]; a = Xt[pub, k]; b = Xt[pri, k]; lo, md, hi = np.quantile(v, [.01, .5, .99]); bl = [float((x < lo).mean()) for x in (v, a, b)]; ab = [float((x > hi).mean()) for x in (v, a, b)]
    rej = max(bl[1], bl[2], ab[1], ab[2]) > 0.05; out['features'][name] = dict(mean=[float(v.mean()), float(a.mean()), float(b.mean())], q=[float(lo), float(md), float(hi)], below=bl, above=ab, psi=[psi(v, a), psi(v, b)], minmax=[float(v.min()), float(Xt[:, k].min()), float(v.max()), float(Xt[:, k].max())], reject=bool(rej))
    print(f'{name:5s} | {v.mean():8.3f} {a.mean():8.3f} {b.mean():8.3f} | {lo:8.3f} {md:8.3f} {hi:8.3f} | {v.min():8.3f} {Xt[:, k].min():8.3f} {v.max():8.3f} {Xt[:, k].max():8.3f} | {bl[0]:.3f} {bl[1]:.3f} {bl[2]:.3f} | {ab[0]:.3f} {ab[1]:.3f} {ab[2]:.3f} | {psi(v, a):.3f} {psi(v, b):.3f} | {"REJECT" if rej else "ok"}')
# gate outputs of the all-feed fits (FD16P4)
allf = np.arange(len(Mv)); lo = np.quantile(Xv, 0.01, axis=0); hi = np.quantile(Xv, 0.99, axis=0); mu = np.clip(Xv, lo, hi).mean(0); sd = np.clip(Xv, lo, hi).std(0)
st = lambda X: (np.clip(X, lo, hi) - mu) / sd
def w_of(k, d, X):
    n = len(X)
    if k == 'MRULE': return np.where(X[:, 0] >= np.log(50) - 1e-9, 0.4, 0.2), np.full(n, 0.5)
    if k in ('CONSTp', 'CONST2'): return np.full(n, d['wp']), np.full(n, d['wf'])
    if k == 'MSTEP': h = X[:, 0] >= np.log(d['cut']) - 1e-9; return np.where(h, d['hi'], d['lo']), np.full(n, 0.5)
    if k == 'STUMP1': m = X[:, FEATS.index(d['feat'])] > d['cut']; return np.where(m, d['hi'][0], d['lo'][0]), np.where(m, d['hi'][1], d['lo'][1])
    if k == 'PSTUMP':
        mp = X[:, FEATS.index(d['pop']['feat'])] > d['pop']['cut'] if d['pop']['feat'] else np.zeros(n, bool); mf = X[:, FEATS.index(d['ft']['feat'])] > d['ft']['cut'] if d['ft']['feat'] else np.zeros(n, bool)
        return np.where(mp, d['pop']['hi'], d['pop']['lo']), np.where(mf, d['ft']['hi'], d['ft']['lo'])
    return lin_out(st(X), d['a0'], np.array(d['a']), d['b0'], np.array(d['b']))
print('\nGATE OUTPUTS of the all-feed fits on FD16P4 (label-free application to test). REJECT rule (ii): > 1% of public-like / private-like feeds outside the val output range; rule (iii): an output cell with >= 20% of private-like feeds and < 5% of val feeds')
print(f'{"gate":7s} | w_pop mean val pub priv | w_ft mean val pub priv | w_pop range val -> share of test outside (pub, priv) | cells (w_pop, w_ft): share val / pub / priv | verdict')
pick = J['fam']['AUTO']['full']['pick']
for k in ('MRULE', 'MSTEP', 'STUMP1', 'PSTUMP', 'LINM', 'LINp', 'LIN2'):
    d = J['fam'][k]['full'] if k != 'MRULE' else None; wv = w_of(k, d, Xv); wt = w_of(k, d, Xt); r1 = (wv[0].min(), wv[0].max()); r2 = (wv[1].min(), wv[1].max())
    o = [float(np.mean((wt[0][m] < r1[0] - 1e-9) | (wt[0][m] > r1[1] + 1e-9) | (wt[1][m] < r2[0] - 1e-9) | (wt[1][m] > r2[1] + 1e-9))) for m in (pub, pri)]
    cells = ''; rej3 = False
    if k in ('MRULE', 'MSTEP', 'STUMP1', 'PSTUMP'):
        keyv = np.round(wv[0], 2) * 10 + np.round(wv[1], 2); keyt = np.round(wt[0], 2) * 10 + np.round(wt[1], 2)
        for c in np.unique(np.concatenate([keyv, keyt])):
            sv, sp_, sr = (keyv == c).mean(), (keyt[pub] == c).mean(), (keyt[pri] == c).mean(); i = np.flatnonzero(keyt == c)[0] if (keyt == c).any() else None
            wpc = wt[0][i] if i is not None else wv[0][np.flatnonzero(keyv == c)[0]]; wfc = wt[1][i] if i is not None else wv[1][np.flatnonzero(keyv == c)[0]]
            cells += f' ({wpc:.2f},{wfc:.2f}): {sv:.3f}/{sp_:.3f}/{sr:.3f};'; rej3 |= (sr >= 0.2 and sv < 0.05)
    rej = max(o) > 0.01 or rej3
    out['outputs'][k] = dict(wpop_mean=[float(wv[0].mean()), float(wt[0][pub].mean()), float(wt[0][pri].mean())], wft_mean=[float(wv[1].mean()), float(wt[1][pub].mean()), float(wt[1][pri].mean())], outside=o, cells=cells, reject=bool(rej), is_auto_pick=(k == pick))
    print(f'{k:7s}{"*" if k == pick else " "}| {wv[0].mean():.3f} {wt[0][pub].mean():.3f} {wt[0][pri].mean():.3f} | {wv[1].mean():.3f} {wt[1][pub].mean():.3f} {wt[1][pri].mean():.3f} | [{r1[0]:.3f}, {r1[1]:.3f}] -> {o[0]:.4f}, {o[1]:.4f} |{cells} | {"REJECT" if rej else "ok"}')
print('(* = AUTO\'s all-feed pick)')
# what the M-rule changes on test (label-free): share of feeds whose order / top-1 differs from plain
sp_ = score('test', qt, 0.2, 0.5); sm = score('test', qt, np.where(Mt >= 50, 0.4, 0.2), 0.5)
@nb.njit(parallel=True)
def top1(s, off, M):
    o = np.empty(len(M), np.int64)
    for f in nb.prange(len(M)): o[f] = np.argmax(s[off[f]:off[f] + M[f]])
    return o
t1 = top1(sp_, ot, Mt) == top1(sm, ot, Mt); print(f'\nM-rule vs plain on test (label-free): share of feeds M >= 50: public-like {np.mean(Mt[pub] >= 50):.3f}, private-like {np.mean(Mt[pri] >= 50):.3f}, val {np.mean(Mv >= 50):.3f}; same top-1 on long feeds: {t1[Mt >= 50].mean():.3f}')
out['mrule_share_long'] = [float(np.mean(Mv >= 50)), float(np.mean(Mt[pub] >= 50)), float(np.mean(Mt[pri] >= 50))]
json.dump(out, open(HERE / 'out_step6.json', 'w'), indent=1)
