"""Lane A c04: gate g4 of RULE.txt = the linear lens read in NDCG, form of critic/c09_ndcg.py (labels opened).
 N  nested inside one witness: 5 user-grouped folds, both models fitted on 4 folds and read on the 5th, control block and nx3 re-weighted in BOTH arms:
      NDCG(pool + own inputs + nx3 + form) - NDCG(pool + own inputs + nx3)
    deciding cells: H lag-view feeds, H slots 11-13, T all, N all (each >= 0).
 X  cross-witness (fitted on the source, read on the target; control columns common to both): information.
 Yardstick / instrument check: nx3 over the null model without it (c09_ndcg.txt: H +0.00053 +- 0.00026 / +0.00079 +- 0.00033, T +0.00009 +- 0.00044, N +0.00009 +- 0.00045).
The entered value is multiplied by the same constant as in c03_info.py (25 / 300) so that the ridge of condlib.fit_big acts on it as on the other columns.
usage: c04_ndcg.py    output out/c04_ndcg.txt, out/c04_ndcg.json"""
import json
from cs_common import *
lens, condlib = load_lens()
import pmix
P = Log(OUT / 'c04_ndcg.txt')
P('RULE.txt md5 ' + md5(CS + '/RULE.txt'))
V = Views(None)
FORMS = [('F1 cs3a_c', 'cs3a_c', 25.0), ('F2 cs3x_c', 'cs3x_c', 300.0)]
W = pmix.weights(); DAT = {}
for wit in ('H', 'T', 'N'):
    sp = 'hold' if wit == 'H' else 'val'; q = {'H': 'hold', 'T': 'val_twin', 'N': 'val_nat'}[wit]
    D = lens.load_split(sp, 'TO16'); condlib.sp = sp; cn, cl = condlib.control_block(D); off, M = D['off'], D['M']; rows = V.rows(q)[0]
    assert D['rows'] is None or (D['rows'] == rows).all()
    if wit == 'H':
        nz = np.nan_to_num(np.asarray(np.load(NX3 / 'out/files/train/nx_rp3_z.npy', mmap_mode='r')[rows]).astype(np.float32), nan=0.0)
        nr = np.nan_to_num(np.asarray(np.load(NX3 / 'out/files/train/nx_rp3_rk.npy', mmap_mode='r')[rows]).astype(np.float32), nan=0.0)
    elif wit == 'T':
        nz = np.load(NX3 / 'staleval/nx_rp3_z.npy').astype(np.float32); nr = np.load(NX3 / 'staleval/nx_rp3_rk.npy').astype(np.float32)
    else:
        nz = np.load(ROOT / 'data/feats/val/nx_rp3_z.npy').astype(np.float32); nr = np.load(ROOT / 'data/feats/val/nx_rp3_rk.npy').astype(np.float32)
    DAT[wit] = dict(D=D, cn=cn, cl=dict(zip(cn, cl)), nx=[nz, nr], X={fn: (np.nan_to_num(get(q, col)) * sc).astype(np.float32) for fn, col, sc in FORMS})
SETS = [('nx3 over the null without it', None)] + [(f'{fn} over null + nx3', [fn]) for fn, _, _ in FORMS]


def mat(wit, ctrl, with_nx, extra):
    d = DAT[wit]
    return np.ascontiguousarray(np.stack([d['D']['base']] + [d['cl'][n_] for n_ in ctrl] + (d['nx'] if with_nx else []) + [d['X'][k] for k in extra], 1).astype(np.float32))


def cells(wit, d):
    if wit == 'H':
        sl = V.A['train']['slot'][V.hold]; has = V.has[V.hold]; tl = has & (sl >= 11); vl = has & (sl <= 8)
        return [('lag-view feeds', float(d[has].mean()), float(d[has].std() / np.sqrt(has.sum()))), ('slots 11-13', float(d[tl].mean()), float(d[tl].std() / np.sqrt(tl.sum()))), ('slots 6-8', float(d[vl].mean()), float(d[vl].std() / np.sqrt(vl.sum())))]
    return [('all', float(d.mean()), float(d.std() / np.sqrt(len(d))))] + [(f'pmix-{wk}', float(pmix.wmean_se(d, W[wk])[0]), float(pmix.wmean_se(d, W[wk])[1])) for wk in ('public', 'private')]


def fitth(D, X, use=None, th0=None):
    if th0 is None:
        th0 = np.zeros(X.shape[1]); th0[0] = 1.0
    return condlib.fit_big(D, X, use=use, theta0=th0, iters=25)['theta']


RES = {}
P('=== N. nested inside one witness (5 user-grouped folds; both arms re-weight the control block and nx3) ===')
for wit in ('H', 'T', 'N'):
    D = DAT[wit]['D']; ctrl = DAT[wit]['cn']
    for sn, ks in SETS:
        X0 = mat(wit, ctrl, ks is not None, []); X1 = mat(wit, ctrl, True, ks or [])
        t0 = fitth(D, X0); t1 = fitth(D, X1); d = np.zeros(D['n'])
        for k in range(5):
            tr = D['fold'] != k
            a0 = fitth(D, X0, use=tr, th0=t0); a1 = fitth(D, X1, use=tr, th0=t1)
            e = lens.ndcg((X1 @ a1.astype(np.float32)).astype(np.float32), D) - lens.ndcg((X0 @ a0.astype(np.float32)).astype(np.float32), D); d[~tr] = e[~tr]
        c = cells(wit, d); RES[f'N|{wit}|{sn}'] = {a: [m, s] for a, m, s in c}
        if ks:
            RES[f'N|{wit}|{sn}']['coef_full_fit'] = float(t1[-1])
        P(f'  {wit} | {sn:30s} | ' + ' | '.join(f'{a} {m:+.5f} +- {s:.5f}' for a, m, s in c) + (f' | coefficient in the full fit {t1[-1]:+.4f}' if ks else ''))
P('=== X. cross-witness (fitted on the source, read on the target; control columns common to both): information ===')
for src, tg in (('H', 'T'), ('H', 'N'), ('T', 'H'), ('N', 'H')):
    ctrl = [n_ for n_ in DAT[src]['cn'] if n_ in DAT[tg]['cl']]; Ds, Dt = DAT[src]['D'], DAT[tg]['D']
    for sn, ks in SETS:
        t0 = fitth(Ds, mat(src, ctrl, ks is not None, [])); t1 = fitth(Ds, mat(src, ctrl, True, ks or []))
        d = lens.ndcg((mat(tg, ctrl, True, ks or []) @ t1.astype(np.float32)).astype(np.float32), Dt) - lens.ndcg((mat(tg, ctrl, ks is not None, []) @ t0.astype(np.float32)).astype(np.float32), Dt)
        c = cells(tg, d); RES[f'X|{src}->{tg}|{sn}'] = {a: [m, s] for a, m, s in c}
        P(f'  {src} -> {tg} ({len(ctrl)} common controls) | {sn:30s} | ' + ' | '.join(f'{a} {m:+.5f} +- {s:.5f}' for a, m, s in c))
P('=== g4 (>= 0 on H lag-view feeds, H slots 11-13, T all, N all) ===')
for fn, _, _ in FORMS:
    v = [RES[f'N|H|{fn} over null + nx3']['lag-view feeds'][0], RES[f'N|H|{fn} over null + nx3']['slots 11-13'][0], RES[f'N|T|{fn} over null + nx3']['all'][0], RES[f'N|N|{fn} over null + nx3']['all'][0]]
    ok = all(x >= 0 for x in v); RES[f'g4|{fn}'] = dict(values=v, **{'pass': bool(ok)})
    P(f'  {fn}: H lag-view {v[0]:+.5f}, H slots 11-13 {v[1]:+.5f}, T {v[2]:+.5f}, N {v[3]:+.5f} -> g4 {"PASS" if ok else "MISS"}; T at pmix-private (holdout-only clause, bar >= 0) {RES[f"N|T|{fn} over null + nx3"]["pmix-private"][0]:+.5f} +- {RES[f"N|T|{fn} over null + nx3"]["pmix-private"][1]:.5f}')
json.dump(RES, open(OUT / 'c04_ndcg.json', 'x'), indent=1, default=float)
P('DONE')
