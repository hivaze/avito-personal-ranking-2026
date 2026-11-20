"""CRITIC2 c09: the linear lens read in NDCG, with the control block re-weighted in BOTH arms (labels opened).
c06 part C added the columns to the pool alone; that read mixes the columns with the control block's own re-weighting (the columns' coefficients are
conditional on it), so it is replaced here by:
 N  nested, inside one witness: 5 user-grouped folds, both models fitted on 4 folds and read on the 5th:
      NDCG(pool + own inputs [+ nx3] + set) - NDCG(pool + own inputs [+ nx3])
 X  cross-witness: both models fitted on the SOURCE witness (control columns common to both), read on the TARGET.
Sets: nx3 itself over the null model without it (yardstick); over the null model with nx3: trio, pair (ns2_5:z + bt5:rk), ns2_5:z alone.
Paired per-feed mean +- 1 SE; on val also at the pmix weights; on the holdout the lag-view feeds and slots 11-13.
usage: c09_ndcg.py    output out/c09_ndcg.txt"""
from cc_common import *
import pmix
P = Log(OUT / 'c09_ndcg.txt')
V = rs.Views(None)
MAIN = [('kn1_5', None, 'z'), ('ns2_5', 'log1p', 'z'), ('bt5', None, 'rk')]
nm = lambda c: f'{c[0]}:{c[2]}'
W = pmix.weights(); DAT = {}
for wit in ('H', 'T', 'N'):
    sp = 'hold' if wit == 'H' else 'val'; q = {'H': 'hold', 'T': 'val_twin', 'N': 'val_nat'}[wit]
    D = lens.load_split(sp, 'TO16'); condlib.sp = sp; cn, cl = condlib.control_block(D); off, M = D['off'], D['M']; rows = V.rows(q)[0]
    if wit == 'H':
        nz = np.nan_to_num(np.asarray(np.load(NX3 / 'out/files/train/nx_rp3_z.npy', mmap_mode='r')[rows]).astype(np.float32), nan=0.0)
        nr = np.nan_to_num(np.asarray(np.load(NX3 / 'out/files/train/nx_rp3_rk.npy', mmap_mode='r')[rows]).astype(np.float32), nan=0.0)
    elif wit == 'T':
        nz = np.load(NX3 / 'staleval/nx_rp3_z.npy').astype(np.float32); nr = np.load(NX3 / 'staleval/nx_rp3_rk.npy').astype(np.float32)
    else:
        nz = np.load(ROOT / 'data/feats/val/nx_rp3_z.npy').astype(np.float32); nr = np.load(ROOT / 'data/feats/val/nx_rp3_rk.npy').astype(np.float32)
    DAT[wit] = dict(D=D, cn=cn, cl=dict(zip(cn, cl)), nx=[nz, nr], X={nm(c): form(rget(q, c[0]), off, M, c[1], c[2]) for c in MAIN})
names = [nm(c) for c in MAIN]
SETS = [('nx3 over the null without it', None), ('trio over null + nx3', names), ('pair ns2_5:z + bt5:rk', names[1:]), ('ns2_5:z', names[1:2])]


def mat(wit, ctrl, with_nx, extra):
    d = DAT[wit]
    return np.ascontiguousarray(np.stack([d['D']['base']] + [d['cl'][n_] for n_ in ctrl] + (d['nx'] if with_nx else []) + [d['X'][k] for k in extra], 1).astype(np.float32))


def cells(wit, d):
    if wit == 'H':
        sl = V.A['train']['slot'][V.hold]; has = sl >= 6; tl = sl >= 11
        return [f'lag-view feeds {d[has].mean():+.5f} +- {d[has].std() / np.sqrt(has.sum()):.5f}', f'slots 11-13 {d[tl].mean():+.5f} +- {d[tl].std() / np.sqrt(tl.sum()):.5f}']
    return [f'all {d.mean():+.5f} +- {d.std() / np.sqrt(len(d)):.5f}'] + [f'{wn} {pmix.wmean_se(d, W[wk])[0]:+.5f} +- {pmix.wmean_se(d, W[wk])[1]:.5f}' for wn, wk in (('pmix-public', 'public'), ('pmix-private', 'private'))]


def fitth(D, X, use=None, th0=None):
    if th0 is None:
        th0 = np.zeros(X.shape[1]); th0[0] = 1.0
    return condlib.fit_big(D, X, use=use, theta0=th0, iters=25)['theta']


P('=== N. nested inside one witness (5 user-grouped folds; both arms re-weight the control block) ===')
for wit in ('H', 'T', 'N'):
    D = DAT[wit]['D']; ctrl = DAT[wit]['cn']
    for sn, ks in SETS:
        X0 = mat(wit, ctrl, ks is not None, []); X1 = mat(wit, ctrl, True, ks or [])
        t0 = fitth(D, X0); t1 = fitth(D, X1); d = np.zeros(D['n'])
        for k in range(5):
            tr = D['fold'] != k
            a0 = fitth(D, X0, use=tr, th0=t0); a1 = fitth(D, X1, use=tr, th0=t1)
            e = lens.ndcg((X1 @ a1.astype(np.float32)).astype(np.float32), D) - lens.ndcg((X0 @ a0.astype(np.float32)).astype(np.float32), D); d[~tr] = e[~tr]
        P(f'  {wit} | {sn:30s} | ' + ' | '.join(cells(wit, d)))
P('=== X. cross-witness (fitted on the source, read on the target; control columns common to both) ===')
for src, tg in (('H', 'T'), ('H', 'N'), ('T', 'H'), ('N', 'H')):
    ctrl = [n_ for n_ in DAT[src]['cn'] if n_ in DAT[tg]['cl']]; Ds, Dt = DAT[src]['D'], DAT[tg]['D']
    for sn, ks in SETS:
        t0 = fitth(Ds, mat(src, ctrl, ks is not None, [])); t1 = fitth(Ds, mat(src, ctrl, True, ks or []))
        d = lens.ndcg((mat(tg, ctrl, True, ks or []) @ t1.astype(np.float32)).astype(np.float32), Dt) - lens.ndcg((mat(tg, ctrl, ks is not None, []) @ t0.astype(np.float32)).astype(np.float32), Dt)
        P(f'  {src} -> {tg} ({len(ctrl)} common controls) | {sn:30s} | ' + ' | '.join(cells(tg, d)))
P('DONE')
