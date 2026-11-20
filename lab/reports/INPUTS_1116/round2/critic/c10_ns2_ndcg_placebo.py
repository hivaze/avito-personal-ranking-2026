"""CRITIC2 c10: calibration of the nested NDCG read of ns2_5:z on val (c09: +0.00087 +- 0.00029 on T, +0.00051 +- 0.00025 on N). Labels opened.
The same nested read (5 user-grouped folds, both arms re-weight the control block + nx3) for placebo columns:
  U  NPL unconditional within-feed permutations of ns2_5 (raw), form recomputed
  S  NPL permutations inside the feed among rows of the same popularity stratum (as c07): keeps the column's relation to the popularity inputs
Printed: the true read, placebo mean / sd / max, at equal weights and at the pmix-private weights; and bt5:rk, kn1_5:z alone for completeness.
A second fold assignment (seed 777) for the true columns shows the fold noise.
usage: c10_ns2_ndcg_placebo.py    output out/c10_ns2_ndcg_placebo.txt"""
from cc_common import *
import pmix
P = Log(OUT / 'c10_ns2_ndcg_placebo.txt')
V = rs.Views(None)
NPL = 12; NB = 8; W = pmix.weights()
INP = ['it_hist_all_luo', 'it_hist_et1_luo', 'seq_cv_invd20', 'seq_cv_pop']


def fitth(D, X, use=None, th0=None):
    if th0 is None:
        th0 = np.zeros(X.shape[1]); th0[0] = 1.0
    return condlib.fit_big(D, X, use=use, theta0=th0, iters=25)['theta']


for wit in ('T', 'N'):
    q = {'T': 'val_twin', 'N': 'val_nat'}[wit]
    D = lens.load_split('val', 'TO16'); condlib.sp = 'val'; cn, cl = condlib.control_block(D); off, M = D['off'], D['M']; fid = np.repeat(np.arange(D['n']), M)
    if wit == 'T':
        nz = np.load(NX3 / 'staleval/nx_rp3_z.npy').astype(np.float32); nr = np.load(NX3 / 'staleval/nx_rp3_rk.npy').astype(np.float32)
    else:
        nz = np.load(ROOT / 'data/feats/val/nx_rp3_z.npy').astype(np.float32); nr = np.load(ROOT / 'data/feats/val/nx_rp3_rk.npy').astype(np.float32)
    X0 = np.ascontiguousarray(np.stack([D['base']] + cl + [nz, nr], 1).astype(np.float32)); t0 = fitth(D, X0)
    folds = {'folds of record': D['fold'], 'second fold draw': lens.folds_by_user(D['uid'], seed=777)}
    base = {}
    for fn, fo in folds.items():
        a0 = [fitth(D, X0, use=fo != k, th0=t0) for k in range(5)]
        base[fn] = (fo, a0, [lens.ndcg((X0 @ a.astype(np.float32)).astype(np.float32), D) for a in a0])

    def nested(x, fn='folds of record'):
        fo, a0, e0 = base[fn]; X1 = np.ascontiguousarray(np.concatenate([X0, x[:, None].astype(np.float32)], 1)); d = np.zeros(D['n'])
        for k in range(5):
            te = fo == k; a1 = fitth(D, X1, use=~te, th0=np.concatenate([a0[k], [0.0]]))
            d[te] = (lens.ndcg((X1 @ a1.astype(np.float32)).astype(np.float32), D) - e0[k])[te]
        return d

    def show(d):
        return f'all {d.mean():+.5f} +- {d.std() / np.sqrt(len(d)):.5f} | pmix-private {pmix.wmean_se(d, W["private"])[0]:+.5f} +- {pmix.wmean_se(d, W["private"])[1]:.5f}'

    P(f'\n=== witness {wit} ===')
    raw = rget(q, 'ns2_5'); x = form(raw, off, M, 'log1p', 'z')
    for fn in folds:
        P(f'  ns2_5:z, {fn}: ' + show(nested(x, fn)))
    for col, t, fm in (('bt5', None, 'rk'), ('kn1_5', None, 'z')):
        P(f'  {col}:{fm}, folds of record: ' + show(nested(form(rget(q, col), off, M, t, fm))))
    Z = np.column_stack([form(np.load(ROOT / f'nn/feat_cache/val/{n_}.npy').astype(np.float64), off, M, 'log1p', 'z') for n_ in INP] + [np.ones(len(x), np.float32)]).astype(np.float64)
    beta, *_ = np.linalg.lstsq(Z, x.astype(np.float64), rcond=None); pred = Z @ beta
    st = np.minimum((NB * (lens.feed_rk(pred, off, M) + 0.5)).astype(np.int64), NB - 1); key = fid * NB + st; ordk = np.argsort(key, kind='stable')
    rng = np.random.default_rng(1234 if wit == 'T' else 1235)
    for kind in ('U unconditional', 'S within popularity strata'):
        a_, p_ = [], []
        for s in range(NPL):
            if kind[0] == 'U':
                xp = raw[perm_order(fid, rng)]
            else:
                o = np.argsort(key + rng.random(len(key)), kind='stable'); xp = np.empty_like(raw); xp[ordk] = raw[o]
            d = nested(form(xp, off, M, 'log1p', 'z')); a_.append(d.mean()); p_.append(pmix.wmean_se(d, W['private'])[0])
        a_, p_ = np.array(a_), np.array(p_)
        P(f'  placebo {kind} ({NPL} draws): all mean {a_.mean():+.5f}, sd {a_.std():.5f}, max {a_.max():+.5f} | pmix-private mean {p_.mean():+.5f}, sd {p_.std():.5f}, max {p_.max():+.5f}')
    del D
P('DONE')
