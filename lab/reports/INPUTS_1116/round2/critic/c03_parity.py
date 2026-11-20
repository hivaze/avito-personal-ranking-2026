"""CRITIC2 c03 (label-free): the JOINT behaviour of the rescue columns with inputs the ranker already reads, on train rows (pool feeds, D5 lag view)
against val / public / private rows.  No label array is opened.
Populations: TRv = pool rows slots 6-8, TRt = pool rows slots 11-13, VAL = val natural (M >= 50), PUB / PRIV = test natural (M >= 50).
Entered forms: kn1_5:z, ns2_5:z (log1p), bt5:rk; yardstick nx_rp3_z (installed lag-view file on train rows, natural on val / test).
 S1  correlation of each entered form with the within-feed z of log1p(input) for it_hist_all_luo, it_hist_et1_luo, seq_cv_invd20, seq_cv_pop, nx_rp3_z,
     and the OLS R2 / coefficients of the form on these five + listing age bucket, per population
 S2  J1-style conditional shares: P(non-default | it_hist_all_luo > 0), P(non-default | it_hist_all_luo = 0), P(non-default | seq_cv_invd20 > 0)
     for ns2_5 > 0, bt5_seen, imp5 > 0, and for the walk (nx_rp3_z above the feed minimum)
 S3  domain classifier (LightGBM, 3 folds grouped by feed, 250,000 rows per side): AUC of train-lag-view rows against test rows from the inputs
     alone, + nx3 (yardstick), + each new form, + all three.  The increase of the AUC is the joint shift a column adds.
usage: c03_parity.py    output out/c03_parity.txt"""
from cc_common import *
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
sys.path.insert(0, '<repo>/nn_x/w12/stale/hz2')
import hz2_common as hc
P = Log(OUT / 'c03_parity.txt')
V = rs.Views(P)
FC = ROOT / 'nn/feat_cache'
INP = ['it_hist_all_luo', 'it_hist_et1_luo', 'seq_cv_invd20', 'seq_cv_pop']
kn = hc.knots()
POP = {}
for q in ('pool', 'val_nat', 'test_nat'):
    sp = V.Q[q][0]; rows, fl, off, M = V.rows(q); a = V.A[sp]; sel = V.Q[q][1]
    d = dict(off=off, M=M, fl=fl, slot=np.repeat(a['slot'][sel], M), big=np.repeat(M >= 50, M), n=len(rows))
    raw = {}
    for n_ in INP:
        raw[n_] = np.asarray(np.load(FC / sp / f'{n_}.npy', mmap_mode='r')[rows]).astype(np.float64)
        d[n_ + ':z'] = form(raw[n_], off, M, 'log1p', 'z')
    d['hist_pos'] = raw['it_hist_all_luo'] > 0; d['cv_pos'] = raw['seq_cv_invd20'] > 0
    nxz = np.asarray(np.load(FC / sp / 'nx_rp3_z.npy', mmap_mode='r')[rows]).astype(np.float32); nxr = np.asarray(np.load(FC / sp / 'nx_rp3_rk.npy', mmap_mode='r')[rows]).astype(np.float32)
    assert not np.isnan(nxz).any()
    d['nx_rp3_z'] = nxz; d['nx_rp3_rk'] = nxr
    d['walk_pos'] = nxr > np.repeat(np.minimum.reduceat(nxr, off), M)
    f4 = np.asarray(np.load(FLAT / sp / 'cand_f4.npy', mmap_mode='r')[rows]); d['age_b'] = hc.age_bucket(hc.age_hours(f4, kn)); d['f4'] = f4.astype(np.float32)
    for k in (1, 2, 3, 5):
        d[f'f{k}'] = np.asarray(np.load(FLAT / sp / f'cand_f{k}.npy', mmap_mode='r')[rows]).astype(np.float32)
    d['kn1_5:z'] = form(rget(q, 'kn1_5'), off, M, None, 'z'); d['ns2_5:z'] = form(rget(q, 'ns2_5'), off, M, 'log1p', 'z'); d['bt5:rk'] = form(rget(q, 'bt5'), off, M, None, 'rk')
    d['ns2_pos'] = rget(q, 'ns2_5') > 0; d['bt_seen'] = rget(q, 'bt5_seen') > 0; d['imp_pos'] = rget(q, 'imp5') > 0
    if sp == 'test':
        d['pub'] = np.repeat(a['ts'] < hc.PUB_TS, M)
    POP[q] = d
    P(f'{q}: rows {len(rows)} loaded')
pl_, vl_, te_ = POP['pool'], POP['val_nat'], POP['test_nat']
MASK = {'TRv (pool, slots 6-8)': ('pool', (pl_['slot'] >= 6) & (pl_['slot'] <= 8)), 'TRt (pool, slots 11-13)': ('pool', pl_['slot'] >= 11),
        'VAL (M >= 50)': ('val_nat', vl_['big']), 'PUB (M >= 50)': ('test_nat', te_['pub'] & te_['big']), 'PRIV (M >= 50)': ('test_nat', ~te_['pub'] & te_['big'])}
NEW = ['kn1_5:z', 'ns2_5:z', 'bt5:rk', 'nx_rp3_z']
REG = [n_ + ':z' for n_ in INP]
P('\n=== S1. correlation of the entered form with the ranker\'s inputs (within-feed z of log1p), and OLS R2 on the four inputs + nx_rp3_z + age-bucket dummies ===')
for c in NEW:
    P(f'--- {c}')
    for pn, (q, m) in MASK.items():
        d = POP[q]; x = d[c][m].astype(np.float64)
        cors = [float(np.corrcoef(x, d[r][m])[0, 1]) for r in REG + (['nx_rp3_z'] if c != 'nx_rp3_z' else [])]
        Z = np.column_stack([d[r][m] for r in REG] + ([d['nx_rp3_z'][m]] if c != 'nx_rp3_z' else []) + [(d['age_b'][m] == k).astype(np.float64) for k in range(7)])
        beta, *_ = np.linalg.lstsq(Z, x, rcond=None); r2 = 1 - ((x - Z @ beta) ** 2).sum() / ((x - x.mean()) ** 2).sum()
        P(f'  {pn:24s} sd {x.std():.3f} | corr with {", ".join(f"{r.split(":")[0]} {v:+.3f}" for r, v in zip(REG + ["nx_rp3_z"], cors))} | OLS R2 {r2:.3f}; coefficients {" ".join(f"{b:+.3f}" for b in beta[:len(cors)])}')
P('\n=== S2. J1-style conditional shares ===')
for flag in ('ns2_pos', 'bt_seen', 'imp_pos', 'walk_pos'):
    P(f'--- {flag}')
    for pn, (q, m) in MASK.items():
        d = POP[q]; f = d[flag][m]; h = d['hist_pos'][m]; cv = d['cv_pos'][m]
        P(f'  {pn:24s} share {f.mean():.4f} | P(. | it_hist_all_luo > 0) {f[h].mean():.4f} (it_hist > 0 on {h.mean():.4f}) | P(. | it_hist_all_luo = 0) {f[~h].mean():.4f} | P(. | seq_cv_invd20 > 0) {f[cv].mean():.4f} | P(it_hist > 0 | .) {h[f].mean():.4f}')
P('\n=== S3. domain classifier: train lag-view rows (label 0) against test / val rows (label 1); AUC, 3 folds grouped by feed ===')
BASE = REG + ['f1', 'f2', 'f3', 'f4', 'f5']
SETS = [('inputs only', BASE), ('+ nx3 (yardstick)', BASE + ['nx_rp3_z', 'nx_rp3_rk']), ('+ nx3 + kn1_5:z', BASE + ['nx_rp3_z', 'nx_rp3_rk', 'kn1_5:z']),
        ('+ nx3 + ns2_5:z', BASE + ['nx_rp3_z', 'nx_rp3_rk', 'ns2_5:z']), ('+ nx3 + bt5:rk', BASE + ['nx_rp3_z', 'nx_rp3_rk', 'bt5:rk']),
        ('+ nx3 + all three', BASE + ['nx_rp3_z', 'nx_rp3_rk', 'kn1_5:z', 'ns2_5:z', 'bt5:rk'])]
rng = np.random.default_rng(4242); NS = 250000
for a_, b_ in (('TRt (pool, slots 11-13)', 'PRIV (M >= 50)'), ('TRt (pool, slots 11-13)', 'PUB (M >= 50)'), ('TRv (pool, slots 6-8)', 'VAL (M >= 50)')):
    parts = []
    for lab, pn in ((0, a_), (1, b_)):
        q, m = MASK[pn]; d = POP[q]; idx = np.nonzero(m)[0]; idx = np.sort(rng.choice(idx, min(NS, len(idx)), replace=False))
        parts.append((lab, d, idx))
    y = np.concatenate([np.full(len(i), l) for l, _, i in parts]); grp = np.concatenate([d['fl'][i] + l * 10_000_000 for l, d, i in parts])
    out = []
    for sn, cols in SETS:
        X = np.column_stack([np.concatenate([d[c][i] for _, d, i in parts]) for c in cols]).astype(np.float32); pr = np.zeros(len(y))
        for trn, tes in GroupKFold(3).split(X, y, grp):
            mdl = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.08, num_leaves=63, min_child_samples=200, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, n_jobs=12, verbose=-1)
            mdl.fit(X[trn], y[trn]); pr[tes] = mdl.predict_proba(X[tes])[:, 1]
        out.append((sn, roc_auc_score(y, pr)))
    P(f'{a_}  vs  {b_}: ' + ' | '.join(f'{sn} {auc:.4f}' for sn, auc in out))
P('DONE')
