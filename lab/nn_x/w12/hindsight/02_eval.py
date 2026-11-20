"""Teacher vs teacher + past-control vs teacher + hindsight. MEASUREMENT ONLY.
Combiner: (a) LightGBM binary, init_score = 1-D logistic fit of the click on the teacher (so the teacher ranking is kept exactly
when the trees add nothing), (b) logistic regression on within-feed z-scored features. 2-fold cross-fit, folds grouped by user.
Protocol HOLD: only holdout pair feeds (teacher out-of-sample), 3 repeats of the user split, per-feed NDCG averaged over repeats.
Protocol INS : non-holdout sample (teacher IN-SAMPLE), 1 split."""
import os, sys, json, time
os.environ.setdefault('OMP_NUM_THREADS', '12')
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, '<repo>/src')
from metrics import ndcg_per_feed
ROOT = '<repo>'; OUT = f'{ROOT}/nn_x/w12/hindsight/out'
t0 = time.time()
def log(*a): print(f'[{time.time()-t0:7.1f}s]', *a, flush=True)

cols = json.load(open(f'{OUT}/cols.json')); SETS, FEATS = cols['sets'], cols['feats']; NF = len(FEATS)
fm = pd.read_parquet(f'{OUT}/feeds_sel.parquet'); F = np.load(f'{OUT}/F.npy'); rows = np.load(f'{OUT}/rows.npy')
teacher = np.load(f'{ROOT}/nn_x/w12/kd/out/teacher_mean11.npy', mmap_mode='r')
ytr = np.load(f'{ROOT}/data/flat/train/cand_target.npy', mmap_mode='r')
assert len(teacher) == len(ytr) == 246745836
M = fm.M.values.astype(np.int64); off = fm.row0.values.astype(np.int64); nfe = len(fm)
assert len(rows) == M.sum() == len(F) and (off == np.concatenate([[0], np.cumsum(M)[:-1]])).all()
t = np.asarray(teacher[rows], dtype=np.float64); y = np.asarray(ytr[rows]).astype(np.int8); assert np.isfinite(t).all()
fid = np.repeat(np.arange(nfe), M)
def fz(x):
    """within-feed z-score, NaN-aware; NaN -> 0 after scoring"""
    ok = np.isfinite(x); x0 = np.where(ok, x, 0.0); n = np.maximum(np.bincount(fid, ok, nfe), 1)
    mu = np.bincount(fid, x0, nfe) / n; v = np.bincount(fid, np.where(ok, (x0 - mu[fid]) ** 2, 0.0), nfe) / n
    return np.where(ok, (x0 - mu[fid]) / np.sqrt(v[fid] + 1e-12), 0.0)
log('teacher per-feed mean abs', float(np.abs(np.bincount(fid, t, nfe) / M).max()))
tz = fz(t); tc = t - (np.bincount(fid, t, nfe) / M)[fid]
assert (fm.npos.values == np.bincount(fid, y, nfe)).all()   # label / offset alignment
nd_t = ndcg_per_feed(t, y, off, M)
log('feeds', nfe, 'rows', len(y), 'holdout feeds', int(fm.hold.sum()), 'npos==0 feeds', int((fm.npos == 0).sum()))

def cols_of(s, drop_match=True):
    i = SETS.index(s); return [i * NF + j for j in range(NF) if not (drop_match and FEATS[j] == 'item_match')]
ctx = np.stack([tz, tc, np.repeat(M, M), np.repeat(fm.HA.values, M), np.repeat(np.log1p(fm.gap.values), M), np.repeat(fm.k.values, M)], 1).astype(np.float32)
ARMS = {  # name -> list of sets
    'C_ctx': [], 'P_past': ['PAST'], 'F_fut': ['FUT'], 'PF_past+fut': ['PAST', 'FUT'],
    'PNc_past+pastns1': ['PAST', 'PASTNS1'], 'PN_past+pastns1+futns1': ['PAST', 'PASTNS1', 'FUTNS1'],
    'PC_past+pastc': ['PAST', 'PASTC'], 'PCF_past+pastc+futc': ['PAST', 'PASTC', 'FUTC'],
    'L_past+LEAK': ['PAST', 'LEAK']}
ZF = ['cos_max', 'cos_mean', 'cos_top3', 'cos_max_strong', 'cos_max_view', 'cat_share', 'cat_cnt_strong', 'cat_cnt_view']
Zc = {}
def zfeat(s, drop_match=True):
    if s not in Zc:
        i = SETS.index(s); L = [fz(F[:, i * NF + FEATS.index(n)].astype(np.float64)) for n in ZF]
        if s == 'LEAK': L.append(F[:, i * NF + 15].astype(np.float64) > 0)
        Zc[s] = np.stack(L, 1).astype(np.float32)
    return Zc[s]

def run(mask_feeds, tag, reps, gbm_par, rounds):
    idx = np.where(mask_feeds)[0]; users = fm.user_id.values[idx]
    res = {a + '|gbm': np.zeros(nfe) for a in ARMS}; res.update({a + '|lr': np.zeros(nfe) for a in ARMS})
    for rep in range(reps):
        uu = np.unique(users); rng = np.random.default_rng(1000 + rep); half = rng.permutation(len(uu)) < len(uu) // 2
        fold_of_feed = half[np.searchsorted(uu, users)].astype(int)
        ffold = np.full(nfe, -1); ffold[idx] = fold_of_feed; rf = ffold[fid]
        for a, sets in ARMS.items():
            sg = np.full(len(y), np.nan); sl = np.full(len(y), np.nan)
            Xg = np.concatenate([ctx] + [F[:, cols_of(s, s != 'LEAK')] for s in sets], 1)
            Xl = np.concatenate([np.stack([tz, tc], 1).astype(np.float32)] + [zfeat(s) for s in sets], 1)
            for fo in (0, 1):
                tr = rf == 1 - fo; te = rf == fo
                b = LogisticRegression(C=1e3, max_iter=200).fit(np.stack([tz[tr], tc[tr]], 1), y[tr])
                init = lambda m: b.decision_function(np.stack([tz[m], tc[m]], 1))
                ds = lgb.Dataset(Xg[tr], y[tr], init_score=init(tr), free_raw_data=False)
                g = lgb.train(dict(objective='binary', verbose=-1, num_threads=12, seed=7 + rep, **gbm_par), ds, rounds)
                sg[te] = init(te) + g.predict(Xg[te], raw_score=True)
                l = LogisticRegression(C=1.0, max_iter=300).fit(Xl[tr], y[tr]); sl[te] = l.decision_function(Xl[te])
            sub_rows = rf >= 0
            for nm, s in (('gbm', sg), ('lr', sl)):
                assert np.isfinite(s[sub_rows]).all()
                res[f'{a}|{nm}'] += ndcg_per_feed(np.where(sub_rows, s, 0.0), y, off, M) / reps
            log(tag, 'rep', rep, a, 'gbm', float(res[a + '|gbm'][idx].mean() * reps / (rep + 1)), 'lr', float(res[a + '|lr'][idx].mean() * reps / (rep + 1)))
    D = pd.DataFrame({k: v[idx] for k, v in res.items()}); D['T0_teacher'] = nd_t[idx]
    D = pd.concat([D, fm.iloc[idx].reset_index(drop=True)], axis=1); D.to_parquet(f'{OUT}/perfeed_{tag}.parquet'); return D

def cse(d, users):
    """mean and user-clustered SE of a per-feed delta"""
    d = np.asarray(d, float); n = len(d)
    if n < 2: return float('nan'), float('nan')
    m = d.mean(); s = pd.Series(d - m).groupby(np.asarray(users)).sum().values
    return m, float(np.sqrt((s ** 2).sum()) / n)

def report(D, tag):
    P = lambda *a: print(*a, flush=True)
    P(f'\n================ {tag}: {len(D)} feeds, {D.user_id.nunique()} users, mean M {D.M.mean():.1f}, mean npos {D.npos.mean():.2f} ================')
    P(f"teacher alone NDCG {D.T0_teacher.mean():.5f}")
    for cmb in ('gbm', 'lr'):
        P(f'--- combiner {cmb}: arm NDCG | delta vs teacher (user-clustered SE)')
        for a in ARMS:
            m, s = cse(D[f'{a}|{cmb}'] - D.T0_teacher, D.user_id); P(f'  {a:26s} {D[f"{a}|{cmb}"].mean():.5f} | {m:+.5f} +- {s:.5f}')
        pairs = [('past control - ctx', 'P_past', 'C_ctx'), ('FUT beyond past   (PF - P)', 'PF_past+fut', 'P_past'),
                 ('FUT alone vs ctx  (F - C)', 'F_fut', 'C_ctx'),
                 ('FUTNS1 beyond past (PN - PNc)', 'PN_past+pastns1+futns1', 'PNc_past+pastns1'),
                 ('FUT chain beyond past (PCF - PC)', 'PCF_past+pastc+futc', 'PC_past+pastc'), ('LEAK sanity (L - P)', 'L_past+LEAK', 'P_past')]
        cuts = {'all': np.ones(len(D), bool)}
        for nm, lo, hi in (('gap<1d', 0, 86400), ('gap1-3d', 86400, 3 * 86400), ('gap>3d', 3 * 86400, 1e12)):
            cuts[nm] = ((D.gap >= lo) & (D.gap < hi)).values
        for nm, lo, hi in (('nFUT=0', 0, 1), ('nFUT1-5', 1, 6), ('nFUT6-20', 6, 21), ('nFUT21-50', 21, 51), ('nFUT>50', 51, 1e9)):
            cuts[nm] = ((D.n_FUT >= lo) & (D.n_FUT < hi)).values
        for nm, lo, hi in (('nNS1=0', 0, 1), ('nNS1 1-5', 1, 6), ('nNS1 6-20', 6, 21), ('nNS1>20', 21, 1e9)):
            cuts[nm] = ((D.n_FUTNS1 >= lo) & (D.n_FUTNS1 < hi)).values
        cuts['explor(p>0.9,M>=50)'] = (D.p_random > 0.9).values; cuts['prod(p<=0.5)'] = (D.p_random <= 0.5).values
        cuts['M>=50'] = (D.M >= 50).values; cuts['M<50'] = (D.M < 50).values
        P(f'--- combiner {cmb}: paired deltas (mean +- user-clustered SE) by slice')
        P('  ' + f'{"slice":20s} {"n":>7s} {"teacher":>8s} ' + ' '.join(f'{p[0]:>32s}' for p in pairs))
        for nm, mk in cuts.items():
            if mk.sum() < 30: continue
            L = []
            for _, a, b in pairs:
                m, s = cse((D[f'{a}|{cmb}'] - D[f'{b}|{cmb}'])[mk], D.user_id[mk]); L.append(f'{m:+.5f} +- {s:.5f}'.rjust(32))
            P('  ' + f'{nm:20s} {int(mk.sum()):7d} {D.T0_teacher[mk].mean():8.5f} ' + ' '.join(L))

if __name__ == '__main__':
    # standalone single-feature NDCG (NaN -> -9), all selected feeds
    print('standalone NDCG of single features (all selected feeds; teacher %.5f):' % nd_t.mean())
    for s in SETS:
        i = SETS.index(s)
        print('  %-8s ' % s + '  '.join('%s %.4f' % (n, ndcg_per_feed(np.nan_to_num(F[:, i * NF + FEATS.index(n)].astype(np.float64), nan=-9), y, off, M).mean())
                                        for n in ['cos_max', 'cos_mean', 'cos_top3', 'cos_max_strong', 'cos_max_view', 'cat_share', 'item_match']), flush=True)
    hold = fm.hold.values
    Dh = run(hold, 'HOLD', 3, dict(learning_rate=0.03, num_leaves=15, min_data_in_leaf=100, feature_fraction=0.7, lambda_l2=10.0, bagging_fraction=0.8, bagging_freq=1), 200)
    report(Dh, 'HOLDOUT pair feeds (teacher OUT-of-sample)')
    Di = run(~hold, 'INS', 1, dict(learning_rate=0.05, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.7, lambda_l2=10.0), 300)
    report(Di, 'NON-holdout sample (teacher IN-SAMPLE: flagged)')
    log('DONE')
