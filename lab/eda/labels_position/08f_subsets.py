"""Train on likely-random (p_random>0.9) vs likely-production (p<0.05) vs uniform train feeds (all M>=50, M-bin matched, equal size);
evaluate on val (all feeds and M>=50 feeds)."""
from xgbexp import *
from featlib import build, add_hist_sim
import os
nb.set_num_threads(24)
RES = f'{OUT}/08f_subsets.json'
R = json.load(open(RES)) if os.path.exists(RES) else {}
N = 50_000
d = np.load(f'{OUT}/train_prandom.npz'); p = d['p_random']
F = feeds('train'); M = F['M'].to_numpy()
rng = np.random.default_rng(5)
elig = (M >= 50) & ~np.isnan(p)
Rset = np.where(elig & (p > 0.9))[0]; Rset = np.sort(rng.choice(Rset, size=min(N, len(Rset)), replace=False))
edges = [50, 75, 100, 150, 200, 250, 301]
def matched(pool_mask):
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        need = int(((M[Rset] >= lo) & (M[Rset] < hi)).sum())
        pool = np.where(pool_mask & (M >= lo) & (M < hi))[0]
        out.append(rng.choice(pool, size=need, replace=False))
    return np.sort(np.concatenate(out))
Pset = matched(elig & (p < 0.05)); Uset = matched(elig)
print('sizes', len(Rset), len(Pset), len(Uset), 'mean p', p[Rset].mean(), p[Pset].mean(), p[Uset].mean(), flush=True)
dva = pl.read_parquet(f'{OUT}/feat_val.parquet')
gv = groups(dva); ov = offsets(gv); tv = dva['target'].to_numpy(); big = gv >= 50
def ev(bst, feats):
    s = bst.predict(dmat(dva, feats, group=False))
    pf = ndcg_pf(s, tv, ov, gv)
    return pf
for nm, sel in [('R_random_like', Rset), ('P_prod_like', Pset), ('U_uniform', Uset)]:
    df, off, Mx, sel = build('train', sel)
    df = add_hist_sim(df, 'train', sel, off, Mx)
    st = {'pos_rate': float(df['target'].mean()), 'M_mean': float(Mx.mean())}
    for fs_name, feats in [('basic', BASIC), ('rich', RICH)]:
        key = f'{nm}_{fs_name}'
        if key in R:
            continue
        bst = xgb.train(params('rank'), dmat(df, feats), num_boost_round=400)
        pf = ev(bst, feats)
        np.save(f'{OUT}/pf_{key}.npy', pf)
        R[key] = dict(val_ndcg=float(pf.mean()), val_ndcg_M50plus=float(pf[big].mean()), val_ndcg_Mlt50=float(pf[~big].mean()), **st)
        print(key, R[key], flush=True)
        json.dump(R, open(RES, 'w'), indent=1)
# paired comparisons
for fs_name in ['basic', 'rich']:
    a = np.load(f'{OUT}/pf_R_random_like_{fs_name}.npy')
    for other in ['P_prod_like', 'U_uniform']:
        b = np.load(f'{OUT}/pf_{other}_{fs_name}.npy')
        m, se = paired(a, b)
        R[f'paired_R_minus_{other}_{fs_name}'] = dict(diff=float(m), se=float(se))
        print(fs_name, 'R -', other, round(m, 4), '+-', round(se, 4))
json.dump(R, open(RES, 'w'), indent=1)
