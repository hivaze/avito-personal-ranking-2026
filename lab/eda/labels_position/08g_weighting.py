"""Extra: (g1) 200K uniform train sample with group weight 1+4*p_random; (g2) random-like 50K set + pos feature fixed at inference;
(g3) 200K uniform sample re-weighted to val's M distribution."""
from xgbexp import *
from featlib import build, add_hist_sim
import os
nb.set_num_threads(24)
RES = f'{OUT}/08g_weighting.json'
R = json.load(open(RES)) if os.path.exists(RES) else {}
dva = pl.read_parquet(f'{OUT}/feat_val.parquet')
gv = groups(dva); ov = offsets(gv); tv = dva['target'].to_numpy()
def ev(bst, feats, override=None):
    out = {}
    dm = dmat(dva, feats, group=False, override=override)
    for c in [200, 400]:
        s = bst.predict(dm, iteration_range=(0, c)); pf = ndcg_pf(s, tv, ov, gv); out[c] = float(pf.mean())
        np.save(f'{OUT}/pf_{name}_{c}.npy', pf)
    return out
dtr = pl.read_parquet(f'{OUT}/feat_train200k.parquet').with_columns(pl.col('pos').cast(pl.Float32))
pr = np.load(f'{OUT}/train_prandom.npz')['p_random']
fidx_g = dtr.group_by('fid', maintain_order=True).agg(pl.col('fidx').first(), pl.col('M').first())
pg = pr[fidx_g['fidx'].to_numpy()]
pg = np.where(np.isnan(pg), np.nanmean(pg), pg)
name = 'g1_train200k_rich_w1+4p'
if name not in R:
    bst = xgb.train(params('rank'), dmat(dtr, RICH, gweight=(1 + 4 * pg).astype(np.float32)), num_boost_round=400)
    R[name] = ev(bst, RICH); print(name, R[name], flush=True); json.dump(R, open(RES, 'w'), indent=1)
# g3: M-distribution reweighting to val
edges = np.array([5, 10, 20, 30, 50, 75, 100, 150, 200, 250, 301])
Mt = fidx_g['M'].to_numpy(); Mv = gv
bt = np.digitize(Mt, edges[1:-1]); bv = np.digitize(Mv, edges[1:-1])
ratio = (np.bincount(bv, minlength=10) / len(bv)) / (np.bincount(bt, minlength=10) / len(bt))
name = 'g3_train200k_rich_Mreweight'
if name not in R:
    bst = xgb.train(params('rank'), dmat(dtr, RICH, gweight=ratio[bt].astype(np.float32)), num_boost_round=400)
    R[name] = ev(bst, RICH); print(name, R[name], flush=True); json.dump(R, open(RES, 'w'), indent=1)
# g2: random-like set + pos
d = np.load(f'{OUT}/train_prandom.npz'); p = d['p_random']
F = feeds('train'); M = F['M'].to_numpy()
rng = np.random.default_rng(5)
elig = (M >= 50) & ~np.isnan(p)
Rset = np.where(elig & (p > 0.9))[0]; Rset = np.sort(rng.choice(Rset, size=50_000, replace=False))  # same draw as 08f
df, off, Mx, sel = build('train', Rset); df = add_hist_sim(df, 'train', sel, off, Mx).with_columns(pl.col('pos').cast(pl.Float32))
for pv in [5, 20]:
    name = f'g2_Rset_rich_pos{pv}'
    if name not in R:
        if 'bst_g2' not in globals():
            bst_g2 = xgb.train(params('rank'), dmat(df, RICH + ['pos']), num_boost_round=400)
        R[name] = ev(bst_g2, RICH + ['pos'], override={'pos': pv}); print(name, R[name], flush=True); json.dump(R, open(RES, 'w'), indent=1)
