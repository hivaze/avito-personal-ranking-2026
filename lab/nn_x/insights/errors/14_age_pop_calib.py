"""act/exp of G3_to / G3_ft by exact age x LUO popularity bucket (is the young-listing miscalibration a popularity-semantics effect?)."""
from lib import *
sys.path.insert(0, str(ROOT / 'feats_v2/life_'))
import post_time as ptm
V = Val('val'); y, M, fid, n = V.y, V.M, V.fid, V.n
T = pl.read_parquet(OUT / 'out/feed_table.parquet'); keep = ~T['leak'].to_numpy(); kc = keep[fid]
knots = np.load(ROOT / 'data/feats/life_f4_age_knots.npy')
a = ptm.age_hours(np.load(FLAT / 'val/cand_f4.npy'), knots); a = np.where(np.isnan(a), 1e8, a)
ab = np.select([a < 24, a < 168, a < 720], ['<1d', '1-7d', '7-30d'], '>30d')
luo = np.nan_to_num(np.load(FC / 'val/it_hist_all_luo.npy').astype(np.float64), nan=0)
pb = np.select([luo == 0, luo <= 3, luo <= 20], ['0', '1-3', '4-20'], '>20')
S = dict(np.load(OUT / 'out/cand_scores.npz'))
P = {}
for key in ['G3_to', 'G3_ft', 'nf2_tr']:
    z = S[key].astype(np.float64); e = np.exp(z); P[key] = e / np.bincount(fid, e, n)[fid] * V.npos[fid]
print(f'{"age|pop":12s} {"cand%":>6s} {"posrate":>8s} ' + ' '.join(f'{k:>14s}' for k in P))
for A in ['<1d', '1-7d', '7-30d', '>30d']:
    for B in ['0', '1-3', '4-20', '>20']:
        m = (ab == A) & (pb == B) & kc
        if m.sum() < 2000: continue
        print(f'{A + "|" + B:12s} {m.mean()/kc.mean()*100:6.2f} {y[m].mean():8.4f} ' + ' '.join(f'{y[m].sum()/p[m].sum():7.3f}±{np.sqrt(p[m].sum())/p[m].sum():.3f}' for p in P.values()))
