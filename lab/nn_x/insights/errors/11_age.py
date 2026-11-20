"""Exact listing age (f4 -> hours via the life_ knots) of positives vs negatives; G3 NDCG by youngest-positive age; levers by age."""
from lib import *
sys.path.insert(0, str(ROOT / 'feats_v2/life_'))
import post_time as ptm
V = Val('val'); y, M, fid, n = V.y, V.M, V.fid, V.n
T = pl.read_parquet(OUT / 'out/feed_table.parquet'); g = lambda c: T[c].to_numpy(); keep = ~g('leak')
knots = np.load(ROOT / 'data/feats/life_f4_age_knots.npy')
ah = ptm.age_hours(np.load(FLAT / 'val/cand_f4.npy'), knots)
ah = np.where(np.isnan(ah), 24 * 365 * 5, ah)
print('age hours quantiles (all cands):', np.round(np.quantile(ah, [.1, .25, .5, .75, .9]), 1))
print('age hours quantiles (positives):', np.round(np.quantile(ah[y > 0], [.1, .25, .5, .75, .9]), 1))
bins = [0, 6, 24, 72, 168, 720, 2160, 1e9]; lab = ['<6h', '6-24h', '1-3d', '3-7d', '7-30d', '30-90d', '>90d']
b = np.digitize(ah, bins) - 1
print(f'{"age":8s} {"cand share":>10s} {"pos rate":>9s} {"MP rate (G3)":>12s} {"mean pct rank pos":>17s}')
S = dict(np.load(OUT / 'out/cand_scores.npz')); r = V.prank(S['G3'].astype(np.float64)); pct = r / np.maximum(M[fid] - 1, 1)
kc = keep[fid]
for i, l in enumerate(lab):
    m = (b == i) & kc; mp = m & (y > 0)
    print(f'{l:8s} {m.mean()/kc.mean():10.4f} {y[m].mean():9.4f} {np.mean(r[mp] >= M[fid[mp]]/2):12.4f} {pct[mp].mean():17.4f}')
# youngest positive per feed
ya = np.full(n, np.inf); np.minimum.at(ya, fid[y > 0], ah[y > 0])
fb = np.digitize(ya, bins) - 1
D = {'G3': g('ndcg_G3'), 'G3-nf2oof': g('ndcg_G3') - g('ndcg_nf2_oof'), 'FT half': g('ndcg_G3') - g('ndcg_G3_to'), 'm02': g('ndcg_F7x_to') - g('ndcg_F7b_to'),
     'a4u': g('ndcg_a4u_to') - g('ndcg_F7_s12_to'), 'NEW': g('ndcg_N1') - g('ndcg_G3')}
madj = dict(np.load(OUT / 'out/feed_label_vars.npz', allow_pickle=True))['madj']
print('\nfeeds by youngest positive exact age:')
print(f'{"age":8s} {"share":>6s} {"Madj":>16s} ' + ' '.join(f'{k:>16s}' for k in D))
for i, l in enumerate(lab):
    m = (fb == i) & keep
    print(f'{l:8s} {m.mean()/keep.mean():6.3f} {madj[m].mean():+.4f}({se(madj[m]):.4f}) ' + ' '.join(f'{d[m].mean():+.4f}({se(d[m]):.4f})' for d in D.values()))
