"""Calibration (act/exp under the within-feed softmax) by exact listing age, TO vs FT halves; age mix val vs public/private test."""
from lib import *
sys.path.insert(0, str(ROOT / 'feats_v2/life_'))
import post_time as ptm
V = Val('val'); y, M, fid, n = V.y, V.M, V.fid, V.n
T = pl.read_parquet(OUT / 'out/feed_table.parquet'); g = lambda c: T[c].to_numpy(); keep = ~g('leak'); kc = keep[fid]
knots = np.load(ROOT / 'data/feats/life_f4_age_knots.npy')
bins = [0, 6, 24, 72, 168, 720, 2160, 1e9]; lab = ['<6h', '6-24h', '1-3d', '3-7d', '7-30d', '30-90d', '>90d']
def ageb(split):
    a = ptm.age_hours(np.load(FLAT / split / 'cand_f4.npy'), knots); a = np.where(np.isnan(a), 1e8, a)
    return np.digitize(a, bins) - 1
b = ageb('val')
S = dict(np.load(OUT / 'out/cand_scores.npz'))
cold = np.load(OUT / 'out/cold_luo.npy')
print(f'{"score":8s} ' + ' '.join(f'{l:>12s}' for l in lab))
for key in ['G3', 'G3_to', 'G3_ft', 'F7x_to', 'F7b_to', 'F10_to', 'NEW_to', 'nf2_tr', 'nf2_oof']:
    z = S[key].astype(np.float64)
    best = None
    for tmp in [0.6, 0.8, 1.0, 1.2, 1.4]:
        e = np.exp(z * tmp); p = e / np.bincount(fid, e, n)[fid] * V.npos[fid]
        ll = np.sum((y * np.log(p / V.npos[fid] + 1e-12))[kc])
        if best is None or ll > best[1]: best = (tmp, ll, p)
    p = best[2]
    print(f'{key:8s} ' + ' '.join(f'{y[(b == i) & kc].sum()/p[(b == i) & kc].sum():6.3f}±{np.sqrt(p[(b == i) & kc].sum())/p[(b == i) & kc].sum():.3f}' for i in range(len(lab))))
# Tue vs Thu for G3_to
tue = g('tue')[fid]
for dn, dm in [('Tue', tue), ('Thu', ~tue)]:
    z = S['G3_to'].astype(np.float64); e = np.exp(z); p = e / np.bincount(fid, e, n)[fid] * V.npos[fid]
    print(f'G3_to {dn}: ' + ' '.join(f'{lab[i]} {y[(b == i) & kc & dm].sum()/p[(b == i) & kc & dm].sum():.3f}' for i in range(len(lab))))
# age mix of candidates val vs public/private test
bt = ageb('test')
ft = pl.read_parquet(FLAT / 'test/feeds.parquet'); Mt = ft['M'].to_numpy(); fidt = np.repeat(np.arange(len(Mt)), Mt)
pub = (ft['timestamp'].to_numpy() < 1790497920)[fidt]
print('\ncandidate age mix: ' + ' '.join(f'{lab[i]}: val {np.mean(b == i):.3f} pub {np.mean(bt[pub] == i):.3f} priv {np.mean(bt[~pub] == i):.3f} |' for i in range(len(lab))))
# share of feeds whose youngest candidate is < 24h
for nm, bb, f_, nn in [('val', b, fid, n), ('pub', bt[pub], fidt[pub], len(Mt)), ('priv', bt[~pub], fidt[~pub], len(Mt))]:
    fr = np.bincount(f_, bb <= 1, nn) / np.maximum(np.bincount(f_, minlength=nn), 1)
    has = np.bincount(f_, minlength=nn) > 0
    print(f'{nm}: mean share of <24h candidates per feed {fr[has].mean():.4f}')
