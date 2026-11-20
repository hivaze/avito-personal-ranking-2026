"""PRE-DECLARED (written before running): nested user-grouped 5-fold test of a per-age-bucket score offset (7 exact-age buckets
from f4; offsets fitted by conditional-logit likelihood on 4 folds, NDCG on the 5th). Bases: G3 final, G3_to, N1 final.
Also a 'cold' (LUO==0) offset fitted jointly as a comparison arm (cold offsets were CLOSED earlier)."""
from lib import *
from scipy.optimize import minimize
from sklearn.model_selection import GroupKFold
sys.path.insert(0, str(ROOT / 'feats_v2/life_'))
import post_time as ptm
V = Val('val'); y, M, fid, n, off = V.y, V.M, V.fid, V.n, V.off
T = pl.read_parquet(OUT / 'out/feed_table.parquet'); g = lambda c: T[c].to_numpy(); keep = ~g('leak')
knots = np.load(ROOT / 'data/feats/life_f4_age_knots.npy')
a = ptm.age_hours(np.load(FLAT / 'val/cand_f4.npy'), knots); a = np.where(np.isnan(a), 1e8, a)
bins = [0, 6, 24, 72, 168, 720, 2160, 1e9]; B = np.digitize(a, bins) - 1; K = 7
cold = np.load(OUT / 'out/cold_luo.npy').astype(float)
S = dict(np.load(OUT / 'out/cand_scores.npz'))
users = g('user_id').copy(); u0 = users == 0; users[u0] = -np.arange(u0.sum()) - 1
OH = np.zeros((len(y), K)); OH[np.arange(len(y)), B] = 1
def fit(z, rows_mask, X):
    idx = np.flatnonzero(rows_mask[fid]); f_ = fid[idx]; zz = z[idx]; yy = y[idx]; XX = X[idx]
    def nll(th):
        s = zz * th[0] + XX @ th[1:]
        s = s - np.repeat(np.maximum.reduceat(s, np.r_[0, np.flatnonzero(np.diff(f_)) + 1]), np.diff(np.r_[np.r_[0, np.flatnonzero(np.diff(f_)) + 1], len(f_)]))
        e = np.exp(s); Z = np.bincount(f_, e, n)[f_]; p = e / Z
        ll = np.sum(yy * np.log(p + 1e-300))
        r = yy - p * np.bincount(f_, yy, n)[f_]
        gr = np.r_[np.sum(r * zz), XX.T @ r]
        return -ll, -gr
    th0 = np.r_[1.0, np.zeros(X.shape[1])]
    res = minimize(nll, th0, jac=True, method='L-BFGS-B')
    return res.x
for base in ['G3', 'G3_to', 'N1']:
    z = S[base].astype(np.float64); b0 = V.ndcg(z)
    for arm, X in [('age7', OH[:, 1:]), ('cold', cold[:, None]), ('age7+cold', np.column_stack([OH[:, 1:], cold]))]:
        out = np.zeros(len(y)); ths = []
        for tr, te in GroupKFold(5).split(np.zeros(n), groups=users):
            mtr = np.zeros(n, bool); mtr[tr] = True
            th = fit(z, mtr & keep, X); ths.append(th)
            mte = np.zeros(n, bool); mte[te] = True
            rows = mte[fid]
            out[rows] = z[rows] * th[0] + X[rows] @ th[1:]
        d = V.ndcg(out) - b0
        print(f'{base:6s} {arm:10s} nested {d[keep].mean():+.5f} ± {se(d[keep]):.5f}  | M<50 {d[keep & (M < 50)].mean():+.5f} M>=50 {d[keep & (M >= 50)].mean():+.5f} '
              f'coldpos {d[keep & g("coldpos")].mean():+.5f} warmpos {d[keep & g("warmpos")].mean():+.5f} Tue {d[keep & g("tue")].mean():+.5f} Thu {d[keep & ~g("tue")].mean():+.5f}')
        print('        mean coefs (scale, offsets vs <6h):', np.round(np.mean(ths, 0), 3))
