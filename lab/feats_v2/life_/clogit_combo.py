"""Joint conditional-logit of target on base OOF score + a feature set (quantile transformed, NaN indicators);
reports LR and the NDCG of the fitted linear score vs the base OOF (in-sample, few params; plus a 2-fold
cross-fitted NDCG by user-grouped feed halves to remove the optimism)."""
import sys
sys.path.insert(0, "<repo>/src"); sys.path.insert(0, "<repo>/feats_v2/life_")
from common import *
import clogit_screen as C

f, M, off = feed_meta("val")
y = np.load(FLAT / "val/cand_target.npy").astype(np.float64)
fidx = np.load(FLAT / "val/cand_fidx.npy")
oof = np.load(sys.argv[2] if len(sys.argv) > 2 else C.BASE_OOF)
names = open(sys.argv[1]).read().strip().split(",") if not "," in sys.argv[1] else sys.argv[1].split(",")
s = (oof - oof.mean()) / oof.std()
cols = [s]
for n in names:
    q, nan = C.qtrans(np.load(FEATS / "val" / f"{n}.npy").astype(np.float64))
    cols.append(q)
    if 0 < nan.mean() < 1:
        cols.append(nan)
Z = np.stack(cols, 1)
npos = np.bincount(fidx, weights=y, minlength=len(M))
users = f["user_id"].to_numpy()
half = (np.random.default_rng(0).permutation(np.unique(users))[: len(np.unique(users)) // 2])
fh = np.isin(users, half)
noise = np.random.default_rng(1).random(len(y)) * 1e-9
pb = ndcg_per_feed(oof + noise, y, off, M)
def fit_on(feedmask):
    rows = (feedmask & (npos > 0))[fidx]
    y2 = y[rows]; fid2 = fidx[rows]; st = np.r_[0, np.flatnonzero(np.diff(fid2)) + 1]
    ll0, _ = C.fit(Z[rows][:, :1], y2, st, fid2, npos[feedmask & (npos > 0)])
    ll1, w = C.fit(Z[rows], y2, st, fid2, npos[feedmask & (npos > 0)])
    return ll0, ll1, w
ll0, ll1, w = fit_on(np.ones(len(M), bool))
pin = ndcg_per_feed(Z @ w + noise, y, off, M)
# cross-fit
sc = np.zeros(len(y))
for m in (fh, ~fh):
    _, _, wm = fit_on(~m)
    rows = m[fidx]
    sc[rows] = (Z[rows] @ wm)
pcf = ndcg_per_feed(sc + noise, y, off, M)
d1, d2 = pin - pb, pcf - pb
print(f"n_cols={Z.shape[1]-1} LR={2*(ll1-ll0):.1f}  base={pb.mean():.5f}  in-sample delta={d1.mean():+.5f} (se {d1.std()/np.sqrt(len(d1)):.5f})  cross-fit delta={d2.mean():+.5f} (se {d2.std()/np.sqrt(len(d2)):.5f})")
