"""P6b: teacher-conditional value of a privileged item-level label aggregate (TOP1_GAP_EXTERNAL lever 3, PFD-ITEM).
 LOO item click rate / COEC over the whole train window (other feeds only), added to teacher_mean11 on the HOLDOUT (teacher honest).
 Conditional logit [teacher_z, feat_z...] fitted on a user-half, read on the other. CPU, 2 threads."""
from lb_common import *
import polars as pl
T = ROOT / 'nn_x/w12/kd/out'
ftr = pl.read_parquet(FLAT / 'train/feeds.parquet', columns=['user_id', 'M', 'cand_off']); M = ftr['M'].to_numpy().astype(np.int64); off = ftr['cand_off'].to_numpy().astype(np.int64); uid = ftr['user_id'].to_numpy().astype(np.int64)
pr = np.nan_to_num(np.load(ROOT / 'eda/labels_position/out/train_prandom.npz')['p_random'].astype(np.float32), nan=0.0); pr[M < 50] = 0
sel = np.where((pr > 0.9) & (M >= 50))[0]; hold = np.sort(np.random.default_rng(12345).choice(sel, 30000, replace=False))
item = np.load(FLAT / 'train/cand_item_id.npy'); y = np.load(FLAT / 'train/cand_target.npy').astype(np.float32); pos = np.load(FLAT / 'train/cand_pos.npy')
print('rows', len(item), 'item id max', int(item.max()))
u, inv = np.unique(item, return_inverse=True)
n = np.bincount(inv, minlength=len(u)).astype(np.float32); c = np.bincount(inv, weights=y, minlength=len(u)).astype(np.float32)
pc = np.bincount(np.minimum(pos, 300).astype(np.int64), weights=y, minlength=301) / np.maximum(np.bincount(np.minimum(pos, 300).astype(np.int64), minlength=301), 1)
ex = np.bincount(inv, weights=pc[np.minimum(pos, 300).astype(np.int64)].astype(np.float32), minlength=len(u)).astype(np.float32)
rows = np.concatenate([np.arange(o, o + m) for o, m in zip(off[hold], M[hold])]); assert (M[hold] == Mh).all()
ir = inv[rows]; yl = y[rows]; nl = n[ir] - 1; cl = c[ir] - yl; el = ex[ir] - pc[np.minimum(pos[rows], 300).astype(np.int64)]
p0 = float(y.mean())
ctr = np.log((cl + 20 * p0) / (nl + 20)); coec = np.log((cl + 5) / (el + 5)); lgn = np.log1p(nl)
print(f'holdout rows {len(rows)}; LOO coverage n>=1 {np.mean(nl>=1):.3f} n>=5 {np.mean(nl>=5):.3f} n>=20 {np.mean(nl>=20):.3f}')
t = zh(np.asarray(np.load(T / 'teacher_mean11.npy', mmap_mode='r')[rows], np.float32)); e0 = eh(t)
F = np.stack([t, zh(ctr), zh(coec), zh(lgn)], 1)
print(f'alone: teacher {e0.mean():.5f} | LOO ctr {eh(ctr).mean():.5f} | LOO coec {eh(coec).mean():.5f} | log n {eh(lgn).mean():.5f}')
fid = np.repeat(np.arange(len(Mh)), Mh); half = (uid[hold] * 2654435761 % 4294967296) % 2; rh = half[fid]
def fit(mask, cols, iters=200, lr=0.3):
    X = F[mask][:, cols]; f_ = fid[mask]; yy = yl[mask]; b = np.zeros(len(cols)); b[0] = 1.0
    uf, fi = np.unique(f_, return_inverse=True); ypos = np.bincount(fi, weights=yy)
    for _ in range(iters):
        v = X @ b; v = v - np.maximum.reduceat(v, np.r_[0, np.where(np.diff(fi))[0] + 1])[fi]
        ev_ = np.exp(v); p = ev_ / np.bincount(fi, weights=ev_)[fi]
        g = X.T @ (yy - p * ypos[fi]) / len(uf); b += lr * g
    return b
for nm, cols in (('teacher + LOO ctr', [0, 1]), ('teacher + LOO coec', [0, 2]), ('teacher + ctr + coec + log n', [0, 1, 2, 3]), ('teacher + log n (popularity control)', [0, 3])):
    s = np.zeros(len(t)); bs = []
    for h_ in (0, 1):
        b = fit(rh != h_, cols); s[rh == h_] = F[rh == h_][:, cols] @ b; bs.append(np.round(b, 3).tolist())
    print(f'{nm:38s} holdout NDCG {eh(s).mean():.5f}  delta {f(eh(s) - e0)}  coef {bs}')
