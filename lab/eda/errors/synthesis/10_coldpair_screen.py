"""Which existing val/test features carry signal on the weakest pair type (cold positive vs warm negative) beyond nn_final3_trainonly?
For each feature: pairwise AUC on cold+ vs warm- pairs (all / same category) and residual t (feed-clustered) next to dz(base).
Also the same for the fine-tuned OOF minus train-only score difference (what the val fine-tune adds on these pairs)."""
import sys
sys.path.insert(0, "<repo>/src"); sys.path.insert(0, "<repo>/eda/errors/synthesis")
from common import *
from stack_val import feed_rank_z
from qt import leak_mask, path_of, pp
import importlib.util
spec = importlib.util.spec_from_file_location("scr", "<repo>/eda/errors/synthesis/02_screen.py")

D = Path("<repo>/eda/errors/synthesis")
f, M, off = feed_meta("val"); y = np.load(FLAT / "val/cand_target.npy"); kf = ~leak_mask()
cat = np.load(FLAT / "val/cand_category_id.npy"); cold = np.load(D / "derived/val/cold_item.npy").astype(bool)
zb = feed_rank_z(off, M, np.load(pp("nn_final3_trainonly_val.npy")).astype(np.float32))[:, 1].astype(np.float64)
zo = feed_rank_z(off, M, np.load(pp("nn_final3_val_oof.npy")).astype(np.float32))[:, 1].astype(np.float64)
pi, ni, pf = [], [], []
for g in np.nonzero(kf)[0]:
    s, m = off[g], M[g]; yy = y[s:s + m]
    p = np.nonzero((yy == 1) & cold[s:s + m])[0] + s; q = np.nonzero((yy == 0) & ~cold[s:s + m])[0] + s
    if len(p) and len(q):
        pi.append(np.repeat(p, len(q))); ni.append(np.tile(q, len(p))); pf.append(np.full(len(p) * len(q), g))
pi = np.concatenate(pi); ni = np.concatenate(ni); pf = np.concatenate(pf); same = cat[pi] == cat[ni]
print(f"cold+ vs warm- pairs {len(pi):,} in {len(np.unique(pf)):,} feeds; same-category {same.mean():.3f}")


def zfeed(x):
    zx = np.zeros(len(x))
    for g in range(len(M)):
        s, m = off[g], M[g]; v = x[s:s + m]; ok = np.isfinite(v)
        if ok.sum() >= 2 and v[ok].std() > 0:
            zx[s:s + m][ok] = (v[ok] - v[ok].mean()) / v[ok].std()
    return zx


def clogit(X, cl):
    Xs = np.vstack([X, -X]); ys = np.r_[np.ones(len(X)), np.zeros(len(X))]; cls = np.r_[cl, cl]; b = np.zeros(X.shape[1])
    for _ in range(25):
        pr = 1 / (1 + np.exp(-Xs @ b)); b += np.linalg.solve((Xs * (pr * (1 - pr))[:, None]).T @ Xs, Xs.T @ (ys - pr))
    pr = 1 / (1 + np.exp(-Xs @ b)); G = Xs * (ys - pr)[:, None]
    o = np.argsort(cls, kind="stable"); bnd = np.r_[0, np.nonzero(np.diff(cls[o]))[0] + 1]; Gc = np.add.reduceat(G[o], bnd, axis=0)
    Hi = np.linalg.inv((Xs * (pr * (1 - pr))[:, None]).T @ Xs); return b, np.sqrt(np.diag(Hi @ (Gc.T @ Gc) @ Hi))


auc = lambda d: float(((d > 0) + 0.5 * (d == 0)).mean())
print(f"{'base f3_tr':24s} AUC {auc(zb[pi] - zb[ni]):.4f} same-cat {auc((zb[pi] - zb[ni])[same]):.4f}")
print(f"{'f3_oof':24s} AUC {auc(zo[pi] - zo[ni]):.4f} same-cat {auc((zo[pi] - zo[ni])[same]):.4f}")
names = sys.argv[1].split(",")
for n in names:
    x = np.load(path_of(n, "val")).astype(np.float64); zx = zfeed(np.where(np.isfinite(x), x, np.nan))
    d = zx[pi] - zx[ni]
    b, se = clogit(np.c_[zb[pi] - zb[ni], d], pf)
    print(f"{n:24s} AUC {auc(d):.4f} same-cat {auc(d[same]):.4f} | resid t {b[1] / se[1]:+.1f}", flush=True)
