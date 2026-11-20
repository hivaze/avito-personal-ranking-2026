"""C12 Metric-optimal post-processing key B, residual form (PREDECLARED_ccb.txt C12), from a copy of
eda/errors/feed_structure/05_kappa_oracles.py (k05_kappa_oracles_copy.py): p = logistic calibration of b on [b, log M]
(LogisticRegression C=100, as 05), lift(cos bin, samecat) table (EDGES and shrinkage of 05) from within-feed ordered pairs,
both fitted on the training folds (all val for the holdout); EN_i = 1 + sum_{j != i} min(1, p_j lift_ij);
kappaB = log(1/IDCG(EN)) - log(1/EN), interpolated on N = 1..301. PRIMARY: kappaB (1 column). INFO: key log p + kappaB."""
import harness as Hn
from harness import np, log, ROOT, FLAT
import json
import numba as nb
from sklearn.linear_model import LogisticRegression

EDGES = np.array([-1.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 0.99, 0.999, 1.01], np.float32)
NBIN = len(EDGES) - 1
Ns = np.arange(1, 302)
idcg = np.cumsum(1 / np.log2(Ns + 1))


def load_emb(split, emb_idx):
    meta = json.load(open(ROOT / f"data/competition/{split}/item_embeddings/data.meta"))
    E = np.memmap(ROOT / f"data/competition/{split}/item_embeddings/data.memmap", dtype=np.int8, mode="r", shape=tuple(meta["shape"]))
    miss = emb_idx < 0
    ii = np.where(miss, 0, emb_idx)
    o = np.argsort(ii, kind="stable")
    X = np.empty((len(ii), 128), np.float32)
    X[o] = np.asarray(E[ii[o]], dtype=np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
    X[miss] = 0.0
    return X, miss


@nb.njit(parallel=True, cache=True)
def pair_H(X, miss, off, M, t, cat, edges):
    n = len(M)
    nb_ = len(edges) - 1
    H = np.zeros((n, 2, 2, nb_, 2))
    for f in nb.prange(n):
        o = off[f]
        m = M[f]
        Xf = X[o:o + m]
        C = Xf @ Xf.T
        for i in range(m):
            for j in range(m):
                if i == j:
                    continue
                if miss[o + i] or miss[o + j]:
                    continue
                sc = 1 if cat[o + i] == cat[o + j] else 0
                b = np.searchsorted(edges, C[i, j], side="right") - 1
                b = min(max(b, 0), nb_ - 1)
                H[f, t[o + i], sc, b, t[o + j]] += 1
    return H


@nb.njit(parallel=True, cache=True)
def expN(X, miss, off, M, p, cat, edges, L):  # copy of 05_kappa_oracles.expN
    out = np.ones(len(p))
    nb_ = len(edges) - 1
    for f in nb.prange(len(M)):
        o = off[f]
        m = M[f]
        Xf = X[o:o + m]
        C = Xf @ Xf.T
        for i in range(m):
            e = 1.0
            for j in range(m):
                if j == i:
                    continue
                if miss[o + i] or miss[o + j]:
                    lf = 1.0
                else:
                    b = np.searchsorted(edges, C[i, j], side="right") - 1
                    b = min(max(b, 0), nb_ - 1)
                    lf = L[1 if cat[o + i] == cat[o + j] else 0, b]
                e += min(1.0, p[o + j] * lf)
            out[o + i] = e
    return out


def lift_table(H):  # copy of 05_kappa_oracles.lift_table
    L = np.ones((2, H.shape[2]))
    for sc in range(2):
        for b in range(H.shape[2]):
            n1 = H[1, sc, b].sum()
            k1 = H[1, sc, b, 1]
            na = H[:, sc, b].sum()
            ka = H[:, sc, b, 1].sum()
            pa = ka / max(na, 1)
            if pa <= 0:
                continue
            p1 = (k1 + 200 * pa) / (n1 + 200)
            L[sc, b] = p1 / pa
    return L


nb.set_num_threads(12)
yv_i = Hn.yv.astype(np.int64)
Xv, missv = load_emb("val", np.load(FLAT / "val/cand_emb_idx.npy"))
catv = np.load(FLAT / "val/cand_category_id.npy").astype(np.int64)
Hv = pair_H(Xv, missv, Hn.ov, Hn.Mv, yv_i, catv, EDGES)
logMv = np.log(Hn.Mv[Hn.fid])
log("pair H built (all val feeds)")


def fit_cal(rows_mask):
    lr = LogisticRegression(C=100.0, max_iter=1000).fit(np.column_stack([Hn.b[rows_mask], logMv[rows_mask]]), yv_i[rows_mask])
    return lr


def kappa(lr, L, b_, logM, X, miss, off, M, cat):
    p = lr.predict_proba(np.column_stack([b_, logM]))[:, 1]
    EN = expN(X, miss, off, M, p, cat, EDGES, L)
    kB = np.log(np.interp(EN, Ns, 1 / idcg)) - np.log(1 / EN)
    return kB, np.log(p), EN


FOLD = {}
for k in range(5):
    tr = Hn.rowfold != k
    trf = np.ones(Hn.nF, bool)
    trf[Hn.folds[k][1]] = False
    lr = fit_cal(tr)
    L = lift_table(Hv[trf].sum(0))
    kB, lp, EN = kappa(lr, L, Hn.b, logMv, Xv, missv, Hn.ov, Hn.Mv, catv)
    FOLD[k] = (kB, lp)
    log(f"fold {k}: lr coef {lr.coef_.ravel().round(3)} int {lr.intercept_.round(3)}; lift samecat=1 {L[1].round(2)}; "
        f"EN median {np.median(EN):.3f} q90 {np.quantile(EN, 0.9):.3f}")
lr_full = fit_cal(np.ones(len(Hn.yv), bool))
L_full = lift_table(Hv.sum(0))
kB_v, lp_v, EN_v = kappa(lr_full, L_full, Hn.b, logMv, Xv, missv, Hn.ov, Hn.Mv, catv)
log(f"full: lift samecat=1 {L_full[1].round(2)} samecat=0 {L_full[0].round(2)}")
del Xv
# holdout columns: embeddings / categories of the holdout rows from data/flat/train (train memmap)
eh = np.load(FLAT / "train/cand_emb_idx.npy", mmap_mode="r")[Hn.HOLD_ROWS]
Xh, missh = load_emb("train", np.asarray(eh))
cath = np.asarray(np.load(FLAT / "train/cand_category_id.npy", mmap_mode="r")[Hn.HOLD_ROWS]).astype(np.int64)
logMh = np.log(Hn.Mh[Hn.fidh])
kB_h, lp_h, EN_h = kappa(lr_full, L_full, Hn.bh, logMh, Xh, missh, Hn.oh, Hn.Mh, cath)
log(f"holdout EN median {np.median(EN_h):.3f}; emb miss val {missv.mean():.4f} hold {missh.mean():.4f}")
np.savez(Hn.CACHE / "c12_cols.npz", kB_v=kB_v, kB_h=kB_h, **{f"kB_f{k}": FOLD[k][0] for k in range(5)})

# INFO: deterministic key log p + kappaB (nested: fold-k fit applied to fold k) vs control and vs b
c = Hn.control()
key = np.empty(len(Hn.yv))
for k in range(5):
    m = Hn.rowfold == k
    key[m] = (FOLD[k][1] + FOLD[k][0])[m]
dk = Hn.ndv(key) - c["cv"]
dk_b = Hn.ndv(key) - Hn.ndv(Hn.b)
dkh = Hn.ndh(lp_h + kB_h) - Hn.BASE_H_PF
info = dict(key_nested_vs_control=[float(dk.mean()), Hn.se(dk)], key_nested_vs_b=[float(dk_b.mean()), Hn.se(dk_b)],
            key_hold=[float(dkh.mean()), Hn.se(dkh)], key_hold_late=float(dkh[Hn.H_LATE].mean()),
            key_hold_early=float(dkh[Hn.H_EARLY].mean()))
log(f"INFO deterministic key log p + kappaB: nested val vs control {dk.mean():+.5f} ({Hn.se(dk):.5f}) vs b {dk_b.mean():+.5f}; "
    f"holdout {dkh.mean():+.5f} ({Hn.se(dkh):.5f}) late {dkh[Hn.H_LATE].mean():+.5f} early {dkh[Hn.H_EARLY].mean():+.5f}")
Hn.run_item("C12", ["kappaB"], [kB_v], [kB_h], xv_fold=lambda k: [FOLD[k][0]], extra=info)
