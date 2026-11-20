"""Cheap NDCG check of single unexploited signals on top of the actual best model (v5 blend), no stacking:
score = z(v5) + lam * zf(x), zf = per-feed z-score of x (NaN -> feed mean), lam picked from a grid by 5-fold GroupKFold(user)
on the training folds and applied to the held-out fold (nested, so the reported delta is honest).  Leak-free feeds only.
"""
import os
os.environ.setdefault("NUMBA_NUM_THREADS", "16")
import sys
sys.path.insert(0, "<repo>/src")
from common import *
from stack_val import feed_rank_z
from sklearn.model_selection import GroupKFold

D = Path("<repo>/eda/errors/missed_positives")
GRID = np.array([-0.3, -0.2, -0.15, -0.1, -0.07, -0.05, -0.03, -0.02, -0.01, 0, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3])
SIG = ["pre_v3_fe", "pre_v3_new", "pre_v3_fe_noid", "pre_v3_id_contrib", "d_cat_pur20", "d_cat_run", "d_top_cat20", "feed_cat_share",
       "f3", "cat_share_all", "clu_ccos1024", "trd_skde_s6h", "trd_kde_sday", "d_f3_resid", "d_dup95_h", "it_hist_all_luo"]


def path_of(n):
    p = FEATS / "val" / f"{n}.npy"
    return p if p.exists() else D / "derived/val" / f"{n}.npy"


def main():
    fr = np.load(D / "frame.npz")
    y, best, M, off, leakf, fid = (fr[k] for k in ["y", "best", "M", "off", "leakf", "fid"])
    nf = len(M); kf = ~leakf
    f = pl.read_parquet(FLAT / "val/feeds.parquet")
    users = f["user_id"].to_numpy().copy(); users[users == 0] = -np.arange(1, (users == 0).sum() + 1)
    zb = feed_rank_z(off, M, best.astype(np.float32))[:, 1].astype(np.float64)
    base = ndcg_per_feed(zb, y, off, M)
    print(f"v5 blend (re-z) leak-free {base[kf].mean():.5f}")
    folds = list(GroupKFold(n_splits=5, shuffle=True, random_state=0).split(np.arange(nf), groups=users))
    for n in SIG:
        x = np.load(path_of(n)).astype(np.float64)
        fm = np.bincount(fid, weights=np.nan_to_num(x), minlength=nf) / np.maximum(np.bincount(fid, weights=np.isfinite(x).astype(float), minlength=nf), 1)
        x = np.where(np.isfinite(x), x, fm[fid]).astype(np.float32)
        zx = feed_rank_z(off, M, x)[:, 1].astype(np.float64)
        pers = np.stack([ndcg_per_feed(zb + l * zx, y, off, M) for l in GRID])  # (grid, feeds)
        out = np.empty(nf); lams = []
        for tr, va in folds:
            trk = tr[kf[tr]]
            l = int(np.argmax(pers[:, trk].mean(1)))
            lams.append(GRID[l]); out[va] = pers[l, va]
        d = (out - base)[kf]
        lbest = GRID[int(np.argmax(pers[:, kf].mean(1)))]
        print(f"{n:20s} nested d {d.mean():+.5f} ± {d.std() / np.sqrt(len(d)):.5f} | fold lams {lams} | in-sample best lam {lbest:+.2f} "
              f"d {pers[:, kf].mean(1).max() - base[kf].mean():+.5f}", flush=True)


if __name__ == "__main__":
    main()
