"""How much is "exploration intent" worth?  novel = candidate category absent from the 32 newest history events.
 (a) ORACLE feed type: feeds whose positives are all novel get +c on the z-score of their novel candidates (c on a grid,
     in-sample best; an upper bound for any feed-level exploration detector).
 (b) ORACLE feed type, but also within novel candidates the order is the model's -> shows the within-novel ordering limit.
 (c) REALISTIC: + c * logit(p_explore) * novel, p_explore = out-of-fold HGB prediction from feed-level features (04_segments),
     c picked by nested 5-fold CV.  Leak-free feeds.
"""
import os
os.environ.setdefault("NUMBA_NUM_THREADS", "16")
import sys
sys.path.insert(0, "<repo>/src")
from common import *
from stack_val import feed_rank_z
from sklearn.model_selection import GroupKFold

D = Path("<repo>/eda/errors/missed_positives")


def main():
    fr = np.load(D / "frame.npz")
    y, best, M, off, leakf, fid = (fr[k] for k in ["y", "best", "M", "off", "leakf", "fid"])
    nf = len(M); kf = ~leakf
    zb = feed_rank_z(off, M, best.astype(np.float32))[:, 1].astype(np.float64)
    base = ndcg_per_feed(zb, y, off, M)
    cmin = np.load(D / "derived/val/cat_min_hpos.npy"); ccnt = np.load(D / "derived/val/cat_cnt_all.npy")
    nov = ((ccnt == 0) | (cmin >= 32)).astype(np.float64)
    pos = y > 0
    allnov = np.bincount(fid, weights=pos * nov, minlength=nf) == np.bincount(fid, weights=pos.astype(float), minlength=nf)
    print(f"base {base[kf].mean():.5f}; explore feeds {allnov[kf].mean():.3f}")
    for c in [0.5, 1, 2, 3, 5, 10]:
        s = zb + c * nov * allnov[fid]
        per = ndcg_per_feed(s, y, off, M)
        d = (per - base)[kf]
        print(f"ORACLE feed type, +{c} on novel: {per[kf].mean():.5f} (d {d.mean():+.5f})")
    # oracle of the reverse type too: in non-explore feeds, push novel down
    for c in [1, 3, 10]:
        s = zb + c * nov * allnov[fid] - c * nov * (~allnov)[fid]
        per = ndcg_per_feed(s, y, off, M)
        print(f"ORACLE both directions ±{c}: {per[kf].mean():.5f} (d {(per - base)[kf].mean():+.5f})")
    ep = np.load(D / "explore_pred.npy")
    pex = np.full(nf, np.nan); pex[ep[:, 0].astype(int)] = ep[:, 1]
    pex = np.where(np.isfinite(pex), pex, np.nanmean(pex))
    lg = np.log(pex / (1 - pex)) - np.log(np.nanmean(pex) / (1 - np.nanmean(pex)))
    grid = np.array([-0.3, -0.2, -0.1, -0.05, 0, 0.05, 0.1, 0.2, 0.3, 0.5])
    pers = np.stack([ndcg_per_feed(zb + g * lg[fid] * nov, y, off, M) for g in grid])
    f = pl.read_parquet(FLAT / "val/feeds.parquet")
    users = f["user_id"].to_numpy().copy(); users[users == 0] = -np.arange(1, (users == 0).sum() + 1)
    out = np.empty(nf); lams = []
    for tr, va in GroupKFold(n_splits=5, shuffle=True, random_state=0).split(np.arange(nf), groups=users):
        trk = tr[kf[tr]]; l = int(np.argmax(pers[:, trk].mean(1))); lams.append(grid[l]); out[va] = pers[l, va]
    d = (out - base)[kf]
    print(f"REALISTIC (OOF p_explore x novel), nested: d {d.mean():+.5f} ± {d.std() / np.sqrt(len(d)):.5f}, lams {lams}; "
          f"in-sample best {pers[:, kf].mean(1).max() - base[kf].mean():+.5f}")


if __name__ == "__main__":
    main()
