"""User's own historical exploration rate (own history only, leak-free): among event_type-1 events at hpos j with at least
32 older events, the share whose category does not occur in the 32 older events (j+1..j+32).  Also the rate restricted to
the 30 newest such events (recent exploration mood).  Does it predict exploration feeds beyond the novel-candidate share?
"""
import os
os.environ.setdefault("NUMBA_NUM_THREADS", "16")
import sys
sys.path.insert(0, "<repo>/src")
from common import *
import numba as nb
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict

D = Path("<repo>/eda/errors/missed_positives")


@nb.njit(parallel=True, cache=True)
def kern(hoff, H, hcat, het, hpos, out):
    for f in nb.prange(len(H)):
        ho, h = hoff[f], H[f]
        order = np.argsort(hpos[ho:ho + h])
        cats = hcat[ho:ho + h][order]; ets = het[ho:ho + h][order]
        n = 0; k = 0; nr = 0; kr = 0
        for j in range(h - 32):
            if ets[j] != 1:
                continue
            c = cats[j]; novel = True
            for t in range(j + 1, j + 33):
                if cats[t] == c:
                    novel = False; break
            n += 1; k += novel
            if nr < 30:
                nr += 1; kr += novel
        out[f, 0] = k / n if n > 0 else np.nan
        out[f, 1] = kr / nr if nr > 0 else np.nan
        out[f, 2] = n


def main():
    for split in ["val", "test"]:
        f, M, off = feed_meta(split)
        d = FLAT / split
        out = np.zeros((len(M), 3))
        kern(f["hist_off"].to_numpy().astype(np.int64), f["H"].to_numpy().astype(np.int64), np.load(d / "hist_category_id.npy").astype(np.int64),
             np.load(d / "hist_event_type.npy").astype(np.int64), np.load(d / "hist_hpos.npy").astype(np.int64), out)
        np.save(D / "derived" / split / "d_u_explore_rate.npy", np.repeat(out[:, 0], M).astype(np.float32))
        np.save(D / "derived" / split / "d_u_explore_rate30.npy", np.repeat(out[:, 1], M).astype(np.float32))
        if split == "val":
            outv, Mv, offv, fv = out, M, off, f
    fr = np.load(D / "frame.npz")
    y, leakf, fid = fr["y"], fr["leakf"], fr["fid"]
    nf = len(Mv); kf = ~leakf
    cmin = np.load(D / "derived/val/cat_min_hpos.npy"); ccnt = np.load(D / "derived/val/cat_cnt_all.npy")
    nov = ((ccnt == 0) | (cmin >= 32)).astype(float); pos = y > 0
    allnov = np.bincount(fid, weights=pos * nov, minlength=nf) == np.bincount(fid, weights=pos.astype(float), minlength=nf)
    novshare = np.bincount(fid, weights=nov, minlength=nf) / Mv
    ok = kf & np.isfinite(outv[:, 0])
    print(f"feeds with a defined rate: {ok.mean():.3f}; mean rate {np.nanmean(outv[ok, 0]):.3f}, recent {np.nanmean(outv[ok, 1]):.3f}")
    for nm, x in [("hist explore rate", outv[:, 0]), ("recent explore rate (30)", outv[:, 1]), ("novel cand share", novshare)]:
        print(f"AUC(explore feed) {nm}: {roc_auc_score(allnov[ok], x[ok]):.3f}")
    users = fv["user_id"].to_numpy().copy(); users[users == 0] = -np.arange(1, (users == 0).sum() + 1)
    lg = lambda p: np.log(np.clip(p, 1e-3, 1 - 1e-3) / (1 - np.clip(p, 1e-3, 1 - 1e-3)))
    X1 = np.c_[lg(novshare[ok])]; X2 = np.c_[lg(novshare[ok]), lg(outv[ok, 0]), lg(outv[ok, 1])]
    for nm, X in [("novshare", X1), ("novshare + explore rates", X2)]:
        pr = cross_val_predict(LogisticRegression(), X, allnov[ok], groups=users[ok], cv=GroupKFold(5), method="predict_proba")[:, 1]
        print(f"CV logistic AUC {nm}: {roc_auc_score(allnov[ok], pr):.3f}")


if __name__ == "__main__":
    main()
