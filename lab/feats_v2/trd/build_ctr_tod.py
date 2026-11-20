"""trd_cctr_tod: category x time-of-day x day-type CTR lift from train labels.

lift(c, b, d) = log( ((pos(c,b,d) + k*ctr(c)) / (imp(c,b,d) + k)) / ctr(c) ),  b = 3-hour MSK daypart (8), d = weekday/weekend,
k = 2000 impressions. Train labels are position-biased, but the bias is roughly constant across dayparts within a category,
so the ratio is usable. For train rows only strictly earlier train days are used (expanding window); val/test use all train.
Also trd_cctr_h: same with weekday/weekend pooled (hour-of-day only; val is weekday-only so the d-split is untestable on val).
"""
import sys, time
sys.path.insert(0, "<repo>/feats_v2/trd")
from trd_lib import *

K = 2000.0


def keys(ts):
    ts = ts.astype(np.int64)
    b = (ts + MSK) // 3600 % 24 // 3
    wk = ((((ts + MSK) // 86400 + 3) % 7) >= 5).astype(np.int64)
    day = (ts + MSK) // 86400
    return b, wk, day


def lift(pos, imp, c_pos, c_imp):
    ctr_c = (c_pos + 1) / (c_imp + 60)
    return np.log(((pos + K * ctr_c) / (imp + K)) / ctr_c)


def main():
    t0 = time.time()
    f = feeds_df("train")
    b_f, wk_f, day_f = keys(f["timestamp"].to_numpy())
    fidx = np.load(FLAT / "train/cand_fidx.npy")
    c = np.load(FLAT / "train/cand_category_id.npy").astype(np.int64)
    y = np.load(FLAT / "train/cand_target.npy").astype(np.float64)
    key = (c * 8 + b_f[fidx]) * 2 + wk_f[fidx]          # 54*8*2 = 864
    keyh = c * 8 + b_f[fidx]
    day = day_f[fidx]
    days = np.unique(day)
    NK = 864
    out = np.empty(len(c), np.float32); outh = np.empty(len(c), np.float32)
    pos = np.zeros(NK); imp = np.zeros(NK)
    order = np.argsort(day, kind="stable"); bnd = np.r_[np.searchsorted(day[order], days), len(order)]
    for i, d in enumerate(days):
        r = order[bnd[i]:bnd[i + 1]]
        cp = pos.reshape(54, 8, 2).sum((1, 2)); ci = imp.reshape(54, 8, 2).sum((1, 2))
        ph = pos.reshape(54, 8, 2).sum(2).ravel(); ih = imp.reshape(54, 8, 2).sum(2).ravel()
        out[r] = lift(pos[key[r]], imp[key[r]], cp[c[r]], ci[c[r]])
        outh[r] = lift(ph[keyh[r]], ih[keyh[r]], cp[c[r]], ci[c[r]])
        pos += np.bincount(key[r], weights=y[r], minlength=NK); imp += np.bincount(key[r], minlength=NK)
    np.save(FEATS / "train/trd_cctr_tod.npy", out); np.save(FEATS / "train/trd_cctr_h.npy", outh)
    cp = pos.reshape(54, 8, 2).sum((1, 2)); ci = imp.reshape(54, 8, 2).sum((1, 2))
    ph = pos.reshape(54, 8, 2).sum(2).ravel(); ih = imp.reshape(54, 8, 2).sum(2).ravel()
    for s in ["val", "test"]:
        fs = feeds_df(s); bs, ws, _ = keys(fs["timestamp"].to_numpy())
        fi = np.load(FLAT / s / "cand_fidx.npy"); cs = np.load(FLAT / s / "cand_category_id.npy").astype(np.int64)
        k2 = (cs * 8 + bs[fi]) * 2 + ws[fi]; kh = cs * 8 + bs[fi]
        np.save(FEATS / s / "trd_cctr_tod.npy", lift(pos[k2], imp[k2], cp[cs], ci[cs]).astype(np.float32))
        np.save(FEATS / s / "trd_cctr_h.npy", lift(ph[kh], ih[kh], cp[cs], ci[cs]).astype(np.float32))
    print("done", f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
