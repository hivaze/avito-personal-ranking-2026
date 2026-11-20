"""Time-aware target encodings from train labels (no own-label leakage).

For train rows on day d, statistics come only from train days < d (expanding window).
For val/test rows, statistics come from all of train. user_id 0 (unknown) gets NaN user stats.
Features (float32 flat, written to data/feats/<split>/):
  tte_it_imp, tte_it_pos, tte_it_ctr, tte_it_coec       item impressions / positives / smoothed CTR / clicks-over-expected (position-debiased)
  tte_u_imp, tte_u_ctr, tte_u_coec                      user-level
  tte_uc_imp, tte_uc_ctr                                user x category
  tte_ui_imp, tte_ui_pos                                same (user, item) shown before in train / positive before
  tte_c_ctr                                             category CTR (time-aware)
"""
import sys, time
sys.path.insert(0, "<repo>/src")
from common import *

PRIOR = 0.0159


def day_of(ts):
    return (ts.astype(np.int64) + 3 * 3600) // 86400


def lookup_sorted(keys_sorted, vals, q):
    """vals aligned to keys_sorted; returns vals at q (0 where missing)."""
    i = np.searchsorted(keys_sorted, q)
    i = np.minimum(i, len(keys_sorted) - 1)
    hit = keys_sorted[i] == q
    return np.where(hit, vals[i], 0.0), hit


def main():
    t0 = time.time()
    ftr, Mtr, _ = feed_meta("train")
    fidx = np.load(FLAT / "train/cand_fidx.npy")
    day_f = day_of(ftr["timestamp"].to_numpy())
    day = day_f[fidx]
    uid = ftr["user_id"].to_numpy().astype(np.int64)[fidx]
    item = np.load(FLAT / "train/cand_item_id.npy").astype(np.int64)
    cat = np.load(FLAT / "train/cand_category_id.npy").astype(np.int64)
    y = np.load(FLAT / "train/cand_target.npy").astype(np.float32)
    pos = np.load(FLAT / "train/cand_pos.npy").astype(np.int64)
    pc = np.bincount(pos, weights=y, minlength=301) / np.maximum(np.bincount(pos, minlength=301), 1)
    ex = pc[pos].astype(np.float32)
    print("loaded", time.time() - t0, flush=True)

    keys = {
        "it": item,
        "u": np.where(uid > 0, uid, -1),
        "uc": np.where(uid > 0, uid * 64 + cat, -1),
        "ui": np.where(uid > 0, uid * 2**27 + item, -1),
        "c": cat,
    }
    days = np.unique(day)
    out_train = {}
    # compact key ids per family over train rows
    for fam, k in keys.items():
        uk, inv = np.unique(k, return_inverse=True)
        n = len(uk)
        imp = np.zeros(n, np.float64); ps = np.zeros(n, np.float64); exs = np.zeros(n, np.float64)
        f_imp = np.empty(len(k), np.float32); f_pos = np.empty(len(k), np.float32); f_ex = np.empty(len(k), np.float32)
        order = np.argsort(day, kind="stable")
        bounds = np.searchsorted(day[order], days)
        bounds = np.r_[bounds, len(order)]
        for di in range(len(days)):
            rows = order[bounds[di]:bounds[di + 1]]
            ii = inv[rows]
            f_imp[rows] = imp[ii]; f_pos[rows] = ps[ii]; f_ex[rows] = exs[ii]
            imp += np.bincount(ii, minlength=n); ps += np.bincount(ii, weights=y[rows], minlength=n)
            exs += np.bincount(ii, weights=ex[rows], minlength=n)
        out_train[fam] = (f_imp, f_pos, f_ex)
        # full-train totals for val/test lookup
        np.savez(FEATS / f"tte_tot_{fam}.npz", keys=uk, imp=imp, pos=ps, ex=exs)
        print(fam, n, time.time() - t0, flush=True)

    def write(split, fam_vals, valid_u):
        d = FEATS / split
        it_imp, it_pos, it_ex = fam_vals["it"]
        np.save(d / "tte_it_imp.npy", it_imp.astype(np.float32))
        np.save(d / "tte_it_pos.npy", it_pos.astype(np.float32))
        np.save(d / "tte_it_ctr.npy", ((it_pos + PRIOR * 20) / (it_imp + 20)).astype(np.float32))
        np.save(d / "tte_it_coec.npy", ((it_pos + 1) / (it_ex + 1)).astype(np.float32))
        u_imp, u_pos, u_ex = fam_vals["u"]
        nanu = lambda a: np.where(valid_u, a, np.nan).astype(np.float32)
        np.save(d / "tte_u_imp.npy", nanu(u_imp))
        np.save(d / "tte_u_ctr.npy", nanu((u_pos + PRIOR * 50) / (u_imp + 50)))
        np.save(d / "tte_u_coec.npy", nanu((u_pos + 1) / (u_ex + 1)))
        uc_imp, uc_pos, uc_ex = fam_vals["uc"]
        np.save(d / "tte_uc_imp.npy", nanu(uc_imp))
        np.save(d / "tte_uc_ctr.npy", nanu((uc_pos + PRIOR * 20) / (uc_imp + 20)))
        ui_imp, ui_pos, _ = fam_vals["ui"]
        np.save(d / "tte_ui_imp.npy", nanu(ui_imp))
        np.save(d / "tte_ui_pos.npy", nanu(ui_pos))
        c_imp, c_pos, _ = fam_vals["c"]
        np.save(d / "tte_c_ctr.npy", ((c_pos + PRIOR * 1000) / (c_imp + 1000)).astype(np.float32))

    write("train", out_train, uid > 0)
    del out_train
    for split in ["val", "test"]:
        f, M, _ = feed_meta(split)
        fi = np.load(FLAT / split / "cand_fidx.npy")
        u = f["user_id"].to_numpy().astype(np.int64)[fi]
        it = np.load(FLAT / split / "cand_item_id.npy").astype(np.int64)
        c = np.load(FLAT / split / "cand_category_id.npy").astype(np.int64)
        q = {"it": it, "u": np.where(u > 0, u, -1), "uc": np.where(u > 0, u * 64 + c, -1),
             "ui": np.where(u > 0, u * 2**27 + it, -1), "c": c}
        vals = {}
        for fam in keys:
            z = np.load(FEATS / f"tte_tot_{fam}.npz")
            a, _ = lookup_sorted(z["keys"], z["imp"], q[fam])
            b, _ = lookup_sorted(z["keys"], z["pos"], q[fam])
            e, _ = lookup_sorted(z["keys"], z["ex"], q[fam])
            vals[fam] = (a, b, e)
        write(split, vals, u > 0)
        print(split, time.time() - t0, flush=True)


if __name__ == "__main__":
    main()
