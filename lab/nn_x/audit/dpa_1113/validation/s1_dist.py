"""S1: distributions of val / holdout / public / private (+ the 134K random-order train pool) on label-free feed
descriptors, with PSI (reference = public, and public vs private) and SMD. Writes out/desc_<set>.parquet + out/s1_dist.log.
  CUDA_VISIBLE_DEVICES="" .venv/bin/python <scratch>/dpa/validation/s1_dist.py
"""
import time
from lib import *  # noqa: F401,F403
from lib import _HOLD

t0 = time.time()
LOG = open(OUT / "s1_dist.log", "w")


def pr(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True); LOG.write(s + "\n"); LOG.flush()


D0 = 20702  # first train day (MSK); train days 20702..20716


def sets():
    """name -> (split, feed indices)"""
    fv, Mv, _ = meta("val"); ft, Mt, _ = meta("test")
    pub = public_mask()
    hold = holdout_index()
    return {"val": ("val", np.arange(len(Mv))), "holdout": ("train", hold), "pool134": ("train", _HOLD["pool134"]),
            "public": ("test", np.where(pub)[0]), "private": ("test", np.where(~pub)[0])}


def cand_rows(split, fidx):
    f, M, off = meta(split)
    m = M[fidx]; o = off[fidx]
    rows = np.repeat(o, m) + (np.arange(m.sum()) - np.repeat(np.cumsum(m) - m, m))
    return rows, m


def main():
    S = sets()
    # ---- train-candidate day bitmaps for every item that occurs in any evaluated set
    ftr, Mtr, offtr = meta("train")
    dtr_feed = day(ftr["timestamp"].to_numpy())
    rows_of = {k: cand_rows(sp, fi) for k, (sp, fi) in S.items()}
    items_of = {k: np.load(FLAT / f"{S[k][0]}/cand_item_id.npy", mmap_mode="r")[rows_of[k][0]].astype(np.int64) for k in S}
    Q = np.unique(np.concatenate(list(items_of.values())))
    pr(f"query items {len(Q):,}  ({time.time() - t0:.0f}s)")
    tr_item = np.load(FLAT / "train/cand_item_id.npy", mmap_mode="r")
    bits = np.zeros(len(Q), np.int32)
    for dd in np.unique(dtr_feed):
        fi = np.where(dtr_feed == dd)[0]
        lo, hi = offtr[fi[0]], offtr[fi[-1]] + Mtr[fi[-1]]
        assert (np.diff(fi) == 1).all()  # train feeds are time-sorted
        it = np.unique(np.asarray(tr_item[lo:hi], np.int64))
        j = np.searchsorted(Q, it); j = np.minimum(j, len(Q) - 1); hit = Q[j] == it
        bits[j[hit]] |= (1 << int(dd - D0))
    pr(f"bitmaps done ({time.time() - t0:.0f}s)")
    # ---- user train-feed days (for 'user seen in train before the feed day')
    utr = ftr["user_id"].to_numpy().astype(np.int64)
    ukey = utr * 64 + (dtr_feed - D0)
    ukey = np.unique(ukey[utr > 0])
    ufirst = {}
    uu, first_idx = np.unique(ukey // 64, return_index=True)
    ufirst_day = (ukey[first_idx] % 64) + D0  # first train day per user
    ucnt_u, ucnt = np.unique(utr[utr > 0], return_counts=True)

    out = {}
    for k, (sp, fi) in S.items():
        f, M, off = meta(sp)
        rows, m = rows_of[k]
        n = len(fi); fid = np.repeat(np.arange(n), m)
        ts = f["timestamp"].to_numpy().astype(np.int64)[fi]
        dfe = day(ts); dc = dfe[fid]
        it = items_of[k]
        b = bits[np.searchsorted(Q, it)]
        # causal for train rows: only days strictly before the feed day; val/test: all train days
        cut = np.clip(dc - D0, 0, 30)
        mask = np.where(cut >= 15, b, b & ((1 << cut.astype(np.int64)) - 1).astype(np.int32))
        has = mask > 0
        last = np.full(len(it), -1, np.int64)
        last[has] = np.floor(np.log2(mask[has].astype(np.float64))).astype(np.int64) + D0
        stale = np.where(has, dc - last, np.nan)
        tte = np.load(FEATS / f"{sp}/tte_it_imp.npy", mmap_mode="r")[rows].astype(np.float64)
        luo = np.load(ROOT / f"nn/feat_cache/{sp}/it_hist_all_luo.npy", mmap_mode="r")[rows].astype(np.float64)
        cat = np.load(FLAT / f"{sp}/cand_category_id.npy", mmap_mode="r")[rows].astype(np.int64)
        f1 = np.load(FLAT / f"{sp}/cand_f1.npy", mmap_mode="r")[rows].astype(np.float64)
        f4 = np.load(FLAT / f"{sp}/cand_f4.npy", mmap_mode="r")[rows].astype(np.float64)
        cnt = lambda x: np.bincount(fid, x, n)  # noqa: E731
        stale_c = np.where(np.isnan(stale), 0, np.minimum(stale, 20))
        d = {
            "M": M[fi].astype(np.float64), "H": f["H"].to_numpy()[fi].astype(np.float64),
            "uid0": (f["user_id"].to_numpy()[fi] == 0).astype(np.float64),
            "cold_tte": cnt((tte == 0).astype(np.float64)) / m,         # never a train candidate before the feed day
            "cold_bits": cnt((~has).astype(np.float64)) / m,            # same from the bitmap (cross-check)
            "luo_zero": cnt((luo == 0).astype(np.float64)) / m,         # model popularity input == 0
            "stale_mean": cnt(stale_c * has) / np.maximum(cnt(has.astype(np.float64)), 1),  # days since last train appearance (warm items)
            "stale_min_warm": None,
            "f1_mean": cnt(f1) / m, "f4_mean": cnt(f4) / m,
            "hour": hour(ts).astype(np.float64), "dow": weekday(ts).astype(np.float64), "day": dfe.astype(np.float64),
        }
        # category: per-feed entropy and top share; global category mix
        key = fid * 64 + cat
        uk, c = np.unique(key, return_counts=True)
        ff = uk // 64; p = c / m[ff]
        d["cat_entropy"] = np.bincount(ff, -p * np.log(p), n)
        d["n_cat"] = np.bincount(ff, minlength=n).astype(np.float64)
        d["top_cat_share"] = np.maximum.reduceat(p, np.r_[0, np.where(np.diff(ff) > 0)[0] + 1]) if len(p) else np.zeros(n)
        d.pop("stale_min_warm")
        # user seen in train strictly before the feed day; n train feeds
        u = f["user_id"].to_numpy().astype(np.int64)[fi]
        j = np.searchsorted(uu, u); j = np.minimum(j, len(uu) - 1); found = (uu[j] == u) & (u > 0)
        d["user_in_train_before"] = (found & (ufirst_day[j] < dfe)).astype(np.float64)
        j2 = np.searchsorted(ucnt_u, u); j2 = np.minimum(j2, len(ucnt_u) - 1)
        d["n_train_feeds_all"] = np.where((ucnt_u[j2] == u) & (u > 0), ucnt[j2], 0).astype(np.float64)
        if sp in ("val", "train"):
            y = np.load(FLAT / f"{sp}/cand_target.npy", mmap_mode="r")[rows].astype(np.float64)
            d["npos"] = cnt(y)
            d["pos_cold_share"] = cnt(y * (tte == 0)) / np.maximum(cnt(y), 1)
        out[k] = d
        catmix = np.bincount(cat, minlength=64).astype(np.float64)
        d["_catmix"] = catmix / catmix.sum()
        pl.DataFrame({kk: v for kk, v in d.items() if not kk.startswith("_")}).write_parquet(OUT / f"desc_{k}.parquet")
        np.save(OUT / f"catmix_{k}.npy", d["_catmix"])
        pr(f"{k}: n {n:,}, cands {len(rows):,} ({time.time() - t0:.0f}s)")

    # ---- tables
    cols = ["M", "H", "uid0", "user_in_train_before", "n_train_feeds_all", "cold_tte", "cold_bits", "luo_zero", "stale_mean",
            "f1_mean", "f4_mean", "cat_entropy", "n_cat", "top_cat_share", "hour", "npos", "pos_cold_share"]
    names = ["val", "holdout", "pool134", "public", "private"]
    pr("\n== means (sd) per set ==")
    pr(f"{'descriptor':22s}" + "".join(f"{n:>18s}" for n in names))
    for c in cols:
        row = f"{c:22s}"
        for nme in names:
            x = out[nme].get(c)
            row += f"{'-':>18s}" if x is None else f"{x.mean():>10.4f} ({x.std():.3f})"
        pr(row)
    pr(f"{'M<50 share':22s}" + "".join(f"{(out[n]['M'] < 50).mean():>18.4f}" for n in names))
    pr(f"{'weekend share':22s}" + "".join(f"{(out[n]['dow'] >= 5).mean():>18.4f}" for n in names))
    pr(f"{'hour 14-24 share':22s}" + "".join(f"{(out[n]['hour'] >= 14).mean():>18.4f}" for n in names))
    pr(f"{'cold_tte>=0.45 share':22s}" + "".join(f"{(out[n]['cold_tte'] >= 0.45).mean():>18.4f}" for n in names))
    pr(f"{'cold_tte>0.3 share':22s}" + "".join(f"{(out[n]['cold_tte'] > 0.3).mean():>18.4f}" for n in names))
    pr("\n== PSI vs public (quantile bins of public; 10 bins) | PSI public vs private | SMD vs public ==")
    pr(f"{'descriptor':22s}" + "".join(f"{'PSI ' + n:>14s}" for n in ["val", "holdout", "pool134", "private"]) +
       "".join(f"{'SMD ' + n:>13s}" for n in ["val", "holdout", "private"]))
    for c in ["M", "H", "uid0", "user_in_train_before", "n_train_feeds_all", "cold_tte", "luo_zero", "stale_mean", "f1_mean", "f4_mean",
              "cat_entropy", "n_cat", "top_cat_share", "hour", "dow"]:
        ref = out["public"][c]
        disc = c in ("uid0", "user_in_train_before", "hour", "dow")
        row = f"{c:22s}"
        for n in ["val", "holdout", "pool134", "private"]:
            if disc:
                K = int(max(ref.max(), out[n][c].max())) + 1
                row += f"{psi_cat(np.bincount(out[n][c].astype(int), minlength=K) / len(out[n][c]), np.bincount(ref.astype(int), minlength=K) / len(ref)):>14.4f}"
            else:
                row += f"{psi(out[n][c], ref):>14.4f}"
        for n in ["val", "holdout", "private"]:
            row += f"{smd(out[n][c], ref):>13.3f}"
        pr(row)
    pr("\n== candidate category mix: PSI vs public (53 categories) ==")
    for n in ["val", "holdout", "pool134", "private"]:
        pr(f"  {n:8s} PSI {psi_cat(out[n]['_catmix'], out['public']['_catmix']):.4f}")
    pr("\n== item staleness distribution of WARM candidates (days since last train-candidate day), share by bucket ==")
    for n in names:
        x = out[n]["stale_mean"]
        pr(f"  {n:8s} feed-mean staleness: p10 {np.quantile(x, .1):.2f} p50 {np.median(x):.2f} p90 {np.quantile(x, .9):.2f}")
    pr("\n== holdout / pool134 by train day: n, cold_tte, luo_zero, stale_mean ==")
    for n in ["holdout", "pool134"]:
        dd = out[n]["day"]
        for u in np.unique(dd):
            mm = dd == u
            pr(f"  {n:8s} day {int(u)} n {mm.sum():6d} cold_tte {out[n]['cold_tte'][mm].mean():.3f} luo_zero {out[n]['luo_zero'][mm].mean():.3f} "
               f"stale {out[n]['stale_mean'][mm].mean():.2f} weekend {(out[n]['dow'][mm] >= 5).mean():.0f}")
    pr("\n== val / test by day ==")
    for n in ["val", "public", "private"]:
        dd = out[n]["day"]
        for u in np.unique(dd):
            mm = dd == u
            pr(f"  {n:8s} day {int(u)} n {mm.sum():6d} cold_tte {out[n]['cold_tte'][mm].mean():.3f} luo_zero {out[n]['luo_zero'][mm].mean():.3f} stale {out[n]['stale_mean'][mm].mean():.2f}")
    pr(f"done {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
