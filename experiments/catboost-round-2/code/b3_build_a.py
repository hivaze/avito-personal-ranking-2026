# Family B, block (a): hand-made click statistics of a user on the candidate's category and
# cluster, taken only from that user's strictly earlier train feeds (holdout feeds excluded, val,
# test and holdout labels never read), plus similarity of the candidate to items the user clicked
# or ignored before.
# How it was run: `python -u b3_build_a.py` from lab/nn_x/w12/cb/famB (CPU, 2026-11-15).
# What it produced: feat/a_<set>.npy for the training set, the holdout, val and test. With blocks
# b and c it gives the 70 extra columns of the best arm of the round (information gain +0.00147 on
# val and +0.00080 on the holdout against a bar of 0.0016).
# Kept as it ran: paths and names refer to the lab tree (see lab/ and docs/glossary.md).
# Original: lab/nn_x/w12/cb/famB/b3_build_a.py
"""B3 block (a): user x candidate-key label statistics from the user's EARLIER train feeds (hand-made, time-aware, holdout-free).

For a target feed t of user u (u > 0) the SOURCE rows are the candidate rows of u's train feeds that
   - are not one of the 30,000 holdout feeds, and
   - have a strictly EARLIER timestamp than t (128 s resolution) when t is a train feed (xm / hold); val / test targets use all of u's
     non-holdout train feeds (every train feed is earlier than val / test).
The target feed itself is never a source (strict inequality), a later snapshot of the user is never a source, val / test / holdout labels are
never read. Unknown users (u = 0) and users without a source feed get NaN.
Columns (float32, one file per set: feat/a_<set>.npy, names in feat/a_names.json):
  ua_ns, ua_npos                                   number of source rows / clicks (feed-constant)
  ua_{cat,c256,c1024,c4096}_{imp,pos,coec}         impressions, clicks, clicks over expected (position prior from non-holdout train rows; (pos+.5)/(ex+.5))
  ua_{cat,c1024}_{imprw,posrw}                     recency-weighted (half-life 2 days) impressions / clicks
  ua_{cat,c1024}_share                             share of the source rows in the candidate's category / cluster
  ue_maxpos, ue_maxneg, ue_meanpos, ue_meanneg     cosine of the candidate to the user's earlier CLICKED / SHOWN-NOT-CLICKED candidates
  ue_k{05,20}_{all,pos,coec}                       kernel sums exp((cos-1)/tau), tau 0.05 / 0.2: mass, clicked mass, kernel COEC
  ue_top5_pos, ue_top5_cos                         clicks among the 5 most similar source rows, their mean cosine
  ua_item_imp, ua_item_pos, ua_item_dt             the SAME item shown / clicked earlier to this user (128 s resolution, so same-day repeats count), hours since last shown
  ua_dt_last                                       hours since the user's most recent source feed (feed-constant)
  ua_c1024_coec_z, ue_k05_coec_z, ue_k20_coec_z    within-feed z of three of them
usage: python -u b3_build_a.py"""
import json, sys, time
sys.path.insert(0, "<repo>/nn_x/w12/cb/famB")
from fb_common import *
import numba as nb

CAP = 4000   # at most the 4,000 most recent source rows per target feed
NAMES = (["ua_ns", "ua_npos"] + [f"ua_{k}_{s}" for k in ("cat", "c256", "c1024", "c4096") for s in ("imp", "pos", "coec")]
         + [f"ua_{k}_{s}" for k in ("cat", "c1024") for s in ("imprw", "posrw")] + ["ua_cat_share", "ua_c1024_share"]
         + ["ue_maxpos", "ue_maxneg", "ue_meanpos", "ue_meanneg"] + [f"ue_k{t}_{s}" for t in ("05", "20") for s in ("all", "pos", "coec")]
         + ["ue_top5_pos", "ue_top5_cos"] + ["ua_item_imp", "ua_item_pos", "ua_item_dt", "ua_dt_last"])
NC = len(NAMES)


@nb.njit(parallel=True, cache=True)
def kernel(t_lo, t_cnt, t_off, t_M, t_ts, tk, tit, temb, tnorm, s_off, s_M, s_ts, sk, sit, sy, sex, semb, snorm, out):
    nt = len(t_lo)
    for t in nb.prange(nt):
        q0 = t_off[t]; m = t_M[t]; cnt = t_cnt[t]
        if cnt == 0:
            for i in range(m):
                for c in range(out.shape[1]):
                    out[q0 + i, c] = np.nan
            continue
        lo = t_lo[t]
        r0 = s_off[lo]; r1 = s_off[lo + cnt - 1] + s_M[lo + cnt - 1]
        if r1 - r0 > CAP:
            r0 = r1 - CAP
        ns = r1 - r0
        w = np.empty(ns, np.float32); dth = np.empty(ns, np.float32); npos = 0.0
        f = lo
        for s in range(r0, r1):
            while s >= s_off[f] + s_M[f]:
                f += 1
            w[s - r0] = 0.5 ** ((t_ts[t] - s_ts[f]) / 172800.0)
            dth[s - r0] = (t_ts[t] - s_ts[f]) / 3600.0
            npos += sy[s]
        dt_last = (t_ts[t] - s_ts[lo + cnt - 1]) / 3600.0
        top_c = np.empty(5, np.float32); top_y = np.empty(5, np.float32)
        for i in range(m):
            o = q0 + i
            out[o, 0] = ns; out[o, 1] = npos
            # ---- key statistics ----
            for k in range(4):
                key = tk[o, k]; imp = 0.0; ps = 0.0; ex = 0.0; irw = 0.0; prw = 0.0
                if key >= 0:
                    for s in range(r0, r1):
                        if sk[s, k] == key:
                            imp += 1.0; ps += sy[s]; ex += sex[s]; irw += w[s - r0]; prw += w[s - r0] * sy[s]
                out[o, 2 + 3 * k] = imp; out[o, 3 + 3 * k] = ps; out[o, 4 + 3 * k] = (ps + 0.5) / (ex + 0.5)
                if k == 0:
                    out[o, 14] = irw; out[o, 15] = prw; out[o, 18] = imp / ns
                elif k == 2:
                    out[o, 16] = irw; out[o, 17] = prw; out[o, 19] = imp / ns
            # ---- embedding kernel statistics ----
            mxp = -2.0; mxn = -2.0; sp = 0.0; sn = 0.0; cp = 0.0; cn = 0.0
            ka5 = 0.0; kp5 = 0.0; ke5 = 0.0; ka20 = 0.0; kp20 = 0.0; ke20 = 0.0
            for j in range(5):
                top_c[j] = -2.0; top_y[j] = 0.0
            tn = tnorm[o]
            for s in range(r0, r1):
                d = 0
                for q in range(128):
                    d += np.int32(temb[o, q]) * np.int32(semb[s, q])
                den = tn * snorm[s]
                c = d / den if den > 0 else 0.0
                y = sy[s]
                if y > 0:
                    sp += c; cp += 1.0
                    if c > mxp: mxp = c
                else:
                    sn += c; cn += 1.0
                    if c > mxn: mxn = c
                e5 = np.exp((c - 1.0) / 0.05); e20 = np.exp((c - 1.0) / 0.2)
                ka5 += e5; kp5 += e5 * y; ke5 += e5 * sex[s]
                ka20 += e20; kp20 += e20 * y; ke20 += e20 * sex[s]
                if c > top_c[4]:
                    j = 4
                    while j > 0 and top_c[j - 1] < c:
                        top_c[j] = top_c[j - 1]; top_y[j] = top_y[j - 1]; j -= 1
                    top_c[j] = c; top_y[j] = y
            out[o, 20] = mxp if cp > 0 else np.nan
            out[o, 21] = mxn if cn > 0 else np.nan
            out[o, 22] = sp / cp if cp > 0 else np.nan
            out[o, 23] = sn / cn if cn > 0 else np.nan
            out[o, 24] = ka5; out[o, 25] = kp5; out[o, 26] = (kp5 + 0.02) / (ke5 + 0.02)
            out[o, 27] = ka20; out[o, 28] = kp20; out[o, 29] = (kp20 + 0.1) / (ke20 + 0.1)
            n5 = 0; s5 = 0.0; y5 = 0.0
            for j in range(5):
                if top_c[j] > -2.0:
                    n5 += 1; s5 += top_c[j]; y5 += top_y[j]
            out[o, 30] = y5; out[o, 31] = s5 / n5 if n5 > 0 else np.nan
            iimp = 0.0; ipos = 0.0; idt = np.nan
            for s in range(r0, r1):
                if sit[s] == tit[o]:
                    iimp += 1.0; ipos += sy[s]; idt = dth[s - r0]      # sources are in time order: the last hit is the most recent
            out[o, 32] = iimp; out[o, 33] = ipos; out[o, 34] = idt; out[o, 35] = dt_last


def main():
    t0 = time.time()
    f = cbx.feeds("train"); uid, ts, Mtr, offtr = f["uid"], f["ts"], f["M"], f["off"]
    S = cbx.train_sets()
    hold = np.zeros(len(Mtr), bool); hold[S["hold"]] = True
    # position prior from non-holdout train rows
    pos = np.load(cbx.FLAT / "train/cand_pos.npy", mmap_mode="r"); y_all = np.load(cbx.FLAT / "train/cand_target.npy", mmap_mode="r")
    pc_path = FEAT / "pos_prior.npy"
    if not pc_path.exists():
        nh = np.repeat(~hold, Mtr)
        p = np.asarray(pos); yy = np.asarray(y_all)
        cnt = np.bincount(p[nh], minlength=512); cl = np.bincount(p[nh], weights=yy[nh], minlength=512)
        pc = ((cl + 0.0159 * 50) / (cnt + 50)).astype(np.float32)
        np.save(pc_path, pc); del nh, p, yy
    pc = np.load(pc_path)
    log(f"position prior ready ({time.time() - t0:.0f}s): pos 0..5 {np.round(pc[:6], 4)}")
    sets = {n: cbx.load(n) for n in ("xm", "hold", "val", "test")}
    # users that occur in a target set
    tu = np.unique(np.concatenate([d.uid_feed for d in sets.values()])); tu = tu[tu > 0]
    src = np.where(~hold & (uid > 0) & np.isin(uid, tu))[0]
    o = np.lexsort((src, ts[src], uid[src])); src = src[o]                       # sorted by (user, time)
    su, sts, sM = uid[src], ts[src].astype(np.int64), Mtr[src]
    soff = np.r_[0, np.cumsum(sM)[:-1]].astype(np.int64)
    rows = cbx._rows_of(offtr[src], sM)                                          # flat train rows of the source feeds (compact order)
    log(f"source feeds {len(src)} rows {len(rows)} users {len(np.unique(su))} ({time.time() - t0:.0f}s)")
    g = lambda name: np.load(cbx.FLAT / f"train/cand_{name}.npy", mmap_mode="r")
    # gather with sorted indices (memmap friendly), then restore the compact order
    so = np.argsort(rows); rs = rows[so]; inv = np.empty_like(so); inv[so] = np.arange(len(so))
    take = lambda a: np.asarray(a[rs])[inv]
    sy = take(y_all).astype(np.float32); sex = pc[take(pos).astype(np.int64)].astype(np.float32)
    sk = np.stack([take(g("category_id")).astype(np.int32)] + [take(np.load(CLU / f"cand_cl_{K}_train.npy", mmap_mode="r")).astype(np.int32) for K in (256, 1024, 4096)], 1)
    sei = take(g("emb_idx")).astype(np.int32); sit = take(g("item_id")).astype(np.int64)
    semb = np.empty((len(rows), 128), np.int8); cbx._gather_rows_i8(cbx.emb_table("train"), sei, semb)
    snorm = np.sqrt((semb.astype(np.float32) ** 2).sum(1)).astype(np.float32)
    log(f"source arrays ready ({time.time() - t0:.0f}s): clicks {int(sy.sum())}, mean ex {sex.mean():.5f}, rss {cbx.rss_gb()[0]:.1f} GB")
    json.dump(NAMES + ["ua_c1024_coec_z", "ue_k05_coec_z", "ue_k20_coec_z"], open(FEAT / "a_names.json", "w"))
    for name, d in sets.items():
        t1 = time.time()
        tts = d.ts_feed.astype(np.int64); tuid = d.uid_feed
        lo = np.searchsorted(su, tuid, "left"); hi = np.searchsorted(su, tuid, "right")
        cnt = np.zeros(d.n_feeds, np.int64)
        if d.split == "train":
            # prefix of the user's sources with a strictly earlier timestamp
            key_s = su.astype(np.int64) * (1 << 32) + sts
            thr = tts if not LAG else ((tts + 3 * 3600) // 86400 - lag_days(d.fidx) + 1) * 86400 - 3 * 3600   # LAG: sources of day <= own day - L only
            key_t = tuid.astype(np.int64) * (1 << 32) + thr
            cnt = np.searchsorted(key_s, key_t, "left") - lo
        else:
            cnt = hi - lo
        cnt = np.where(tuid > 0, cnt, 0).astype(np.int64)
        assert (cnt >= 0).all()
        if d.split == "train":   # no source may be the target feed itself or later
            chk = np.where(cnt > 0)[0][:20000]
            assert all(sts[lo[c] + cnt[c] - 1] < tts[c] for c in chk) and not np.isin(src, S["hold"]).any()
        tk = np.stack([np.asarray(d.cat).astype(np.int32)] + [clu(d, K) for K in (256, 1024, 4096)], 1)
        temb = np.ascontiguousarray(d.emb_cand_i8)
        tnorm = np.sqrt((temb.astype(np.float32) ** 2).sum(1)).astype(np.float32)
        out = np.empty((d.n_rows, NC + 3), np.float32)
        kernel(lo.astype(np.int64), cnt, d.off.astype(np.int64), d.M.astype(np.int64), tts, tk, np.asarray(d.item_id).astype(np.int64), temb, tnorm,
               soff, sM.astype(np.int64), sts, sk, sit, sy, sex, semb, snorm, out[:, :NC])
        for j, c in enumerate((NAMES.index("ua_c1024_coec"), NAMES.index("ue_k05_coec"), NAMES.index("ue_k20_coec"))):
            x = out[:, c].copy(); nanm = np.isnan(x); x[nanm] = 0
            z = cbx.feed_z(d.off, d.M, np.ascontiguousarray(x)); z[nanm] = np.nan
            out[:, NC + j] = z
        if LAG and d.split != "train":
            continue                                               # val / test carry their natural lag: the fresh files are used
        np.save(FEAT / f"{PRE}a_{name}.npy", out)
        cov = (cnt > 0).mean()
        log(f"[{name}] feeds {d.n_feeds} covered {cov:.4f} mean source feeds | covered {cnt[cnt > 0].mean():.2f} rows written {out.shape} ({time.time() - t1:.0f}s)")
    log(f"DONE block a ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
