"""Build the cal_ feature family for train / val / test.

Context of a feed: ctx = weekend*7 + hour_bucket (MSK; buckets 0-6,6-9,9-12,12-15,15-18,18-21,21-24).
Candidate-level features (all interactions of the feed's calendar context with candidate properties):
  cal_lab_cat_ctx   log-lift of the candidate's category in the feed's context vs the category overall, from train labels,
                    position-debiased (expected positives from a position-only model fitted on random-like train feeds).
                    train rows: strictly earlier train days only (expanding window); val/test: all train.
  cal_lab_cat_we    same, context = day type only (weekend vs weekday)
  cal_lab_cat_hb    same, context = hour bucket only (pooled over day types)
  cal_lab_emb_ctx   item_unit . delta_ctx, delta_ctx = (mean emb of positives - mean emb of expected positives) in ctx minus the
                    same over all contexts (label "time direction" in embedding space; dense, works for cold items). Time-aware.
  cal_dem_cat_ctx   log-lift of the category among the newest-5 history events of train feeds in the same context
                    (population "demand" by time of day / day type; no labels)
  cal_dem_emb_ctx   item_unit . (mean emb of newest-5 history events in ctx - overall mean)  (dense)
  cal_sup_cat_ctx   transductive: log-lift of the category among the split's own candidates in the same context (supply)
  cal_ds_cat_ctx    cal_dem_cat_ctx - cal_sup_cat_ctx (demand/supply)
  cal_drift_cat_day transductive: category share among newest-5 history events of same-split feeds on the same MSK day vs
                    train-wide share (seasonal / day-level demand drift)
Feed-level (only act through interactions inside a GBDT):
  cal_mod (minute of day / 1440), cal_weekend, cal_dow, cal_payday (days since last typical RU payday 5/10/20/25),
  cal_u_hb_share (share of the user's earlier train feeds in the same hour bucket), cal_u_we_share (share on weekends),
  cal_u_ntr (number of the user's earlier train feeds)
"""
from calcommon import *

NC = 2 * N_HB
K_LAB = 50.0     # pseudo positives for label lifts
A_CNT = 30.0     # pseudo events for count lifts
D = 128


def load_table(split):
    meta = json.load(open(f"{ROOT}/data/competition/{split}/item_embeddings/data.meta"))
    return np.fromfile(f"{ROOT}/data/competition/{split}/item_embeddings/data.memmap", dtype=np.int8).reshape(meta["shape"])


@nb.njit(parallel=True, cache=True)
def acc_label_emb(off, M, day_f, ctx_f, emb_idx, y, e, table, ND, NCX, nchunk):
    n = len(M)
    Dd = table.shape[1]
    acc = np.zeros((nchunk, ND, NCX, 2, Dd + 1))
    chunk = (n + nchunk - 1) // nchunk
    for c in nb.prange(nchunk):
        v = np.empty(Dd)
        for f in range(c * chunk, min(n, (c + 1) * chunk)):
            d = day_f[f]; x = ctx_f[f]
            for j in range(off[f], off[f] + M[f]):
                ei = emb_idx[j]
                if ei < 0:
                    continue
                s = 0.0
                for k in range(Dd):
                    v[k] = table[ei, k]; s += v[k] * v[k]
                s = np.sqrt(s) + 1e-9
                yy = y[j]; ee = e[j]
                for k in range(Dd):
                    u = v[k] / s
                    acc[c, d, x, 0, k] += yy * u
                    acc[c, d, x, 1, k] += ee * u
                acc[c, d, x, 0, Dd] += yy
                acc[c, d, x, 1, Dd] += ee
    return acc.sum(0)


@nb.njit(parallel=True, cache=True)
def acc_hist_emb(hf, hemb, ctx_f, table, NCX, nchunk):
    n = len(hf)
    Dd = table.shape[1]
    acc = np.zeros((nchunk, NCX, Dd + 1))
    chunk = (n + nchunk - 1) // nchunk
    for c in nb.prange(nchunk):
        for i in range(c * chunk, min(n, (c + 1) * chunk)):
            ei = hemb[i]
            if ei < 0:
                continue
            x = ctx_f[hf[i]]
            s = 0.0
            for k in range(Dd):
                s += float(table[ei, k]) ** 2
            s = np.sqrt(s) + 1e-9
            for k in range(Dd):
                acc[c, x, k] += table[ei, k] / s
            acc[c, x, Dd] += 1.0
    return acc.sum(0)


@nb.njit(parallel=True, cache=True)
def dot_rows(off, M, key_f, emb_idx, table, dirs):
    out = np.full(emb_idx.shape[0], np.nan, np.float32)
    Dd = table.shape[1]
    for f in nb.prange(len(M)):
        kf = key_f[f]
        if kf < 0:
            continue
        for j in range(off[f], off[f] + M[f]):
            ei = emb_idx[j]
            if ei < 0:
                continue
            s = 0.0; dp = 0.0
            for k in range(Dd):
                v = float(table[ei, k])
                s += v * v; dp += v * dirs[kf, k]
            out[j] = dp / (np.sqrt(s) + 1e-9)
    return out


def lab_lifts(Y, E):
    """Y,E: [cat, ctx] -> dict of [cat, ctx] log-lift tables (ctx, weekend-only, hour-bucket-only), shrunk to 0."""
    Rc = (Y.sum(1) + 1.0) / (E.sum(1) + 1.0)
    out = {}
    out["ctx"] = np.log((Y + K_LAB) / (E * Rc[:, None] + K_LAB))
    Yw = np.stack([Y[:, :N_HB].sum(1), Y[:, N_HB:].sum(1)], 1); Ew = np.stack([E[:, :N_HB].sum(1), E[:, N_HB:].sum(1)], 1)
    lw = np.log((Yw + K_LAB) / (Ew * Rc[:, None] + K_LAB))
    out["we"] = np.concatenate([np.repeat(lw[:, :1], N_HB, 1), np.repeat(lw[:, 1:], N_HB, 1)], 1)
    Yh = Y[:, :N_HB] + Y[:, N_HB:]; Eh = E[:, :N_HB] + E[:, N_HB:]
    lh = np.log((Yh + K_LAB) / (Eh * Rc[:, None] + K_LAB))
    out["hb"] = np.concatenate([lh, lh], 1)
    return out


def cnt_lift(Cnt):
    """Cnt: [cat, ctx] counts -> log((C + a)/(expected + a)), expected = ctx total x overall category share."""
    s = Cnt.sum(1) / max(Cnt.sum(), 1)
    Ex = Cnt.sum(0)[None, :] * s[:, None]
    return np.log((Cnt + A_CNT) / (Ex + A_CNT))


def label_dirs(acc_dx):
    """acc_dx: [ctx, 2, D+1] -> [ctx, D] time-specific label direction."""
    P = acc_dx[:, 0, :D]; X = acc_dx[:, 1, :D]; py = acc_dx[:, 0, D]; pe = acc_dx[:, 1, D]
    g_ctx = P / np.maximum(py, 1)[:, None] - X / np.maximum(pe, 1e-9)[:, None]
    g_all = P.sum(0) / max(py.sum(), 1) - X.sum(0) / max(pe.sum(), 1e-9)
    # shrink contexts with few positives toward the global direction
    w = (py / (py + 2000.0))[:, None]
    return (w * (g_ctx - g_all[None, :])).astype(np.float64)


def payday_days(day):
    dm = day_of_month(day)
    paydays = np.array([5, 10, 20, 25])
    # days since the most recent payday (month assumed to be September 2026: previous month's 25th is 31-25+dm days ago)
    best = np.full(len(dm), 99)
    for p in paydays:
        d = dm - p
        best = np.where((d >= 0) & (d < best), d, best)
    best = np.where(best == 99, dm + 31 - 25, best)
    return best


def user_time_profile(F_tr, F_tgt, tgt_is_train):
    """share of the user's earlier train feeds in the same hour bucket / on weekends, and their count."""
    tp_tr = time_parts(F_tr["timestamp"].to_numpy())
    tr = pl.DataFrame({"u": F_tr["user_id"].to_numpy(), "day": tp_tr["day"], "hb": tp_tr["hb"], "we": tp_tr["weekend"].astype(np.int64)})
    tr = tr.filter(pl.col("u") > 0)
    tp_t = time_parts(F_tgt["timestamp"].to_numpy())
    tg = pl.DataFrame({"i": np.arange(len(F_tgt)), "u": F_tgt["user_id"].to_numpy(), "day": tp_t["day"], "hb": tp_t["hb"]})
    if tgt_is_train:
        # per (u, day) counts -> exclusive cumulative over days
        a = tr.group_by("u", "day").agg(pl.len().alias("n"), pl.col("we").sum().alias("nwe")).sort("u", "day")
        a = a.with_columns((pl.col("n").cum_sum().over("u") - pl.col("n")).alias("cn"),
                           (pl.col("nwe").cum_sum().over("u") - pl.col("nwe")).alias("cwe")).select("u", "day", "cn", "cwe")
        b = tr.group_by("u", "hb", "day").agg(pl.len().alias("n")).sort("u", "hb", "day")
        b = b.with_columns(pl.col("n").cum_sum().over("u", "hb").alias("chb")).select("u", "hb", "day", "chb")
        tg = tg.join(a, on=["u", "day"], how="left")
        tg = tg.sort("day").join_asof(b.sort("day"), on="day", by=["u", "hb"], allow_exact_matches=False, strategy="backward").sort("i")
    else:
        a = tr.group_by("u").agg(pl.len().alias("cn"), pl.col("we").sum().alias("cwe"))
        b = tr.group_by("u", "hb").agg(pl.len().alias("chb"))
        tg = tg.join(a, on="u", how="left").join(b, on=["u", "hb"], how="left").sort("i")
    cn = tg["cn"].fill_null(0).to_numpy().astype(np.float64)
    chb = tg["chb"].fill_null(0).to_numpy().astype(np.float64)
    cwe = tg["cwe"].fill_null(0).to_numpy().astype(np.float64)
    u = tg["u"].to_numpy()
    valid = (u > 0) & (cn > 0)
    hb_share = np.where(valid, chb / np.maximum(cn, 1), np.nan)
    we_share = np.where(valid, cwe / np.maximum(cn, 1), np.nan)
    ntr = np.where(u > 0, cn, np.nan)
    return hb_share, we_share, ntr


def main():
    t0 = time.time()
    # ------------------------------------------------------------------ train precompute
    F = feeds("train")
    M = F["M"].to_numpy().astype(np.int64); off = F["cand_off"].to_numpy().astype(np.int64)
    tp = time_parts(F["timestamp"].to_numpy())
    tdays = np.unique(tp["day"]); ND = len(tdays)
    day_f = np.searchsorted(tdays, tp["day"]).astype(np.int64)
    ctx_f = (tp["weekend"].astype(np.int64) * N_HB + tp["hb"]).astype(np.int64)
    fidx = cand("train", "fidx", mmap=False)
    y = cand("train", "target", mmap=False).astype(np.float32)
    pos = cand("train", "pos", mmap=False)
    cat = cand("train", "category_id", mmap=False).astype(np.int64)
    r, re = within_feed_rank(pos, off, M); del pos
    theta = np.load(f"{OUT}/theta_cells.npy")
    e = expected_pos(theta[theta_index(r, re)].astype(np.float64), y.astype(np.float64), off, M).astype(np.float32)
    del r, re
    log("train loaded + expected", time.time() - t0)
    drow = day_f[fidx]; crow = ctx_f[fidx]
    key = (drow * N_CAT + cat) * NC + crow
    Yt = np.bincount(key, weights=y, minlength=ND * N_CAT * NC).reshape(ND, N_CAT, NC)
    Et = np.bincount(key, weights=e, minlength=ND * N_CAT * NC).reshape(ND, N_CAT, NC)
    del key
    np.savez(f"{OUT}/lab_tables.npz", Y=Yt, E=Et, days=tdays)
    Ycum = np.cumsum(Yt, 0) - Yt; Ecum = np.cumsum(Et, 0) - Et   # strictly earlier days
    lifts_by_day = [lab_lifts(Ycum[d], Ecum[d]) for d in range(ND)]
    lifts_full = lab_lifts(Yt.sum(0), Et.sum(0))
    has_prev = Ycum.sum(-1) > 0  # [ND, cat]
    nodata = ~has_prev[drow, cat]
    for nm in ["ctx", "we", "hb"]:
        tab = np.stack([L[nm] for L in lifts_by_day])  # [ND, cat, ctx]
        x = tab[drow, cat, crow].astype(np.float32)
        x[nodata] = np.nan  # no earlier data
        save_feat("train", f"cal_lab_cat_{nm}", x)
    log("label category lifts", time.time() - t0)

    table = load_table("train")
    emb = cand("train", "emb_idx", mmap=False)
    acc = acc_label_emb(off, M, day_f, ctx_f, emb, y, e, table, ND, NC, 96)   # [ND, NC, 2, D+1]
    np.save(f"{OUT}/lab_emb_acc.npy", acc)
    acc_cum = np.cumsum(acc, 0) - acc
    dirs_day = np.stack([label_dirs(acc_cum[d]) for d in range(ND)])  # [ND, NC, D]
    dirs_full = label_dirs(acc.sum(0))
    key_f = np.where(acc_cum[day_f, :, 0, D].sum(-1) > 0, day_f * NC + ctx_f, -1)
    save_feat("train", "cal_lab_emb_ctx", dot_rows(off, M, key_f, emb, table, dirs_day.reshape(ND * NC, D)))
    log("label emb directions", time.time() - t0, "norms", np.round(np.linalg.norm(dirs_full, axis=1), 4))
    del y, e

    # history demand (newest 5 events), train only
    hp = hist("train", "hpos", mmap=False); hm = np.where(hp < 5)[0]; del hp
    hf = hist("train", "fidx", mmap=False)[hm]
    hc = hist("train", "category_id", mmap=False)[hm].astype(np.int64)
    he = hist("train", "emb_idx", mmap=False)[hm]
    Dcnt = np.bincount(hc * NC + ctx_f[hf], minlength=N_CAT * NC).reshape(N_CAT, NC).astype(np.float64)
    hacc = acc_hist_emb(hf.astype(np.int64), he, ctx_f, table, NC, 96)
    m_ctx = hacc[:, :D] / hacc[:, D:]
    m_all = hacc[:, :D].sum(0) / hacc[:, D].sum()
    dem_dirs = (m_ctx - m_all[None, :]).astype(np.float64)
    dem_lift = cnt_lift(Dcnt)
    # train-wide newest-5 category share (for drift) and per-train-day counts
    share_tr = np.bincount(hc, minlength=N_CAT) / len(hc)
    Dday_tr = np.bincount(day_f[hf] * N_CAT + hc, minlength=ND * N_CAT).reshape(ND, N_CAT).astype(np.float64)
    np.savez(f"{OUT}/demand_tables.npz", Dcnt=Dcnt, dem_dirs=dem_dirs, share_tr=share_tr, dirs_full=dirs_full,
             lab_ctx=lifts_full["ctx"], lab_we=lifts_full["we"], lab_hb=lifts_full["hb"])
    del hf, hc, he
    log("demand tables", time.time() - t0)

    def write_split(split, F_s, off_s, M_s, fidx_s, cat_s, emb_s, table_s, lab, dirs_key_f, dirs, drift_tab, is_train):
        tps = time_parts(F_s["timestamp"].to_numpy())
        ctx_s = (tps["weekend"].astype(np.int64) * N_HB + tps["hb"]).astype(np.int64)
        crow_s = ctx_s[fidx_s]
        if not is_train:
            for nm in ["ctx", "we", "hb"]:
                save_feat(split, f"cal_lab_cat_{nm}", lab[nm][cat_s, crow_s])
            save_feat(split, "cal_lab_emb_ctx", dot_rows(off_s, M_s, dirs_key_f, emb_s, table_s, dirs))
        save_feat(split, "cal_dem_cat_ctx", dem_lift[cat_s, crow_s])
        save_feat(split, "cal_dem_emb_ctx", dot_rows(off_s, M_s, ctx_s, emb_s, table_s, dem_dirs))
        # transductive supply within the split
        S = np.bincount(cat_s * NC + crow_s, minlength=N_CAT * NC).reshape(N_CAT, NC).astype(np.float64)
        sup = cnt_lift(S)
        save_feat(split, "cal_sup_cat_ctx", sup[cat_s, crow_s])
        save_feat(split, "cal_ds_cat_ctx", dem_lift[cat_s, crow_s] - sup[cat_s, crow_s])
        # drift: same-day newest-5 hist category share vs train-wide share
        sdays = np.unique(tps["day"]); dfs = np.searchsorted(sdays, tps["day"])
        save_feat(split, "cal_drift_cat_day", drift_tab(sdays, dfs)[dfs[fidx_s], cat_s])
        # feed-level
        save_feat(split, "cal_mod", (tps["sec"] / 86400.0)[fidx_s])
        save_feat(split, "cal_weekend", tps["weekend"][fidx_s])
        save_feat(split, "cal_dow", tps["dow"][fidx_s])
        save_feat(split, "cal_payday", payday_days(tps["day"])[fidx_s])
        hbs, wes, ntr = user_time_profile(F, F_s, is_train)
        save_feat(split, "cal_u_hb_share", hbs[fidx_s]); save_feat(split, "cal_u_we_share", wes[fidx_s]); save_feat(split, "cal_u_ntr", ntr[fidx_s])
        log("written", split, time.time() - t0)

    def drift_train(sdays, dfs):
        tot = Dday_tr.sum(1, keepdims=True)
        return np.log((Dday_tr + A_CNT) / (tot * share_tr[None, :] + A_CNT))

    write_split("train", F, off, M, fidx, cat, emb, table, None, None, None, drift_train, True)
    del table, emb, fidx, cat

    for split in ["val", "test"]:
        F_s = feeds(split)
        M_s = F_s["M"].to_numpy().astype(np.int64); off_s = F_s["cand_off"].to_numpy().astype(np.int64)
        fidx_s = cand(split, "fidx", mmap=False); cat_s = cand(split, "category_id", mmap=False).astype(np.int64)
        emb_s = cand(split, "emb_idx", mmap=False); table_s = load_table(split)
        tps = time_parts(F_s["timestamp"].to_numpy())
        ctx_s = (tps["weekend"].astype(np.int64) * N_HB + tps["hb"]).astype(np.int64)
        hp = hist(split, "hpos", mmap=False); hm = hp < 5
        hf_s = hist(split, "fidx", mmap=False)[hm]; hc_s = hist(split, "category_id", mmap=False)[hm].astype(np.int64)

        def drift_s(sdays, dfs, hf_s=hf_s, hc_s=hc_s):
            Dd_ = np.bincount(dfs[hf_s] * N_CAT + hc_s, minlength=len(sdays) * N_CAT).reshape(len(sdays), N_CAT).astype(np.float64)
            tot = Dd_.sum(1, keepdims=True)
            return np.log((Dd_ + A_CNT) / (tot * share_tr[None, :] + A_CNT))

        write_split(split, F_s, off_s, M_s, fidx_s, cat_s, emb_s, table_s, lifts_full, ctx_s, dirs_full, drift_s, False)
    log("done", time.time() - t0)


if __name__ == "__main__":
    main()
