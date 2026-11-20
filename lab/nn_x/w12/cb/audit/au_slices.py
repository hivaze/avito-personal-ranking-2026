"""audit 2: where the NDCG gap (W16 train-only pool minus tree) lives. Feed-level slices + positive-level decomposition.
Per-positive contribution c = 1/log2(rank+2)/IDCG with the pessimistic-tie rank; sum over a feed's positives = the feed's NDCG,
so sum over a bin of positives of (c_W16 - c_tree) / n_feeds is that bin's share of the gap (SE over feeds, zero for feeds without it)."""
from au_common import *
from au_ctx import build
m = Meta()
TR = {"cb26": TREES[[k for k in TREES if k.startswith("cb26_e3")][0]], "w25": TREES[[k for k in TREES if k.startswith("w25_A2c")][0]],
      "K134": TREES[[k for k in TREES if k.startswith("lgb_K_xm_134118")][0]]}


def contrib(s, y, off, M):
    """per-row contribution to the feed NDCG (0 for negatives) and 0-based pessimistic rank of every row."""
    n = len(M); fid = np.repeat(np.arange(n), M)
    order = np.lexsort((y, -s.astype(np.float64), fid))
    rank = np.empty(len(s), np.int64); rank[order] = np.arange(len(s)) - np.repeat(off, M)
    npos = np.bincount(fid, weights=y, minlength=n)
    cs = np.r_[0, np.cumsum(1.0 / np.log2(np.arange(M.max()) + 2.0))]
    idcg = cs[npos.astype(np.int64)]
    c = np.where(y > 0, 1.0 / np.log2(rank + 2.0) / np.maximum(idcg[fid], 1e-12), 0.0)
    return c, rank, fid, npos


def table(title, groups, e_w, e_t, names):
    """groups: list of (label, mask over feeds)."""
    print(f"\n-- {title} --")
    hdr = f"{'slice':26s} {'n':>6s} {'share':>6s} {'W16':>7s}"
    for t in names: hdr += f" | {t+' gap':>20s} {'z':>5s} {'%gap':>5s}"
    print(hdr)
    tot = {t: (e_w - e_t[t]).sum() for t in names}
    for lab, msk in groups:
        if msk.sum() == 0: continue
        s = f"{lab:26s} {msk.sum():6d} {msk.mean():6.3f} {e_w[msk].mean():7.4f}"
        for t in names:
            d = (e_w - e_t[t])[msk]
            s += f" | {d.mean():+.5f} +- {se(d):.5f} {d.mean()/max(se(d),1e-12):5.1f} {100*d.sum()/tot[t]:5.1f}"
        print(s)


def bins(x, edges, labels):
    return [(lab, (x >= lo) & (x < hi)) for lab, (lo, hi) in zip(labels, zip(edges[:-1], edges[1:]))]


for sp in ("val", "hold"):
    c = build(sp); y, off, M = m.y[sp], m.off[sp], m.M[sp]
    assert (c["M"] == M).all()
    zw = w16_pool(m, sp); names = [t for t in TR if sp in TR[t]]
    zt = {t: m.z(sp, np.load(TR[t][sp])) for t in names}
    cw, rw, fid, npos = contrib(zw, y, off, M)
    e_w = np.bincount(fid, weights=cw, minlength=len(M)); e_w[npos == 0] = 1.0
    assert abs(e_w.mean() - ndcg_fast(zw, y, off, M).mean()) < 1e-9
    ct, rt, e_t = {}, {}, {}
    for t in names:
        ct[t], rt[t], _, _ = contrib(zt[t], y, off, M); e = np.bincount(fid, weights=ct[t], minlength=len(M)); e[npos == 0] = 1.0; e_t[t] = e
    print(f"\n================ {sp.upper()}: {len(M)} feeds, {int(y.sum())} positives; W16 {e_w.mean():.5f}; " + "; ".join(f"{t} {e_t[t].mean():.5f} (gap {fmt(e_w - e_t[t])})" for t in names))
    H, uid = c["H"], c["uid"]
    table("feed size M", bins(M, [0, 20, 50, 100, 200, 10**6], ["M 5-19", "M 20-49", "M 50-99", "M 100-199", "M 200+"]), e_w, e_t, names)
    table("history length H", bins(H, [0, 1, 10, 50, 150, 256, 10**6], ["H 0", "H 1-9", "H 10-49", "H 50-149", "H 150-255", "H 256 (cap)"]), e_w, e_t, names)
    known = (uid != 0) & (c["n_other"] > 0)
    if sp == "val":
        g = [("user 0", uid == 0), ("known train user", known), ("new user", (uid != 0) & ~known)]
    else:
        g = [("user 0", uid == 0), ("user with other tr feeds", known), ("user with only this feed", (uid != 0) & ~known)]
    table("user type", g, e_w, e_t, names)
    table("known users by n other train feeds", bins(np.where(uid == 0, -9, c["n_other"]), [1, 2, 5, 20, 10**6], ["1", "2-4", "5-19", "20+"]), e_w, e_t, names)
    cold = (np.nan_to_num(c["c_it_hist_all_luo"]) <= 0).astype(np.float64); cs = np.bincount(fid, weights=cold, minlength=len(M)) / M
    q = np.quantile(cs, [1 / 3, 2 / 3])
    table(f"cold-candidate share terciles (cuts {q[0]:.3f}, {q[1]:.3f})", [("cold low", cs <= q[0]), ("cold mid", (cs > q[0]) & (cs <= q[1])), ("cold high", cs > q[1])], e_w, e_t, names)
    table("cold-candidate share, fixed bins", bins(cs, [0, 0.1, 0.25, 0.5, 0.75, 1.01], ["<0.10", "0.10-0.25", "0.25-0.50", "0.50-0.75", ">=0.75"]), e_w, e_t, names)
    table("number of positives", bins(npos, [0, 1, 2, 3, 5, 10**6], ["0 (NDCG=1)", "1", "2", "3-4", "5+"]), e_w, e_t, names)
    wd = (c["day"] + 3) % 7  # 1970-01-01 was a Thursday: (day + 3) % 7 -> 0 = Monday
    table("weekday (0=Mon)", [(f"wd {k} ({['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][k]})", wd == k) for k in range(7)], e_w, e_t, names)
    table("hour (Moscow)", bins(c["hour"], [0, 6, 12, 18, 24], ["00-05", "06-11", "12-17", "18-23"]), e_w, e_t, names)
    # ---------- positive level ----------
    P = y > 0; nF = len(M)
    def ptable(title, groups):
        print(f"\n-- positives: {title} --")
        hdr = f"{'bin':26s} {'n_pos':>7s} {'share':>6s} {'c_W16':>7s} {'medrk%W':>7s}"
        for t in names: hdr += f" | {t+': gap share':>24s} {'%gap':>5s} {'c_tree':>7s} {'medrk%':>6s} {'rel.loss':>8s}"
        print(hdr)
        for lab, msk in groups:
            mk = msk & P
            if mk.sum() == 0: continue
            rpw = (rw[mk] / np.maximum(M[fid[mk]] - 1, 1))
            s = f"{lab:26s} {mk.sum():7d} {mk.sum()/P.sum():6.3f} {cw[mk].mean():7.4f} {100*np.median(rpw):7.1f}"
            for t in names:
                d = np.bincount(fid[mk], weights=(cw - ct[t])[mk], minlength=nF)
                tot = (e_w - e_t[t]).sum()
                s += f" | {d.mean():+.5f} +- {se(d):.5f} {100*d.sum()/tot:5.1f} {ct[t][mk].mean():7.4f} {100*np.median(rt[t][mk]/np.maximum(M[fid[mk]]-1,1)):6.1f} {100*(1-ct[t][mk].sum()/cw[mk].sum()):7.1f}%"
            print(s)
    pop = np.nan_to_num(c["c_it_hist_all_luo"])
    ptable("item popularity = it_hist_all_luo (count of the item in other users' histories)", bins(pop, [-1, 0.5, 3, 10, 30, 100, 1e12], ["0 (cold)", "1-2", "3-9", "10-29", "30-99", "100+"]))
    imp = np.nan_to_num(c["c_tte_it_imp"])
    ptable("item train impressions (tte_it_imp, time-aware)", bins(imp, [-1, 0.5, 5, 20, 100, 1e12], ["0", "1-4", "5-19", "20-99", "100+"]))
    cc = np.nan_to_num(c["c_cat_cnt_all"]); Hrow = H[fid]
    ptable("category of the positive in the user's history", [("no history (H=0)", Hrow == 0), ("category unseen", (Hrow > 0) & (cc <= 0)), ("category seen 1-2x", (Hrow > 0) & (cc > 0) & (cc < 3)),
                                                              ("category seen 3-9x", (cc >= 3) & (cc < 10)), ("category seen 10+x", cc >= 10)])
    sm = c["c_seq_max_all"]; has = np.isfinite(sm) & (Hrow > 0)
    ptable("max cosine to a history event (seq_max_all), within-feed rank pct of the positive (0 = least similar)",
           [("no history / NaN", ~has)] + [(lab, has & mk) for lab, mk in bins(np.nan_to_num(c["c_seq_max_all_rk"], nan=-1), [0, 0.5, 0.8, 0.95, 1.01], ["bottom half", "50-80%", "80-95%", "top 5%"])])
    qs = np.nanquantile(sm[P & has], [0.25, 0.5, 0.75])
    ptable(f"max cosine to a history event, absolute (quartiles of positives: {qs.round(3).tolist()})",
           [(lab, has & mk) for lab, mk in bins(np.nan_to_num(sm, nan=-9), [-2, qs[0], qs[1], qs[2], 9], ["Q1 (least similar)", "Q2", "Q3", "Q4 (most similar)"])])
    em = np.nan_to_num(c["c_emb_missing"]) > 0
    ptable("item embedding missing", [("embedding present", ~em), ("embedding missing", em)])
    ptable("cold x category", [("warm, cat seen", (pop > 0) & (cc > 0)), ("warm, cat unseen", (pop > 0) & (cc <= 0)), ("cold, cat seen", (pop <= 0) & (cc > 0)), ("cold, cat unseen", (pop <= 0) & (cc <= 0))])
    # rank transition: where W16 places the positives the tree misses
    print("\n-- positives: head hit rates (share of positives at pessimistic rank < k) --")
    for k in (1, 3, 10, 30):
        print(f"  rank < {k:2d}: W16 {(rw[P] < k).mean():.4f}" + "".join(f" | {t} {(rt[t][P] < k).mean():.4f}" for t in names))
    t = names[0]; rb = lambda r: np.digitize(r, [1, 3, 10, 30])
    A = np.zeros((5, 5)); G = np.zeros((5, 5))
    np.add.at(A, (rb(rw[P]), rb(rt[t][P])), 1); np.add.at(G, (rb(rw[P]), rb(rt[t][P])), (cw - ct[t])[P])
    print(f"-- positives: transition W16 rank bucket (rows) x {t} rank bucket (cols) [0, 1-2, 3-9, 10-29, 30+]: % of the total gap --")
    print(np.round(100 * G / G.sum(), 1)); print("   counts (% of positives)"); print(np.round(100 * A / A.sum(), 1))
