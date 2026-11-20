"""Build the candidate new feature families (label-free) for val, test and the train subsets trRL / trProd.
Writes out/feats/<split>/<name>.npy (float32, row order = the split's flat candidate order; for trRL/trProd the
concatenated rows of the subset feeds in feed order, as lib_av.rows_of).

Families (prefix -> meaning):
  h_   history-structure (feed-internal): recency-weighted same-category mass (tau 5 / 20), its share, per-source
       same-category counts (source 1 = feed clicks, 2, >=3), source-1 share, first same-category type-1 recency,
       repeat-engaged same-category items, rank of the candidate's category in the user's recency-weighted category
       profile, history category entropy (feed-level), number of source-1 events (feed-level)
  rc_  candidate-set-relative within (feed, category): z and pct-rank of maxsim32, f1, pre2_fe_s, log1p(it_hist_all_luo),
       cos_u, ctx_nd_knn_h, log1p(seq_cv_invd20); group size; max f1 of the group
  a_   age-adjusted popularity: listing age in days (f4 knots), LUO popularity per day of life, LUO residual vs the
       f4-bin mean of the same split (transductive, label-free) and vs the trRL f4-bin mean, the f4-bin zero rate
  u_   user x item variants (val/test): cos_u - cos_hall, cos_u - cos_h10, cos_u * maxsim32, cosine with the user
       vector's residual after projecting out the recent-history centroid, cosine with that centroid, user-zero flag
  xs_  within-split exposure counts (val from val, test from test; label-free but transductive / test-set information):
       other-user candidate count (all / before own ts / after own ts), other users whose split history holds the
       item (all / before own ts)
  k_   cold-item content-kNN prior over the split's item table (val/test): similarity-weighted train popularity
       (log1p it_hist_all), train CTR and clicks-over-expected of the 20 nearest train-exposed items, top-1 cosine,
       near-duplicate count (cos > 0.95), mean popularity of the 5 nearest
"""
import sys, time, json
sys.path.insert(0, "<repo>/nn_x/audit/features")
from lib_av import *
import faiss

HERE = Path(__file__).resolve().parent
OUTF = HERE / "out/feats"
T0 = time.time()
LOG = open(HERE / "out/s03.log", "a")


def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.write(s + "\n"); LOG.flush()


KN = np.load(FEATS / "life_f4_age_knots.npy"); K64 = 64; CAP = 63 / 64


def age_hours(x):
    x = x.astype(np.float64); u = x * K64; k = np.clip(np.floor(u).astype(int), 1, K64 - 2); w = u - k
    a = KN[k] * (1 - w) + KN[k + 1] * w
    return np.where(x < CAP, a, 5 * 365 * 24.0)


def save(split, name, x):
    d = OUTF / split; d.mkdir(parents=True, exist_ok=True)
    np.save(d / f"{name}.npy", np.asarray(x, np.float32))


class S:
    """Arrays of one split (or of a feed subset of train), with local offsets."""

    def __init__(self, split, feeds=None):
        self.split = split; self.src = "train" if split.startswith("tr") else split
        f, M, off = meta(self.src)
        H = f["H"].to_numpy().astype(np.int64); hoff = f["hist_off"].to_numpy().astype(np.int64)
        ts = f["timestamp"].to_numpy().astype(np.int64); user = f["user_id"].to_numpy().astype(np.int64)
        if feeds is None:
            feeds = np.arange(len(M)); crows = None; hrows = None
        else:
            crows = rows_of(off, M, feeds); hrows = rows_of(hoff, H, feeds)
        self.feeds = feeds; self.M = M[feeds]; self.H = H[feeds]; self.off = local_off(self.M); self.hoff = local_off(self.H)
        self.ts = ts[feeds]; self.user = user[feeds]; self.crows = crows; self.hrows = hrows
        self.n = len(feeds); self.fid = np.repeat(np.arange(self.n), self.M)
        g = lambda n, r: (np.load(FLAT / self.src / f"{n}.npy", mmap_mode="r") if r is None
                          else np.asarray(np.load(FLAT / self.src / f"{n}.npy", mmap_mode="r")[r]))
        self.c_cat = np.asarray(g("cand_category_id", crows)).astype(np.int64)
        self.c_f1 = np.asarray(g("cand_f1", crows)).astype(np.float32); self.c_f4 = np.asarray(g("cand_f4", crows)).astype(np.float32)
        self.c_item = np.asarray(g("cand_item_id", crows)).astype(np.int64); self.c_emb = np.asarray(g("cand_emb_idx", crows)).astype(np.int64)
        self.h_cat = np.asarray(g("hist_category_id", hrows)).astype(np.int64); self.h_et = np.asarray(g("hist_event_type", hrows)).astype(np.int64)
        self.h_es = np.asarray(g("hist_event_source", hrows)).astype(np.int64); self.h_item = np.asarray(g("hist_item_id", hrows)).astype(np.int64)
        self.h_emb = np.asarray(g("hist_emb_idx", hrows)).astype(np.int64)
        self.user_emb = np.asarray(g("user_emb", feeds if crows is not None else None)).astype(np.float32)

    def feat(self, name):
        p = feat_file(self.src, name)
        x = np.load(p, mmap_mode="r")
        return np.asarray(x if self.crows is None else x[self.crows], np.float32)


# ------------------------------------------------------------------ family h_
@nb.njit(cache=True)
def hist_kernel(M, off, H, hoff, ccat, hcat, het, hes, hitem, out, feedout):
    nF = len(M)
    for i in range(nF):
        o = off[i]; m = M[i]; ho = hoff[i]; h = H[i]
        # per-feed category profile (tau 5 / 20), source counts, repeat items
        w5 = np.zeros(h); w20 = np.zeros(h); s5 = 0.0; s20 = 0.0; n_src1 = 0
        for k in range(h):
            w5[k] = np.exp(-k / 5.0); w20[k] = np.exp(-k / 20.0); s5 += w5[k]; s20 += w20[k]
            if hes[ho + k] == 1:
                n_src1 += 1
        # category mass map via sorting categories (<= 60 categories): use a dense array of size 256
        mass5 = np.zeros(256); mass20 = np.zeros(256)
        for k in range(h):
            c = hcat[ho + k]
            if 0 <= c < 256:
                mass5[c] += w5[k]; mass20[c] += w20[k]
        ent = 0.0
        if s5 > 0:
            for c in range(256):
                if mass5[c] > 0:
                    p = mass5[c] / s5; ent -= p * np.log(p)
        feedout[i, 0] = ent; feedout[i, 1] = n_src1
        # repeat items: count events per item (h <= 256: O(h^2) is fine)
        rep_cat = np.zeros(256)
        for k in range(h):
            it = hitem[ho + k]; first = True; cnt = 0
            for k2 in range(h):
                if hitem[ho + k2] == it:
                    cnt += 1
                    if k2 < k:
                        first = False
            if first and cnt >= 2:
                c = hcat[ho + k]
                if 0 <= c < 256:
                    rep_cat[c] += 1
        for j in range(m):
            c = ccat[o + j]
            if not (0 <= c < 256):
                continue
            out[o + j, 0] = mass5[c]; out[o + j, 1] = mass20[c]
            out[o + j, 2] = mass5[c] / s5 if s5 > 0 else 0.0
            src1 = 0; src2 = 0; src3 = 0; first_t1 = -1
            for k in range(h):
                if hcat[ho + k] == c:
                    es = hes[ho + k]
                    if es == 1:
                        src1 += 1
                    elif es == 2:
                        src2 += 1
                    elif es >= 3:
                        src3 += 1
                    if first_t1 < 0 and het[ho + k] == 1:
                        first_t1 = k
            out[o + j, 3] = src1; out[o + j, 4] = src2; out[o + j, 5] = src3
            out[o + j, 6] = src1 / n_src1 if n_src1 > 0 else np.nan
            out[o + j, 7] = 1.0 / (1.0 + first_t1) if first_t1 >= 0 else 0.0
            out[o + j, 8] = rep_cat[c]
            # rank of the candidate's category in the tau-5 profile (1 = top); absent -> 0
            if mass5[c] > 0:
                rk = 1
                for c2 in range(256):
                    if mass5[c2] > mass5[c]:
                        rk += 1
                out[o + j, 9] = rk
            else:
                out[o + j, 9] = 0.0
    return out, feedout


H_NAMES = ["h_cat_rw5", "h_cat_rw20", "h_cat_rw5_share", "h_cat_src1", "h_cat_src2", "h_cat_src3p", "h_cat_s1_share",
           "h_cat_t1_first", "h_cat_rep", "h_cat_rank"]


def build_h(s):
    out = np.zeros((len(s.c_cat), len(H_NAMES))); feedout = np.zeros((s.n, 2))
    hist_kernel(s.M, s.off, s.H, s.hoff, s.c_cat, s.h_cat, s.h_et, s.h_es, s.h_item, out, feedout)
    for j, n in enumerate(H_NAMES):
        save(s.split, n, out[:, j])
    save(s.split, "h_ent", feedout[s.fid, 0]); save(s.split, "h_n_src1", feedout[s.fid, 1])
    log(f"[{s.split}] h_ done {time.time()-T0:.0f}s")


# ------------------------------------------------------------------ family rc_
@nb.njit(cache=True)
def group_rel(M, off, cat, x, out_z, out_rk, out_n, out_max):
    for i in range(len(M)):
        o = off[i]; m = M[i]
        for c in np.unique(cat[o:o + m]):
            idx = np.where(cat[o:o + m] == c)[0] + o
            n = len(idx); mn = 1e30; cnt = 0
            for t in idx:
                if x[t] == x[t]:
                    cnt += 1
                    if x[t] < mn:
                        mn = x[t]
            v = np.empty(n)
            for q, t in enumerate(idx):
                v[q] = x[t] if x[t] == x[t] else (mn if cnt > 0 else 0.0)
            mu = v.mean(); sd = v.std(); mx = v.max()
            order = np.argsort(v); ranks = np.empty(n); q0 = 0
            while q0 < n:
                j = q0
                while j + 1 < n and v[order[j + 1]] == v[order[q0]]:
                    j += 1
                avg = 0.5 * (q0 + j) / max(n - 1, 1) - 0.5
                for k in range(q0, j + 1):
                    ranks[order[k]] = avg
                q0 = j + 1
            for q, t in enumerate(idx):
                out_z[t] = (v[q] - mu) / sd if sd > 1e-9 else 0.0
                out_rk[t] = ranks[q] if n > 1 else 0.0
                out_n[t] = n; out_max[t] = mx


RC_COLS = {"maxsim32": None, "f1": None, "pre2_fe_s": None, "it_hist_all_luo": "log1p", "cos_u": None,
           "ctx_nd_knn_h": None, "seq_cv_invd20": "log1p"}


def build_rc(s):
    first = True
    for c, tr in RC_COLS.items():
        x = s.feat(c).astype(np.float64)
        if tr == "log1p":
            x = np.log1p(np.maximum(x, 0))
        z = np.zeros(len(x)); rk = np.zeros(len(x)); n = np.zeros(len(x)); mx = np.zeros(len(x))
        group_rel(s.M, s.off, s.c_cat, x, z, rk, n, mx)
        save(s.split, f"rc_{c}_z", z); save(s.split, f"rc_{c}_rk", rk)
        if first:
            save(s.split, "rc_ncat", n); first = False
        if c == "f1":
            save(s.split, "rc_cat_maxf1", mx)
    log(f"[{s.split}] rc_ done {time.time()-T0:.0f}s")


# ------------------------------------------------------------------ family a_
def build_a(s, ref_means=None):
    luo = s.feat("it_hist_all_luo").astype(np.float64); cv = s.feat("seq_cv_invd20").astype(np.float64)
    age_d = age_hours(s.c_f4) / 24.0
    save(s.split, "a_age_days", np.log1p(age_d))
    save(s.split, "a_luo_rate", np.log1p(luo / np.maximum(age_d, 0.25)))
    b = np.clip(np.floor(s.c_f4.astype(np.float64) * 64).astype(int), 0, 63)
    ll = np.log1p(np.maximum(luo, 0)); lc = np.log1p(np.maximum(cv, 0))
    cnt = np.bincount(b, minlength=64).astype(np.float64)
    m_ll = np.bincount(b, ll, 64) / np.maximum(cnt, 1); m_lc = np.bincount(b, lc, 64) / np.maximum(cnt, 1)
    z_rate = np.bincount(b, (luo == 0).astype(np.float64), 64) / np.maximum(cnt, 1)
    save(s.split, "a_luo_resid_split", ll - m_ll[b]); save(s.split, "a_cv_resid_split", lc - m_lc[b])
    save(s.split, "a_zero_rate_split", z_rate[b])
    if ref_means is not None:
        save(s.split, "a_luo_resid_tr", ll - ref_means["m_ll"][b]); save(s.split, "a_zero_rate_tr", ref_means["z"][b])
    log(f"[{s.split}] a_ done; f4-bin zero rate (bins 2,9,20,40): {np.round(z_rate[[2, 9, 20, 40]], 3)} {time.time()-T0:.0f}s")
    return dict(m_ll=m_ll, z=z_rate)


# ------------------------------------------------------------------ family u_ (val/test)
@nb.njit(cache=True)
def hist_centroid(M, off, H, hoff, hemb, table, out_c):
    for i in range(len(M)):
        ho = hoff[i]; h = min(H[i], 32); acc = np.zeros(128); n = 0
        for k in range(h):
            e = hemb[ho + k]
            if e >= 0:
                for d in range(128):
                    acc[d] += table[e, d]
                n += 1
        if n > 0:
            nrm = np.sqrt((acc * acc).sum())
            if nrm > 0:
                for d in range(128):
                    out_c[i, d] = acc[d] / nrm


def build_u(s, table):
    cos_u = s.feat("cos_u").astype(np.float64); cos_hall = s.feat("cos_hall").astype(np.float64)
    cos_h10 = s.feat("cos_h10").astype(np.float64); maxsim = s.feat("maxsim32").astype(np.float64)
    save(s.split, "u_minus_hall", cos_u - cos_hall); save(s.split, "u_minus_h10", cos_u - cos_h10)
    save(s.split, "u_x_maxsim", cos_u * maxsim)
    hc = np.zeros((s.n, 128)); hist_centroid(s.M, s.off, s.H, s.hoff, s.h_emb, table, hc)
    ue = s.user_emb.astype(np.float64); un = np.linalg.norm(ue, axis=1); uz = un == 0
    uen = ue / np.maximum(un, 1e-6)[:, None]
    proj = (uen * hc).sum(1); ures = uen - proj[:, None] * hc
    urn = np.linalg.norm(ures, axis=1); ures = ures / np.maximum(urn, 1e-6)[:, None]
    ce = table[np.maximum(s.c_emb, 0)].astype(np.float64); cn = np.linalg.norm(ce, axis=1); ce = ce / np.maximum(cn, 1e-6)[:, None]
    ok = s.c_emb >= 0
    r = (ce * ures[s.fid]).sum(1); r[~ok] = np.nan; r[uz[s.fid]] = np.nan
    c = (ce * hc[s.fid]).sum(1); c[~ok] = np.nan; c[(np.linalg.norm(hc, axis=1) == 0)[s.fid]] = np.nan
    save(s.split, "u_res_cos", r); save(s.split, "u_hc_cos", c); save(s.split, "u_hc_proj", proj[s.fid])
    save(s.split, "u_zero", uz[s.fid].astype(np.float64))
    log(f"[{s.split}] u_ done; user-zero share {uz.mean():.4f}, mean u.hc proj {np.nanmean(proj):.3f} {time.time()-T0:.0f}s")


# ------------------------------------------------------------------ family xs_ (val/test, split-internal)
@nb.njit(cache=True)
def count_before_after(item_sorted_keys, item_sorted_ts, item_sorted_user, q_item, q_ts, q_user, out_other, out_before, out_after):
    for r in range(len(q_item)):
        it = q_item[r]
        lo = np.searchsorted(item_sorted_keys, it); hi = np.searchsorted(item_sorted_keys, it, side="right")
        other = 0; before = 0; after = 0
        for t in range(lo, hi):
            if item_sorted_user[t] == q_user[r] and q_user[r] != 0:
                continue
            other += 1
            if item_sorted_ts[t] < q_ts[r]:
                before += 1
            elif item_sorted_ts[t] > q_ts[r]:
                after += 1
        out_other[r] = other; out_before[r] = before; out_after[r] = after


def build_xs(s):
    ts_row = s.ts[s.fid]; user_row = s.user[s.fid]
    # candidate exposures in the split (one row per (feed, item))
    order = np.argsort(s.c_item, kind="stable")
    ks, kts, ku = s.c_item[order], ts_row[order], user_row[order]
    o1 = np.zeros(len(s.c_item)); o2 = np.zeros(len(s.c_item)); o3 = np.zeros(len(s.c_item))
    count_before_after(ks, kts, ku, s.c_item, ts_row, user_row, o1, o2, o3)
    # own row is counted in 'other' when user == 0 (cannot be excluded by user): subtract 1 for user-0 rows
    u0 = user_row == 0; o1[u0] -= 1
    save(s.split, "xs_cand_other", np.log1p(o1)); save(s.split, "xs_cand_before", np.log1p(o2)); save(s.split, "xs_cand_after", np.log1p(o3))
    # other users' split histories containing the item: distinct (item, user) with the user's earliest feed ts
    hf = np.repeat(np.arange(s.n), s.H)
    df = pl.DataFrame({"item": s.h_item, "user": s.user[hf], "ts": s.ts[hf], "fidx": hf})
    # for user 0 every feed is its own 'user': key by -(fidx+1)
    df = df.with_columns(pl.when(pl.col("user") == 0).then(-(pl.col("fidx") + 1)).otherwise(pl.col("user")).alias("ukey"))
    g = df.group_by(["item", "ukey"]).agg(pl.col("ts").min()).sort("item")
    gi, gts, gu = g["item"].to_numpy().astype(np.int64), g["ts"].to_numpy().astype(np.int64), g["ukey"].to_numpy().astype(np.int64)
    qkey = np.where(user_row == 0, -(s.fid + 1), user_row)
    h1 = np.zeros(len(s.c_item)); h2 = np.zeros(len(s.c_item)); h3 = np.zeros(len(s.c_item))
    count_before_after(gi, gts, gu, s.c_item, ts_row, qkey, h1, h2, h3)
    save(s.split, "xs_hist_other", np.log1p(h1)); save(s.split, "xs_hist_before", np.log1p(h2)); save(s.split, "xs_hist_after", np.log1p(h3))
    log(f"[{s.split}] xs_ done; cand other>0 {np.mean(o1 > 0):.3f} before>0 {np.mean(o2 > 0):.3f} after>0 {np.mean(o3 > 0):.3f}; "
        f"hist other>0 {np.mean(h1 > 0):.3f} before>0 {np.mean(h2 > 0):.3f} {time.time()-T0:.0f}s")


# ------------------------------------------------------------------ family k_ (val/test)
def build_k(s, table):
    st = np.load(FEATS / "item_stats.npz"); hist_all = st["hist_all"]; cct = st["cand_cnt_train"]
    tt = np.load(FEATS / "tte_tot_it.npz"); keys, imp, pos, ex = tt["keys"], tt["imp"], tt["pos"], tt["ex"]
    prior = pos.sum() / imp.sum()
    # emb row -> item id map from candidates and histories
    pairs = np.unique(np.stack([np.r_[s.c_emb, s.h_emb], np.r_[s.c_item, s.h_item]], 1), axis=0)
    pairs = pairs[pairs[:, 0] >= 0]
    ref = pairs[cct[pairs[:, 1]] > 0]  # items exposed in train (have tte stats and train popularity)
    ref_emb, ref_item = ref[:, 0], ref[:, 1]
    log(f"[{s.split}] k_: {len(pairs)} table rows with ids, {len(ref)} train-exposed reference items")
    X = table[ref_emb].astype(np.float32) / 256.0; faiss.normalize_L2(X)
    quant = faiss.IndexFlatIP(128); index = faiss.IndexIVFFlat(quant, 128, 2048, faiss.METRIC_INNER_PRODUCT)
    faiss.omp_set_num_threads(12)
    rng = np.random.default_rng(0); tr = X[rng.choice(len(X), min(len(X), 200_000), replace=False)]
    index.train(tr); index.add(X); index.nprobe = 24
    qemb, qinv = np.unique(np.maximum(s.c_emb, 0), return_inverse=True)
    Q = table[qemb].astype(np.float32) / 256.0; faiss.normalize_L2(Q)
    KQ = 21
    D_, I_ = index.search(Q, KQ)
    # map query emb row -> item id (for self exclusion): candidate rows
    q_item = np.full(len(qemb), -1, np.int64); q_item[qinv] = s.c_item
    nb_item = np.where(I_ >= 0, ref_item[np.maximum(I_, 0)], -1)
    self_m = (nb_item == q_item[:, None]) | (I_ < 0)
    Dm = np.where(self_m, -np.inf, D_)
    # drop self and keep the first 20 valid per row
    ordr = np.argsort(-Dm, axis=1)[:, :20]
    Dk = np.take_along_axis(Dm, ordr, 1); Ik = np.take_along_axis(nb_item, ordr, 1)
    valid = np.isfinite(Dk)
    pop = np.where(valid, np.log1p(hist_all[np.maximum(Ik, 0)]), 0.0)
    pos_i = np.searchsorted(keys, np.maximum(Ik, 0)); pos_i = np.minimum(pos_i, len(keys) - 1)
    hit = keys[pos_i] == Ik
    ctr = np.where(hit, (pos[pos_i] + prior * 20) / (imp[pos_i] + 20), prior)
    coec = np.where(hit, (pos[pos_i] + 1) / (ex[pos_i] + 1), 1.0)
    w = np.where(valid, np.exp(10 * np.clip(Dk, -1, 1)), 0.0); ws = w.sum(1)
    k_pop = (w * pop).sum(1) / np.maximum(ws, 1e-9); k_ctr = (w * ctr).sum(1) / np.maximum(ws, 1e-9)
    k_coec = (w * coec).sum(1) / np.maximum(ws, 1e-9)
    k_top1 = np.where(valid[:, 0], Dk[:, 0], np.nan); k_n95 = (valid & (Dk > 0.95)).sum(1)
    k_pop5 = np.where(valid[:, :5], pop[:, :5], np.nan).mean(1) if valid.any() else np.nan
    with np.errstate(all="ignore"):
        k_pop5 = np.nanmean(np.where(valid[:, :5], pop[:, :5], np.nan), 1)
    none = ws == 0
    for n, v in (("k_pop", k_pop), ("k_ctr", k_ctr), ("k_coec", k_coec), ("k_top1", k_top1), ("k_n95", k_n95.astype(float)), ("k_pop5", k_pop5)):
        v = v.astype(np.float64); v[none] = np.nan
        row = v[qinv]; row[s.c_emb < 0] = np.nan
        save(s.split, n, row)
    # own-item train popularity for reference (cold flag)
    log(f"[{s.split}] k_ done; queries {len(qemb)}, no-neighbour share {none.mean():.4f}, mean top1 cos {np.nanmean(k_top1):.3f} {time.time()-T0:.0f}s")


def load_table(split):
    d = ROOT / "data/competition" / split / "item_embeddings"
    m = json.loads((d / "data.meta").read_text())
    return np.fromfile(d / "data.memmap", dtype=np.int8).reshape(m["shape"])


if __name__ == "__main__":
    which = sys.argv[1:] or ["trRL", "trProd", "val", "test"]
    ftr, Mtr, offtr, subs = train_subsets()
    ref = None
    for sp in which:
        if sp.startswith("tr"):
            s = S(sp, subs[sp]); build_h(s); build_rc(s); r = build_a(s, ref)
            if sp == "trRL":
                ref = r; np.savez(OUTF / "a_ref_trRL.npz", **ref)
        else:
            if ref is None and (OUTF / "a_ref_trRL.npz").exists():
                z = np.load(OUTF / "a_ref_trRL.npz"); ref = dict(m_ll=z["m_ll"], z=z["z"])
            s = S(sp); table = load_table(sp)
            build_h(s); build_rc(s); build_a(s, ref); build_u(s, table); build_xs(s); build_k(s, table)
        del s
    log("DONE")
