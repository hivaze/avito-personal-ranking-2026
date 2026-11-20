"""(1) Same-user LATER-feed history leak, measured on top of G3 (val) and its exact test coverage.
Variants: later feeds in val only (the realistic analogue of test, where only later TEST feeds exist) vs later in val+test.
Also: later-candidate flag (item shown again later => not clicked), the later feed's user-embedding variant,
and all orderings of same-user feeds across splits (coverage table)."""
from lib import *

V = Split("val"); T = Split("test"); TR = feeds("train")
base = g3(V); pf0 = V.ndcg(base); log(f"G3 val {pf0.mean():.5f}")

# ---------- all feeds of non-zero users that appear in val or test ----------
SID = {"train": 0, "val": 1, "test": 2}
users = np.unique(np.concatenate([V.user[V.user > 0], T.user[T.user > 0]]))
meta = []
for s, F in (("train", TR), ("val", V.f), ("test", T.f)):
    m = F.filter(pl.col("user_id").is_in(users.tolist())).select("fidx", "user_id", "timestamp", "M", "H", "cand_off", "hist_off")
    meta.append(m.with_columns(pl.lit(SID[s]).cast(pl.Int8).alias("sid")))
meta = pl.concat(meta)
log("feeds of val/test users by split:", meta.group_by("sid").len().sort("sid").to_dicts())


def hist_rows(split, m):
    F = m.filter(pl.col("sid") == SID[split])
    ho = F["hist_off"].to_numpy(); H = F["H"].to_numpy()
    idx = np.concatenate([np.arange(o, o + h) for o, h in zip(ho, H)]) if len(F) else np.zeros(0, np.int64)
    return pl.DataFrame({"sid": np.full(len(idx), SID[split], np.int8), "ofidx": np.repeat(F["fidx"].to_numpy(), H),
                         "user_id": np.repeat(F["user_id"].to_numpy(), H), "ots": np.repeat(F["timestamp"].to_numpy(), H),
                         "item_id": np.asarray(arr(split, "hist_item_id")[idx]), "etype": np.asarray(arr(split, "hist_event_type")[idx]),
                         "esrc": np.asarray(arr(split, "hist_event_source")[idx]), "hpos": np.asarray(arr(split, "hist_hpos")[idx])})


def cand_rows(split, m):
    F = m.filter(pl.col("sid") == SID[split])
    co = F["cand_off"].to_numpy(); M = F["M"].to_numpy()
    idx = np.concatenate([np.arange(o, o + k) for o, k in zip(co, M)]) if len(F) else np.zeros(0, np.int64)
    return pl.DataFrame({"sid": np.full(len(idx), SID[split], np.int8), "ofidx": np.repeat(F["fidx"].to_numpy(), M),
                         "user_id": np.repeat(F["user_id"].to_numpy(), M), "ots": np.repeat(F["timestamp"].to_numpy(), M),
                         "item_id": np.asarray(arr(split, "cand_item_id")[idx])})


HIST = pl.concat([hist_rows(s, meta) for s in ("train", "val", "test")])
CAND = pl.concat([cand_rows(s, meta) for s in ("train", "val", "test")])
log("linked hist rows", len(HIST), "cand rows", len(CAND))


def link(S, split):
    """per-candidate flags for split S (flat row order). Returns DataFrame with row + flags."""
    sid = SID[split]
    b = pl.DataFrame({"row": np.arange(S.N, dtype=np.int64), "fidx": S.fid.astype(np.int32), "item_id": S.item,
                      "user_id": S.user[S.fid], "ts": S.ts[S.fid].astype(np.int32)}).filter(pl.col("user_id") > 0)
    hm = b.join(HIST, on=["user_id", "item_id"]).filter(~((pl.col("sid") == sid) & (pl.col("ofidx") == pl.col("fidx"))))
    hm = hm.with_columns((pl.col("ots").cast(pl.Int64) - pl.col("ts").cast(pl.Int64)).alias("dt"))
    later = hm.filter(pl.col("dt") > 0)
    # earliest later feed; oldest event in it (max hpos) gives the type
    agg = (later.sort(["row", "dt", "hpos"], descending=[False, False, True]).group_by("row")
           .agg(pl.col("dt").first().alias("lh_dt"), pl.col("etype").first().alias("lh_type"), pl.col("esrc").first().alias("lh_src"),
                pl.col("sid").first().alias("lh_sid"), pl.col("sid").min().alias("lh_sid_min"),
                ((pl.col("etype") == 1) & (pl.col("esrc") == 1)).any().alias("lh_fc"),
                (pl.col("sid") == sid).any().alias("lh_same_split")))
    earlier = hm.filter(pl.col("dt") < 0).group_by("row").agg(pl.len().alias("eh_n"), pl.col("etype").min().alias("eh_type"),
                                                              pl.col("dt").max().alias("eh_dt"))
    cm = b.join(CAND, on=["user_id", "item_id"]).filter(~((pl.col("sid") == sid) & (pl.col("ofidx") == pl.col("fidx"))))
    cm = cm.with_columns((pl.col("ots").cast(pl.Int64) - pl.col("ts").cast(pl.Int64)).alias("dt"))
    lc = cm.filter(pl.col("dt") > 0).group_by("row").agg(pl.len().alias("lc_n"), pl.col("dt").min().alias("lc_dt"),
                                                        (pl.col("sid") == sid).any().alias("lc_same_split"))
    ec = cm.filter(pl.col("dt") < 0).group_by("row").agg(pl.len().alias("ec_n"))
    out = (pl.DataFrame({"row": np.arange(S.N, dtype=np.int64)}).join(agg, on="row", how="left").join(earlier, on="row", how="left")
           .join(lc, on="row", how="left").join(ec, on="row", how="left").sort("row"))
    return out


# per-feed: does the user have a later feed (by split), gap to next feed
def feed_links(S, split):
    sid = SID[split]
    fm = pl.DataFrame({"fidx": np.arange(S.n, dtype=np.int32), "user_id": S.user, "ts": S.ts.astype(np.int64)}).filter(pl.col("user_id") > 0)
    o = meta.select("sid", pl.col("fidx").alias("ofidx"), "user_id", pl.col("timestamp").cast(pl.Int64).alias("ots"))
    j = fm.join(o, on="user_id").filter(~((pl.col("sid") == sid) & (pl.col("ofidx") == pl.col("fidx"))))
    j = j.with_columns((pl.col("ots") - pl.col("ts")).alias("dt"))
    g = j.group_by("fidx").agg(
        (pl.col("dt") > 0).sum().alias("n_later_any"),
        ((pl.col("dt") > 0) & (pl.col("sid") == sid)).sum().alias("n_later_same"),
        ((pl.col("dt") > 0) & (pl.col("sid") == 2)).sum().alias("n_later_test"),
        ((pl.col("dt") > 0) & (pl.col("sid") == 1)).sum().alias("n_later_val"),
        ((pl.col("dt") < 0) & (pl.col("sid") == 0)).sum().alias("n_earlier_train"),
        ((pl.col("dt") < 0) & (pl.col("sid") == 1)).sum().alias("n_earlier_val"),
        ((pl.col("dt") < 0) & (pl.col("sid") == 2)).sum().alias("n_earlier_test"),
        (pl.col("dt") == 0).sum().alias("n_same_ts"),
        pl.col("dt").filter((pl.col("dt") > 0) & (pl.col("sid") == sid)).min().alias("gap_next_same"),
        pl.col("dt").filter(pl.col("dt") > 0).min().alias("gap_next_any"))
    full = pl.DataFrame({"fidx": np.arange(S.n, dtype=np.int32)}).join(g, on="fidx", how="left").fill_null(0).sort("fidx")
    return full


LV = link(V, "val"); LT = link(T, "test")
FV = feed_links(V, "val"); FT = feed_links(T, "test")
LV.write_parquet(f"{OUT}/out_link_val.parquet"); LT.write_parquet(f"{OUT}/out_link_test.parquet")
FV.write_parquet(f"{OUT}/out_feedlinks_val.parquet"); FT.write_parquet(f"{OUT}/out_feedlinks_test.parquet")

print("\n=== Coverage: same-user feed orderings (non-zero users) ===")
for name, S, F in (("val", V, FV), ("test", T, FT)):
    masks = {"all": np.ones(S.n, bool)} if name == "val" else {"public": S.public, "private": ~S.public, "all": np.ones(S.n, bool)}
    for mk, mm in masks.items():
        c = F.filter(pl.Series(mm))
        print(f"{name}/{mk:8s} feeds {mm.sum():6d} | later feed same split {int((c['n_later_same'] > 0).sum()):4d} | later in test {int((c['n_later_test'] > 0).sum()):4d} "
              f"| later in val {int((c['n_later_val'] > 0).sum()):4d} | earlier train {int((c['n_earlier_train'] > 0).sum()):5d} | earlier val {int((c['n_earlier_val'] > 0).sum()):4d} "
              f"| earlier test {int((c['n_earlier_test'] > 0).sum()):4d} | same-ts other feed {int((c['n_same_ts'] > 0).sum()):3d}")
    gaps = F.filter(pl.col("n_later_same") > 0)["gap_next_same"].to_numpy() / 3600
    if len(gaps):
        print(f"   gap to next same-split feed (h): median {np.median(gaps):.1f}, q25 {np.percentile(gaps, 25):.1f}, q75 {np.percentile(gaps, 75):.1f}, "
              f"<=1h {(gaps <= 1).mean():.2f}, <=6h {(gaps <= 6).mean():.2f}, <=24h {(gaps <= 24).mean():.2f}")

# ---------- val: flags ----------
lh_sid_min = LV["lh_sid_min"].to_numpy()
leak_any = LV["lh_dt"].is_not_null().to_numpy()            # later feed in val or test
leak_valonly = LV["lh_same_split"].fill_null(False).to_numpy()  # later feed in val (realistic analogue)
lc_any = LV["lc_n"].is_not_null().to_numpy()
lc_valonly = LV["lc_same_split"].fill_null(False).to_numpy()
y = V.y
print("\n=== Val candidate flags (base rate %.4f) ===" % y.mean())
for nm, m in (("later_hist any(val+test)", leak_any), ("later_hist val-only", leak_valonly), ("later_cand any", lc_any), ("later_cand val-only", lc_valonly),
              ("earlier_hist (past info)", LV["eh_n"].is_not_null().to_numpy()), ("earlier_cand (past info)", LV["ec_n"].is_not_null().to_numpy())):
    if m.sum():
        print(f"  {nm:28s} rows {m.sum():5d} feeds {len(np.unique(V.fid[m])):4d} target rate {y[m].mean():.3f} positives {int(y[m].sum())}")
# by event type of first occurrence
lt = LV["lh_type"].to_numpy()
for t in range(1, 11):
    m = leak_any & (lt == t)
    if m.sum():
        print(f"    first-occurrence type {t:2d}: rows {m.sum():4d} target rate {y[m].mean():.3f}")
fc = LV["lh_fc"].fill_null(False).to_numpy()
print(f"    has type1/src1 event later: rows {(leak_any & fc).sum()} rate {y[leak_any & fc].mean():.3f}; without: rows {(leak_any & ~fc).sum()} rate {y[leak_any & ~fc].mean():.3f}")

# ---------- val gain on G3 ----------
print("\n=== Val gain on top of G3 (push leaked items to top / later-candidates to bottom) ===")
res = {}
for nm, pos, neg in (("later_hist any", leak_any, None), ("later_hist any + later_cand", leak_any, lc_any),
                     ("later_hist val-only", leak_valonly, None), ("later_hist val-only + later_cand val-only", leak_valonly, lc_valonly),
                     ("later_hist any, type-1/src-1 only", leak_any & fc, None),
                     ("later_cand any only (push down)", None, lc_any)):
    s = base.copy()
    if pos is not None: s += BIG * pos
    if neg is not None: s -= BIG * neg
    pf = V.ndcg(s); res[nm] = pf
    aff = np.zeros(V.n, bool)
    if pos is not None: aff |= np.bincount(V.fid, pos, V.n) > 0
    if neg is not None: aff |= np.bincount(V.fid, neg, V.n) > 0
    d, se, z, n = paired(pf, pf0)
    da, sea, _, na = paired(pf, pf0, aff)
    print(f"  {nm:48s} delta {d:+.5f} ± {se:.5f} (z {z:+.1f}) | affected feeds {na:4d}: {pf0[aff].mean():.3f} -> {pf[aff].mean():.3f} (per-feed {da:+.3f} ± {sea:.3f})")

# gain by gap to the leaking feed
print("\n  per-affected-feed gain by gap to the next same-user feed (any split), later_hist any:")
pf = res["later_hist any"]
gap_any = FV["gap_next_any"].to_numpy() / 3600
aff = np.bincount(V.fid, leak_any, V.n) > 0
for lo, hi in ((0, 1), (1, 6), (6, 24), (24, 72), (72, 1e9)):
    m = aff & (gap_any > lo) & (gap_any <= hi)
    has_later = (FV["n_later_any"].to_numpy() > 0) & (gap_any > lo) & (gap_any <= hi)
    if has_later.sum():
        print(f"    gap ({lo},{hi}] h: feeds with later feed {has_later.sum():4d}, affected {m.sum():4d} ({m.sum() / has_later.sum():.2f}); "
              f"gain per feed-with-later-feed {(pf - pf0)[has_later].mean():+.3f}, per affected {(pf - pf0)[m].mean() if m.sum() else float('nan'):+.3f}")

# ---------- projection to test ----------
print("\n=== Projection to test ===")
# per-feed-with-later-feed gain on val, bucketed by gap; apply to the test gap distribution
gap_t = FT["gap_next_same"].to_numpy() / 3600
has_t = FT["n_later_same"].to_numpy() > 0
has_v = FV["n_later_any"].to_numpy() > 0
edges = [0, 1, 6, 24, 72, 1e9]
gv = pf - pf0
proj = {}
for mk, mm in (("public", T.public), ("private", ~T.public), ("all", np.ones(T.n, bool))):
    tot = 0.0; parts = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mv = has_v & (gap_any > lo) & (gap_any <= hi); mt = has_t & mm & (gap_t > lo) & (gap_t <= hi)
        g = gv[mv].mean() if mv.sum() else gv[has_v].mean()
        tot += g * mt.sum(); parts.append(f"({lo},{hi}]:{mt.sum()}x{g:+.3f}")
    proj[mk] = tot / mm.sum()
    flat = gv[has_v].mean() * (has_t & mm).sum() / mm.sum()
    print(f"  {mk:8s} feeds {mm.sum():5d}, with later same-user test feed {(has_t & mm).sum():4d} ({(has_t & mm).mean():.4%}); "
          f"projected gain gap-matched {proj[mk]:+.5f}, flat {flat:+.5f} | {' '.join(parts)}")
# flagged test candidates
lt_pos = LT["lh_same_split"].fill_null(False).to_numpy(); lt_neg = LT["lc_same_split"].fill_null(False).to_numpy()
for mk, mm in (("public", T.public), ("private", ~T.public)):
    fp = np.bincount(T.fid, lt_pos, T.n) > 0; fn = np.bincount(T.fid, lt_neg, T.n) > 0
    print(f"  {mk:8s} test feeds with >=1 later_hist candidate {int((fp & mm).sum())}, with later_cand {int((fn & mm).sum())}, either {int(((fp | fn) & mm).sum())}; "
          f"flagged rows {int((lt_pos & mm[T.fid]).sum())} / {int((lt_neg & mm[T.fid]).sum())}")
np.save(f"{OUT}/out_test_leak_pos.npy", lt_pos); np.save(f"{OUT}/out_test_leak_neg.npy", lt_neg)

# ---------- later feed's user embedding ----------
print("\n=== Later-feed user-embedding variant (val feeds whose user has a later feed in val/test) ===")
emb = np.memmap(f"{ROOT}/data/competition/val/item_embeddings/data.memmap", dtype=np.int8, mode="r", shape=(2516711, 128))
cei = np.asarray(arr("val", "cand_emb_idx"))
ue = {"val": np.asarray(arr("val", "user_emb")).astype(np.float32), "test": np.asarray(arr("test", "user_emb")).astype(np.float32)}
# next feed (any split among val/test) per val feed
o = meta.filter(pl.col("sid") >= 1).select("sid", pl.col("fidx").alias("ofidx"), "user_id", pl.col("timestamp").cast(pl.Int64).alias("ots"))
fm = pl.DataFrame({"fidx": np.arange(V.n, dtype=np.int32), "user_id": V.user, "ts": V.ts}).filter(pl.col("user_id") > 0)
j = fm.join(o, on="user_id").filter(~((pl.col("sid") == 1) & (pl.col("ofidx") == pl.col("fidx")))).with_columns((pl.col("ots") - pl.col("ts")).alias("dt"))
nxt = j.filter(pl.col("dt") > 0).sort(["fidx", "dt"]).group_by("fidx", maintain_order=True).agg(pl.col("sid").first(), pl.col("ofidx").first(), pl.col("dt").first())
prv = j.filter(pl.col("dt") < 0).sort(["fidx", "dt"], descending=[False, True]).group_by("fidx", maintain_order=True).agg(pl.col("sid").first(), pl.col("ofidx").first(), pl.col("dt").first())
fx = nxt["fidx"].to_numpy(); nsid = nxt["sid"].to_numpy(); nof = nxt["ofidx"].to_numpy()
ue_next = np.zeros((V.n, 128), np.float32); has_next = np.zeros(V.n, bool)
for i, s_, of in zip(fx, nsid, nof):
    ue_next[i] = ue["val" if s_ == 1 else "test"][of]; has_next[i] = True
ue_own = ue["val"]


def cos_scores(uemb):
    m = cei >= 0
    e = np.zeros((V.N, 128), np.float32); e[m] = emb[cei[m]].astype(np.float32)
    u = uemb[V.fid]
    num = (e * u).sum(1); den = np.linalg.norm(e, axis=1) * np.linalg.norm(u, axis=1) + 1e-6
    return num / den


cos_own = cos_scores(ue_own); cos_next = cos_scores(ue_next)
rows = has_next[V.fid]
print(f"  feeds with a later feed: {has_next.sum()}; candidate rows {rows.sum()}; corr(cos_own, cos_next) on those rows {np.corrcoef(cos_own[rows], cos_next[rows])[0, 1]:.3f}")
tb = tiebreak(V)
for nm, sc in (("cos(ue_own)", cos_own), ("cos(ue_next)", cos_next)):
    pf = V.ndcg(sc + tb)
    print(f"  single-score NDCG on feeds with later feed: {nm:14s} {pf[has_next].mean():.4f} (G3 there {pf0[has_next].mean():.4f})")
zo, zn = V.z(cos_own), V.z(cos_next)
for w in (0.1, 0.2, 0.3, 0.5, 1.0):
    for nm, zz in (("own", zo), ("next", zn), ("next-own", zn - zo)):
        s = base + w * zz * rows
        pf = V.ndcg(s)
        d, se, z, n = paired(pf, pf0, has_next)
        print(f"    G3 + {w} * z(cos {nm:8s}) on affected feeds: per-feed {d:+.4f} ± {se:.4f} (z {z:+.2f}, n={n}); overall {(pf.mean() - pf0.mean()):+.5f}")
# combined with the item leak
s = base + BIG * leak_any - BIG * lc_any
pfl = V.ndcg(s)
for w in (0.3, 0.5):
    pf = V.ndcg(s + w * (zn - zo) * rows)
    d, se, z, n = paired(pf, pfl, has_next)
    print(f"    on top of item leak: + {w} * z(next-own): per-feed {d:+.4f} ± {se:.4f} (z {z:+.2f}); overall vs leak-only {(pf.mean() - pfl.mean()):+.5f}")
# previous feed's embedding (past info, allowed)
px = prv["fidx"].to_numpy(); psid = prv["sid"].to_numpy(); pof = prv["ofidx"].to_numpy()
ue_prev = np.zeros((V.n, 128), np.float32); has_prev = np.zeros(V.n, bool)
for i, s_, of in zip(px, psid, pof):
    ue_prev[i] = ue["val" if s_ == 1 else "test"][of]; has_prev[i] = True
cos_prev = cos_scores(ue_prev); zp = V.z(cos_prev); rp = has_prev[V.fid]
for w in (0.2, 0.5):
    pf = V.ndcg(base + w * (zp - zo) * rp)
    d, se, z, n = paired(pf, pf0, has_prev)
    print(f"    PAST-ONLY: G3 + {w} * z(cos prev - own) on feeds with an earlier val/test feed: per-feed {d:+.4f} ± {se:.4f} (z {z:+.2f}, n={n})")
log("done 01")
