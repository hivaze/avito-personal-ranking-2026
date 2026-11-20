"""(3) Candidate-set structure within a split: co-candidacy (ranker-implied similarity), candidate frequency,
duplicate / overlapping candidate sets, candidate array position, within-set f-score ranks. Residual on G3 (nested).
Val uses val feeds only (the realistic analogue: on test one would use test feeds). Also val+test pooled for coverage."""
from lib import *

V = Split("val"); T = Split("test")
base = g3(V); pf0 = V.ndcg(base); log(f"G3 val {pf0.mean():.5f}")
y = V.y


def cand_table(S):
    return pl.DataFrame({"cf": S.fid.astype(np.int32), "item_id": S.item, "cuser": S.user[S.fid], "cts": S.ts[S.fid]})


def hist_table(S, maxpos=None):
    hp = np.asarray(arr(S.split, "hist_hpos"))
    df = pl.DataFrame({"hf": S.hfid.astype(np.int32), "item_id": S.hitem, "hpos": hp.astype(np.int16),
                       "fc": (np.asarray(arr(S.split, "hist_event_type")) == 1) & (np.asarray(arr(S.split, "hist_event_source")) == 1)})
    if maxpos is not None:
        df = df.filter(pl.col("hpos") < maxpos)
    return df.group_by(["hf", "item_id"]).agg(pl.col("hpos").min(), pl.col("fc").any())


def cocand_signals(S, C_other, H_self, tag):
    """For candidate c of feed X: sum over other feeds Y (in C_other) with c in cand(Y) of |cand(Y) ∩ hist(X)| (weighted variants).
    Also: number of other feeds containing c (cand freq), and co-candidacy with X's own other candidates."""
    # (X, Y, overlap) : hist items of X that are candidates of Y
    XY = (H_self.rename({"hf": "X"}).join(C_other.rename({"cf": "Y"}), on="item_id")
          .filter(pl.col("X") != pl.col("Y"))
          .group_by(["X", "Y"]).agg(pl.len().alias("ov"), (0.95 ** pl.col("hpos")).sum().alias("ov_rec"), pl.col("fc").sum().alias("ov_fc")))
    log(f"  {tag}: (X,Y) pairs with hist(X)∩cand(Y) != {{}}: {len(XY)}")
    # candidates c of Y, restricted to c in cand(X)
    cx = pl.DataFrame({"row": np.arange(S.N, dtype=np.int64), "X": S.fid.astype(np.int32), "item_id": S.item})
    cy = C_other.rename({"cf": "Y"}).select("Y", "item_id")
    j = XY.join(cy, on="Y").join(cx, on=["X", "item_id"])
    agg = j.group_by("row").agg(pl.col("ov").sum().alias(f"{tag}_cc_ov"), pl.col("ov_rec").sum().alias(f"{tag}_cc_rec"), pl.col("ov_fc").sum().alias(f"{tag}_cc_fc"),
                                pl.len().alias(f"{tag}_cc_nfeeds"), pl.col("ov").max().alias(f"{tag}_cc_maxov"))
    # candidate frequency among other feeds, and co-candidacy with X's own other candidates (set cohesion)
    cnt = cy.group_by("item_id").len().rename({"len": "cnt_other"})
    cx2 = cx.join(cnt, on="item_id", how="left").fill_null(0)
    # pairs (X, Y) sharing >= 1 candidate
    XYc = cx.join(cy, on="item_id").filter(pl.col("X") != pl.col("Y")).group_by(["X", "Y"]).agg(pl.len().alias("share"))
    j2 = XYc.join(cy, on="Y").join(cx, on=["X", "item_id"]).group_by("row").agg((pl.col("share") - 1).sum().alias(f"{tag}_coh"), (pl.col("share") >= 5).sum().alias(f"{tag}_coh5"))
    out = cx2.join(agg, on="row", how="left").join(j2, on="row", how="left").fill_null(0).sort("row")
    feedov = XYc.group_by("X").agg(pl.col("share").max().alias("max_share"), pl.len().alias("n_overlap_feeds"))
    return {c: out[c].to_numpy().astype(np.float32) for c in out.columns if c not in ("row", "X", "item_id")}, XYc, feedov


CV_ = cand_table(V); CT_ = cand_table(T)
HV32 = hist_table(V, 32); HVall = hist_table(V)
sig_v, XYc_v, fov_v = cocand_signals(V, CV_, HVall, "v")       # val feeds only
sig_v32, _, _ = cocand_signals(V, CV_, HV32, "v32")             # last-32 history
sig_vt, _, _ = cocand_signals(V, pl.concat([CV_, CT_.with_columns(pl.col("cf") + 10_000_000)]), HVall, "vt")  # val+test candidate sets
sig_t, XYc_t, fov_t = cocand_signals(T, CT_, hist_table(T), "t")
log("co-candidacy signals built")

print("\n=== Coverage ===")
for nm, s in (("v_cc_ov (val feeds)", sig_v["v_cc_ov"]), ("v32_cc_ov", sig_v32["v32_cc_ov"]), ("vt_cc_ov (val+test feeds)", sig_vt["vt_cc_ov"]),
              ("cnt_other (val)", sig_v["cnt_other"]), ("v_coh", sig_v["v_coh"])):
    print(f"  val {nm:26s} nonzero rows {np.mean(s > 0):.4f} feeds {np.mean(np.bincount(V.fid, s > 0, V.n) > 0):.4f} | pos rate nonzero {y[s > 0].mean():.4f} zero {y[s <= 0].mean():.4f}")
for mk, mm in (("public", T.public), ("private", ~T.public)):
    r = mm[T.fid]
    print(f"  test/{mk:8s} t_cc_ov nonzero rows {np.mean(sig_t['t_cc_ov'][r] > 0):.4f} cnt_other nonzero {np.mean(sig_t['cnt_other'][r] > 0):.4f} t_coh nonzero {np.mean(sig_t['t_coh'][r] > 0):.4f}")

print("\n=== Duplicate / overlapping candidate sets (pairs of feeds sharing candidates) ===")
for nm, S, XYc, fov in (("val", V, XYc_v, fov_v), ("test", T, XYc_t, fov_t)):
    sh = XYc["share"].to_numpy()
    Mx = S.M[XYc["X"].to_numpy()]; My = S.M[XYc["Y"].to_numpy()]
    jac = sh / (Mx + My - sh)
    same_user = (S.user[XYc["X"].to_numpy()] == S.user[XYc["Y"].to_numpy()]) & (S.user[XYc["X"].to_numpy()] > 0)
    print(f"  {nm}: ordered pairs sharing >=1 cand {len(sh)}, >=5 {int((sh >= 5).sum())}, >=20 {int((sh >= 20).sum())}, jaccard>=0.5 {int((jac >= 0.5).sum())}, "
          f"identical sets {int(((sh == Mx) & (sh == My)).sum())}; same-user among share>=5: {int((same_user & (sh >= 5)).sum())}")
    big = XYc.filter(pl.col("share") >= 20).head(5)
    if len(big):
        X = big["X"].to_numpy(); Y = big["Y"].to_numpy()
        print("   examples share>=20: " + "; ".join(f"X{x}(u{S.user[x]},M{S.M[x]},t{S.ts[x]}) Y{yy}(u{S.user[yy]},M{S.M[yy]},t{S.ts[yy]}) share {s_}" for x, yy, s_ in zip(X, Y, big["share"].to_numpy())))
# val<->test shared candidate sets (val labels known => a test feed sharing a val feed's set could read val labels)
jx = (CV_.rename({"cf": "X"}).select("X", "item_id").join(CT_.rename({"cf": "Y"}).select("Y", "item_id"), on="item_id")
      .group_by(["X", "Y"]).agg(pl.len().alias("share")))
sh = jx["share"].to_numpy()
print(f"  val<->test feed pairs sharing >=1 cand {len(sh)}, >=5 {int((sh >= 5).sum())}, >=20 {int((sh >= 20).sum())}; test feeds with a val feed sharing >=5: {jx.filter(pl.col('share') >= 5)['Y'].n_unique()}")

print("\n=== Candidate array position within the feed (common permutation) ===")
pos_in_feed = np.arange(V.N) - np.repeat(V.off, V.M)
tb = tiebreak(V)
rnd = V.ndcg(tb)
print(f"  random order {rnd.mean():.4f}; +pos {V.ndcg(pos_in_feed + tb).mean():.4f}; -pos {V.ndcg(-pos_in_feed + tb).mean():.4f}")
# position relative to M (fraction), and the positive-rate curve by position
for lo, hi in ((0, 5), (5, 20), (20, 50), (50, 100), (100, 200), (200, 300)):
    m = (pos_in_feed >= lo) & (pos_in_feed < hi)
    print(f"    pos [{lo},{hi}): rows {m.sum():7d} pos rate {y[m].mean():.4f}")
nested_blend(V, base, pos_in_feed / np.maximum(V.M[V.fid] - 1, 1), pf0, label="array position (fraction)")
nested_blend(V, base, -(pos_in_feed / np.maximum(V.M[V.fid] - 1, 1)), pf0, label="-array position")

print("\n=== Nested blends on G3 ===")
F = {"g3": base, "M": np.log(V.M[V.fid]).astype(np.float32), "luo": np.log1p(feat("val", "it_hist_all_luo")), "ccs": feat("val", "it_cand_cnt_split"),
     "f1r": V.rank_pct(feat("val", "f1") if os.path.exists(f"{FEATS}/val/f1.npy") else np.asarray(arr("val", "cand_f1")))}
for d_ in (sig_v, sig_v32, sig_vt):
    for k, v in d_.items():
        F[k] = np.log1p(v)
for nm in ("v_cc_ov", "v_cc_rec", "v_cc_fc", "v_cc_nfeeds", "v_cc_maxov", "v32_cc_ov", "v32_cc_rec", "vt_cc_ov", "vt_cc_rec", "cnt_other", "v_coh", "v_coh5"):
    nested_blend(V, base, F[nm], pf0, label=nm)

print("\n=== GBDT residual on G3 ===")
BASE = ["g3", "M", "luo", "ccs"]
for lab, add in {"co-candidacy val feeds": ["v_cc_ov", "v_cc_rec", "v_cc_fc", "v_cc_nfeeds", "v_cc_maxov", "v32_cc_ov", "v32_cc_rec"],
                 "co-candidacy val+test feeds": ["vt_cc_ov", "vt_cc_rec", "vt_cc_fc", "vt_cc_nfeeds", "vt_cc_maxov"],
                 "cand freq + cohesion": ["cnt_other", "v_coh", "v_coh5"]}.items():
    lgb_residual(V, F, pf0, BASE, add, label=lab)
np.save(f"{OUT}/out_cocand_val_v.npy", np.stack([sig_v[k] for k in sorted(sig_v)], 1))
log("done 03")
