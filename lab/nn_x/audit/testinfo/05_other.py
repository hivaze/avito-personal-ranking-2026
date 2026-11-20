"""(5) Other channels: (a) val LABELS as an item signal for test (the item was a val positive / val candidate), simulated on val
with user-grouped OOF; (b) the same user's EARLIER feeds (past info): earlier own-feed history / candidates; (c) item
embedding drift between split tables (edited listing proxy); (d) item in own history? (0 by construction); (e) same-ts feeds."""
from lib import *

V = Split("val"); T = Split("test"); TR = feeds("train")
base = g3(V); pf0 = V.ndcg(base); log(f"G3 val {pf0.mean():.5f}")
y = V.y

# ---------- (a) val labels as item target encoding ----------
print("\n=== (a) val labels as item-level signal: 'item is a positive in another val feed (other user)' ===")
folds = user_group_folds(V, 5, 0)
fold_of_feed = np.empty(V.n, np.int64)
for k, (tr, te) in enumerate(folds):
    fold_of_feed[te] = k
fold_row = fold_of_feed[V.fid]
oof_pos = np.zeros(V.N, np.float32); oof_cnt = np.zeros(V.N, np.float32)
for k in range(5):
    m = fold_row != k
    d = pl.DataFrame({"item_id": V.item[m], "y": y[m]}).group_by("item_id").agg(pl.col("y").sum().alias("p"), pl.len().alias("c"))
    q = pl.DataFrame({"item_id": V.item[fold_row == k]}).join(d, on="item_id", how="left").fill_null(0)
    oof_pos[fold_row == k] = q["p"].to_numpy(); oof_cnt[fold_row == k] = q["c"].to_numpy()
print(f"  val rows whose item is a positive elsewhere (OOF, 4/5 of val): {np.mean(oof_pos > 0):.4f}; pos rate there {y[oof_pos > 0].mean():.4f} vs {y[oof_pos <= 0].mean():.4f}; "
      f"rows whose item is a candidate elsewhere {np.mean(oof_cnt > 0):.4f}")
nested_blend(V, base, oof_pos, pf0, label="val-positive count (OOF)")
nested_blend(V, base, oof_pos / (oof_cnt + 2), pf0, label="val smoothed CTR (OOF)")
# combined with train-label TE already used (it_tr_pos)
F = {"g3": base, "trpos": np.log1p(feat("val", "it_tr_pos")), "trctr": feat("val", "it_tr_ctr"), "trc": np.log1p(feat("val", "it_cand_cnt_train")),
     "luo": np.log1p(feat("val", "it_hist_all_luo")), "M": np.log(V.M[V.fid]).astype(np.float32),
     "vpos": np.log1p(oof_pos), "vcnt": np.log1p(oof_cnt), "vctr": oof_pos / (oof_cnt + 2)}
lgb_residual(V, F, pf0, ["g3", "trpos", "trctr", "trc", "luo", "M"], ["vpos", "vcnt", "vctr"], label="val-label TE (OOF, 4/5 density)")
# coverage on test: item was a val positive / val candidate
vp = pl.DataFrame({"item_id": V.item, "y": y}).group_by("item_id").agg(pl.col("y").sum().alias("p"), pl.len().alias("c"))
q = pl.DataFrame({"item_id": T.item}).join(vp, on="item_id", how="left").fill_null(0)
tp = q["p"].to_numpy(); tc = q["c"].to_numpy()
trp = feat("test", "it_tr_pos")
for mk, mm in (("public", T.public), ("private", ~T.public)):
    r = mm[T.fid]
    print(f"  test/{mk:8s}: rows whose item is a val positive {np.mean(tp[r] > 0):.4f} (feeds {np.mean(np.bincount(T.fid, tp > 0, T.n)[mm] > 0):.4f}); val candidate {np.mean(tc[r] > 0):.4f}; "
          f"train positive {np.mean(trp[r] > 0):.4f}; val positive AND not train positive {np.mean((tp[r] > 0) & (trp[r] <= 0)):.4f}")
print(f"  val (OOF 4/5) for scale: rows whose item is a val positive {np.mean(oof_pos > 0):.4f}, train positive {np.mean(feat('val', 'it_tr_pos') > 0):.4f}")

# ---------- (b) earlier own feeds (past info) ----------
print("\n=== (b) same user's EARLIER feeds (train/val): past-only, allowed ===")
L = pl.read_parquet(f"{OUT}/out_link_val.parquet")
eh = L["eh_n"].fill_null(0).to_numpy().astype(np.float32); ehd = L["eh_dt"].to_numpy(); eht = L["eh_type"].to_numpy()
ec = L["ec_n"].fill_null(0).to_numpy().astype(np.float32)
print(f"  earlier_hist rows {np.mean(eh > 0):.5f} (pos rate {y[eh > 0].mean():.4f}); earlier_cand rows {np.mean(ec > 0):.5f} (pos rate {y[ec > 0].mean():.4f})")
for nm, s in (("earlier_hist", eh), ("-earlier_hist", -eh), ("earlier_cand", ec), ("-earlier_cand", -ec)):
    nested_blend(V, base, s, pf0, label=nm)
# earlier own-feed labels (train): item was a train positive of the SAME user => cannot be a candidate again (filtered); check
tr_pos_same_user = 0
Lt = pl.read_parquet(f"{OUT}/out_link_test.parquet")
for mk, mm in (("public", T.public), ("private", ~T.public)):
    r = mm[T.fid]
    print(f"  test/{mk:8s}: earlier_hist rows {np.mean(Lt['eh_n'].fill_null(0).to_numpy()[r] > 0):.5f}; earlier_cand rows {np.mean(Lt['ec_n'].fill_null(0).to_numpy()[r] > 0):.5f}")

# ---------- (c) embedding drift between split tables ----------
print("\n=== (c) item embedding differs between the train table and the val table (edited listing proxy) ===")
emb_v = np.memmap(f"{ROOT}/data/competition/val/item_embeddings/data.memmap", dtype=np.int8, mode="r", shape=(2516711, 128))
emb_tr = np.memmap(f"{ROOT}/data/competition/train/item_embeddings/data.memmap", dtype=np.int8, mode="r", shape=(81557944, 128))
cei = np.asarray(arr("val", "cand_emb_idx")).astype(np.int64)
# map val candidate items -> train emb idx via train candidate arrays (chunked)
u_items = np.unique(V.item)
tr_idx = np.full(len(u_items), -1, np.int64)
ci_tr = arr("train", "cand_item_id"); ce_tr = arr("train", "cand_emb_idx")
CH = 20_000_000
for s0 in range(0, len(ci_tr), CH):
    it = np.asarray(ci_tr[s0:s0 + CH]); ie = np.asarray(ce_tr[s0:s0 + CH])
    m = np.isin(it, u_items)
    if m.any():
        p = np.searchsorted(u_items, it[m]); tr_idx[p] = ie[m]
hi_tr = arr("train", "hist_item_id"); he_tr = arr("train", "hist_emb_idx")
for s0 in range(0, len(hi_tr), CH):
    it = np.asarray(hi_tr[s0:s0 + CH]); ie = np.asarray(he_tr[s0:s0 + CH])
    m = np.isin(it, u_items) & (ie >= 0)
    if m.any():
        p = np.searchsorted(u_items, it[m]); tr_idx[p] = np.where(tr_idx[p] >= 0, tr_idx[p], ie[m])
log(f"  val candidate items with a train embedding: {np.mean(tr_idx >= 0):.4f}")
pos = np.searchsorted(u_items, V.item); row_tr = tr_idx[pos]
m = (row_tr >= 0) & (cei >= 0)
cos = np.full(V.N, np.nan)
sel = np.flatnonzero(m)
for s0 in range(0, len(sel), 200000):
    ii = sel[s0:s0 + 200000]
    a = emb_v[cei[ii]].astype(np.float32); b = emb_tr[row_tr[ii]].astype(np.float32)
    cos[ii] = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-6)
changed = m & (cos < 0.9999)
print(f"  rows with train embedding {m.mean():.4f}; identical {np.mean(cos[m] >= 0.9999):.4f}; changed (cos<0.9999) {changed.mean():.4f} of all rows; "
      f"pos rate changed {y[changed].mean():.4f} vs identical {y[m & ~changed].mean():.4f} vs no train emb {y[~m].mean():.4f}")
drift = np.where(m, 1 - np.nan_to_num(cos, nan=1.0), 0.0)
nested_blend(V, base, drift, pf0, label="embedding drift (1-cos)")
nested_blend(V, base, changed.astype(np.float64), pf0, label="embedding changed flag")
nested_blend(V, base, m.astype(np.float64), pf0, label="has train embedding (warm flag)")
F2 = {"g3": base, "luo": F["luo"], "trc": F["trc"], "M": F["M"], "warm": m.astype(np.float32), "drift": drift.astype(np.float32), "changed": changed.astype(np.float32),
      "f4": np.asarray(arr("val", "cand_f4")).astype(np.float32)}
lgb_residual(V, F2, pf0, ["g3", "luo", "trc", "M", "warm", "f4"], ["drift", "changed"], label="embedding drift")

# ---------- (d) sanity: candidate in own history ----------
own = pl.DataFrame({"f": V.fid.astype(np.int32), "item_id": V.item}).join(pl.DataFrame({"f": V.hfid.astype(np.int32), "item_id": V.hitem}).unique(), on=["f", "item_id"])
print(f"\n=== (d) candidates present in their own feed's history: {len(own)} (val) ===")

# ---------- (e) feeds sharing a timestamp tick / same user same tick ----------
print("\n=== (e) same-tick structure ===")
for nm, S in (("val", V), ("test", T)):
    d = pl.DataFrame({"ts": S.ts, "u": S.user}).filter(pl.col("u") > 0).group_by(["ts", "u"]).len()
    print(f"  {nm}: (user, tick) groups with >1 feed: {int((d['len'] > 1).sum())}")
log("done 05")
