"""(1b) Extension of the same-user later-feed leak: the later feed's NEW history events (items not in the earlier feed's
history) reveal the user's post-feed intent even when the exact candidate is not among them (recall of the exact-item flag
is 77% on val). Score A's candidates by category match / embedding similarity to B's new events, on top of G3 and on top
of the exact-item leak. Val feeds with a later feed in val/test (117)."""
from lib import *

V = Split("val"); T = Split("test")
base = g3(V); pf0 = V.ndcg(base); log(f"G3 val {pf0.mean():.5f}")
y = V.y
FL = pl.read_parquet(f"{OUT}/out_feedlinks_val.parquet")
L = pl.read_parquet(f"{OUT}/out_link_val.parquet")
leak_any = L["lh_dt"].is_not_null().to_numpy(); lc_any = L["lc_n"].is_not_null().to_numpy()
EMB = {"val": np.memmap(f"{ROOT}/data/competition/val/item_embeddings/data.memmap", dtype=np.int8, mode="r", shape=(2516711, 128)),
       "test": np.memmap(f"{ROOT}/data/competition/test/item_embeddings/data.memmap", dtype=np.int8, mode="r", shape=(4710948, 128))}
HE = {"val": np.asarray(arr("val", "hist_emb_idx")), "test": np.asarray(arr("test", "hist_emb_idx"))}
HC = {"val": np.asarray(arr("val", "hist_category_id")), "test": np.asarray(arr("test", "hist_category_id"))}
HT = {"val": np.asarray(arr("val", "hist_event_type")), "test": np.asarray(arr("test", "hist_event_type"))}
HS = {"val": np.asarray(arr("val", "hist_event_source")), "test": np.asarray(arr("test", "hist_event_source"))}
S = {"val": V, "test": T}
cei = np.asarray(arr("val", "cand_emb_idx")); ccat = np.asarray(arr("val", "cand_category_id"))

# next feed per val feed (val or test), via same construction as 01
meta = pl.concat([V.f.select("fidx", "user_id", "timestamp").with_columns(pl.lit(1).cast(pl.Int8).alias("sid")),
                  T.f.select("fidx", "user_id", "timestamp").with_columns(pl.lit(2).cast(pl.Int8).alias("sid"))])
fm = pl.DataFrame({"fidx": np.arange(V.n, dtype=np.int32), "user_id": V.user, "ts": V.ts}).filter(pl.col("user_id") > 0)
j = fm.join(meta.rename({"fidx": "ofidx", "timestamp": "ots"}), on="user_id").filter(~((pl.col("sid") == 1) & (pl.col("ofidx") == pl.col("fidx"))))
j = j.with_columns((pl.col("ots").cast(pl.Int64) - pl.col("ts")).alias("dt")).filter(pl.col("dt") > 0).sort(["fidx", "dt"])
nxt = j.group_by("fidx", maintain_order=True).agg(pl.col("sid").first(), pl.col("ofidx").first(), pl.col("dt").first())
print(f"val feeds with a later feed: {len(nxt)}")

cat_new = np.zeros(V.N); emb_new_max = np.zeros(V.N); emb_new_mean = np.zeros(V.N); cat_new_fc = np.zeros(V.N); emb_new_fc_max = np.zeros(V.N)
has_next = np.zeros(V.n, bool); n_new = np.zeros(V.n, int)
for fx, sid, of, dt in zip(nxt["fidx"].to_numpy(), nxt["sid"].to_numpy(), nxt["ofidx"].to_numpy(), nxt["dt"].to_numpy()):
    sp = "val" if sid == 1 else "test"; B = S[sp]
    hb = slice(B.hoff[of], B.hoff[of] + B.H[of]); ha = slice(V.hoff[fx], V.hoff[fx] + V.H[fx])
    items_b = B.hitem[hb]; new = ~np.isin(items_b, V.hitem[ha])
    # new events = B's history items absent from A's history (includes A's own positives; we also test excluding exact candidates)
    cs = slice(V.off[fx], V.off[fx] + V.M[fx])
    not_cand = ~np.isin(items_b, V.item[cs])
    sel = new & not_cand  # intent signal that is NOT the exact-item leak
    has_next[fx] = True; n_new[fx] = sel.sum()
    if sel.sum() == 0:
        continue
    cats = HC[sp][hb][sel]; fc = (HT[sp][hb][sel] == 1) & (HS[sp][hb][sel] == 1)
    ei = HE[sp][hb][sel]; ok = ei >= 0
    cc = ccat[cs]
    cat_new[cs] = np.isin(cc, cats); cat_new_fc[cs] = np.isin(cc, cats[fc])
    if ok.sum():
        E = EMB[sp][ei[ok]].astype(np.float32); E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-6
        ce = cei[cs]; cok = ce >= 0
        C = np.zeros((V.M[fx], 128), np.float32); C[cok] = EMB["val"][ce[cok]].astype(np.float32); C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-6
        sim = C @ E.T
        emb_new_max[cs] = sim.max(1); emb_new_mean[cs] = sim.mean(1)
        if (fc & ok).sum():
            Ef = EMB[sp][ei[fc & ok]].astype(np.float32); Ef /= np.linalg.norm(Ef, axis=1, keepdims=True) + 1e-6
            emb_new_fc_max[cs] = (C @ Ef.T).max(1)
rows = has_next[V.fid]
print(f"feeds with later feed {has_next.sum()}, new non-candidate events per feed: median {np.median(n_new[has_next]):.0f}, mean {n_new[has_next].mean():.1f}, zero {np.mean(n_new[has_next] == 0):.2f}")
print(f"on affected rows: pos rate cat_new=1 {y[rows & (cat_new > 0)].mean():.4f} vs 0 {y[rows & (cat_new == 0)].mean():.4f}; cat_new_fc=1 {y[rows & (cat_new_fc > 0)].mean():.4f}")
# own-history baseline for the same quantities (what a legit model already has): category in A's own recent history
own_cat = np.zeros(V.N)
for fx in np.flatnonzero(has_next):
    ha = slice(V.hoff[fx], V.hoff[fx] + min(V.H[fx], 32)); cs = slice(V.off[fx], V.off[fx] + V.M[fx])
    own_cat[cs] = np.isin(ccat[cs], HC["val"][ha])
print(f"on affected rows: pos rate own_cat(last32)=1 {y[rows & (own_cat > 0)].mean():.4f} vs 0 {y[rows & (own_cat == 0)].mean():.4f}")

print("\n=== gains on affected feeds (per-feed, paired) ===")
leak = base + BIG * leak_any - BIG * lc_any
pfl = V.ndcg(leak)
print(f"  exact-item leak on affected feeds: {pf0[has_next].mean():.4f} -> {pfl[has_next].mean():.4f}")
# unflagged positives in affected feeds
pos_rows = rows & (y > 0)
print(f"  positives in affected feeds {int(pos_rows.sum())}, flagged by exact-item {int((pos_rows & leak_any).sum())}; "
      f"of the unflagged, cat_new=1 for {int((pos_rows & ~leak_any & (cat_new > 0)).sum())}, emb_new_max>0.8 for {int((pos_rows & ~leak_any & (emb_new_max > 0.8)).sum())}")
for nm, s in (("cat_new", cat_new), ("cat_new_fc", cat_new_fc), ("emb_new_max", emb_new_max), ("emb_new_fc_max", emb_new_fc_max), ("emb_new_mean", emb_new_mean)):
    zz = V.z(s) * rows
    for w in (0.2, 0.5, 1.0, 2.0):
        pa = V.ndcg(base + w * zz); pb = V.ndcg(leak + w * zz)
        d1, s1, z1, n1 = paired(pa, pf0, has_next); d2, s2, z2, _ = paired(pb, pfl, has_next)
        print(f"  {nm:14s} w={w:3.1f}: on G3 per-feed {d1:+.4f} ± {s1:.4f} (z {z1:+.2f}); on top of exact leak {d2:+.4f} ± {s2:.4f} (z {z2:+.2f}); overall vs G3 {(pa.mean() - pf0.mean()):+.5f} / vs leak {(pb.mean() - pfl.mean()):+.5f}")
log("done 07")
