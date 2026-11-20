"""(2) Cross-user test-period signals: the candidate item appears in OTHER users' val/test histories.
What it_hist_*_luo already encodes: counts over ALL histories of all splits (train+val+test) minus the same user's later
snapshots. Here we build window-restricted and recency-weighted variants and measure them as a residual on G3
(nested, user-grouped). Realistic analogue for test = only same-split (val) histories; optimistic = val+test histories."""
from lib import *

V = Split("val"); T = Split("test")
base = g3(V); pf0 = V.ndcg(base); log(f"G3 val {pf0.mean():.5f}")
HOUR = 3600.0


def hist_table(S, sid):
    df = pl.DataFrame({"sid": np.full(len(S.hitem), sid, np.int8), "hfidx": S.hfid.astype(np.int32), "huser": S.user[S.hfid],
                       "hts": S.ts[S.hfid], "item_id": S.hitem,
                       "etype": np.asarray(arr(S.split, "hist_event_type")), "esrc": np.asarray(arr(S.split, "hist_event_source")),
                       "hpos": np.asarray(arr(S.split, "hist_hpos")).astype(np.int16)})
    df = df.with_columns(((pl.col("etype") == 1) & (pl.col("esrc") == 1)).alias("fc"), (pl.col("etype") == 1).alias("t1"))
    # one row per (feed, item): any event / any feed-click / newest position
    return df.group_by(["sid", "hfidx", "huser", "hts", "item_id"]).agg(pl.col("fc").any(), pl.col("t1").any(), pl.col("hpos").min())


def signals(S, sid, HT, tag):
    """counts of OTHER users' history feeds (in HT) containing the candidate item, split by time and event type."""
    b = pl.DataFrame({"row": np.arange(S.N, dtype=np.int64), "fidx": S.fid.astype(np.int32), "user_id": S.user[S.fid], "ts": S.ts[S.fid],
                      "item_id": S.item})
    j = b.join(HT, on="item_id").filter(~((pl.col("sid") == sid) & (pl.col("hfidx") == pl.col("fidx"))))
    j = j.filter((pl.col("huser") != pl.col("user_id")) | (pl.col("user_id") == 0))
    j = j.with_columns(((pl.col("hts") - pl.col("ts")) / HOUR).alias("dth"))
    later = pl.col("dth") > 0
    agg = j.group_by("row").agg(
        pl.len().alias(f"{tag}_all"), pl.col("fc").sum().alias(f"{tag}_fc"), pl.col("t1").sum().alias(f"{tag}_t1"),
        later.sum().alias(f"{tag}_later"), (later & pl.col("fc")).sum().alias(f"{tag}_later_fc"),
        (~later).sum().alias(f"{tag}_earlier"), (~later & pl.col("fc")).sum().alias(f"{tag}_earlier_fc"),
        (later & (pl.col("dth") <= 24)).sum().alias(f"{tag}_later24"), (later & (pl.col("dth") <= 48) & pl.col("fc")).sum().alias(f"{tag}_later48_fc"),
        (~later & (pl.col("dth") >= -24)).sum().alias(f"{tag}_earlier24"),
        (-pl.col("dth").abs() / 24.0).exp().sum().alias(f"{tag}_rec24"), (-pl.col("dth").abs() / 72.0).exp().sum().alias(f"{tag}_rec72"),
        ((-pl.col("dth").abs() / 24.0).exp() * pl.col("fc")).sum().alias(f"{tag}_rec24_fc"),
        # newest-position feed clicks: the item was clicked very recently by the other user
        (pl.col("fc") & (pl.col("hpos") < 8)).sum().alias(f"{tag}_fc_hpos8"))
    out = pl.DataFrame({"row": np.arange(S.N, dtype=np.int64)}).join(agg, on="row", how="left").fill_null(0).sort("row")
    return {c: out[c].to_numpy().astype(np.float32) for c in out.columns if c != "row"}


HV = hist_table(V, 1); HTt = hist_table(T, 2)
log("hist tables", len(HV), len(HTt))
sig_v = signals(V, 1, HV, "w")                       # val-window only (realistic for test)
sig_vt = signals(V, 1, pl.concat([HV, HTt]), "vt")   # val+test windows (optimistic)
sig_t = signals(T, 2, HTt, "w")                      # test-window only, for coverage
log("signals built")

luo = feat("val", "it_hist_all_luo"); luo1 = feat("val", "it_hist_et1_luo"); trc = feat("val", "it_cand_cnt_train")
seqpop = feat("val", "seq_cv_pop") if os.path.exists(f"{FEATS}/val/seq_cv_pop.npy") else np.zeros(V.N, np.float32)
cold = (trc == 0) & (luo - sig_vt["vt_all"] <= 0)  # never in train candidates and no train-history count (approx: luo minus val/test window)
# better cold definition: item not seen in train at all = it_hist_all_luo == count from val+test windows and train cand count 0
train_hist_cnt = luo - sig_vt["vt_all"]
cold_tr = (trc == 0) & (train_hist_cnt <= 0.5)
y = V.y
print(f"\nval rows {V.N}; cold (not in train cand/hist) {cold_tr.mean():.3f} rows; positive rate cold {y[cold_tr].mean():.4f} warm {y[~cold_tr].mean():.4f}")
print("check: it_hist_all_luo - (val+test other-user window count) < 0 on", (train_hist_cnt < -0.5).sum(), "rows (same-user earlier snapshots etc.)")

print("\n=== Coverage (share of candidate rows with nonzero signal) ===")
for nm, s in (("w_all (val window)", sig_v["w_all"]), ("w_fc", sig_v["w_fc"]), ("w_later", sig_v["w_later"]), ("w_earlier", sig_v["w_earlier"]),
              ("vt_all (val+test)", sig_vt["vt_all"]), ("vt_later_fc", sig_vt["vt_later_fc"]), ("it_hist_all_luo", luo)):
    print(f"  val {nm:22s} nonzero {np.mean(s > 0):.4f} | pos rate nonzero {y[s > 0].mean():.4f} zero {y[s <= 0].mean():.4f} | cold rows nonzero {np.mean(s[cold_tr] > 0):.4f}")
for mk, mm in (("public", T.public), ("private", ~T.public)):
    r = mm[T.fid]
    print(f"  test/{mk:8s} w_all nonzero {np.mean(sig_t['w_all'][r] > 0):.4f} w_fc {np.mean(sig_t['w_fc'][r] > 0):.4f} w_later {np.mean(sig_t['w_later'][r] > 0):.4f} "
          f"w_earlier {np.mean(sig_t['w_earlier'][r] > 0):.4f} w_later48_fc {np.mean(sig_t['w_later48_fc'][r] > 0):.4f}")

print("\n=== Single-signal value: nested rank blend on G3 (z of log1p(signal)) ===")
F = {"g3": base, "luo": np.log1p(luo), "luo1": np.log1p(luo1), "trc": np.log1p(trc), "cold": cold_tr.astype(np.float32),
     "M": np.log(V.M[V.fid]).astype(np.float32), "seqpop": np.log1p(seqpop)}
for k, v in list(sig_v.items()) + list(sig_vt.items()):
    F[k] = np.log1p(v)
for nm in ("w_all", "w_fc", "w_later", "w_later_fc", "w_earlier", "w_earlier_fc", "w_rec24", "w_rec24_fc", "w_fc_hpos8", "vt_all", "vt_fc", "vt_later_fc", "vt_rec24_fc", "luo", "luo1"):
    nested_blend(V, base, F[nm], pf0, label=nm)

print("\n=== GBDT residual on G3 (user-grouped 5-fold OOF, lambdarank; A = G3 + what we already have, B = A + new) ===")
BASE = ["g3", "luo", "luo1", "trc", "cold", "M", "seqpop"]
runs = {
    "val-window all": ["w_all", "w_fc", "w_t1", "w_later", "w_later_fc", "w_earlier", "w_earlier_fc", "w_rec24", "w_rec72", "w_rec24_fc", "w_fc_hpos8", "w_later24", "w_earlier24"],
    "val-window earlier-only (strictly past)": ["w_earlier", "w_earlier_fc", "w_earlier24"],
    "val+test windows (optimistic)": ["vt_all", "vt_fc", "vt_later", "vt_later_fc", "vt_earlier", "vt_earlier_fc", "vt_rec24", "vt_rec24_fc", "vt_later48_fc"],
}
pfA = None
for lab, add in runs.items():
    a, b = lgb_residual(V, F, pf0, BASE, add, label=lab)
    pfA = a
    for cn, cm in (("cold feeds (>=50% cold rows)", np.bincount(V.fid, cold_tr, V.n) / V.M >= 0.5), ("warm feeds", np.bincount(V.fid, cold_tr, V.n) / V.M < 0.5)):
        print(f"      {lab} B-A on {cn}: {fmt(b, a, cm)}")
    # positives-level view: share of positives that are cold and flagged
np.save(f"{OUT}/out_crossuser_val_wsig.npy", np.stack([sig_v[k] for k in sorted(sig_v)], 1))
log("done 02")
