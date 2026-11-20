"""Re-run the GBDT residual tests of 02/03/05 with the init-score harness (boosting starts from G3, so the control model
A reproduces G3 instead of losing 0.009 to binning). Also cold-only views for the cross-user signals."""
from lib import *
import importlib.util

V = Split("val"); T = Split("test")
base = g3(V); pf0 = V.ndcg(base); log(f"G3 val {pf0.mean():.5f}")
y = V.y
HOUR = 3600.0


# ---- cross-user signals (copied from 02) ----
def hist_table(S, sid):
    df = pl.DataFrame({"sid": np.full(len(S.hitem), sid, np.int8), "hfidx": S.hfid.astype(np.int32), "huser": S.user[S.hfid],
                       "hts": S.ts[S.hfid], "item_id": S.hitem,
                       "etype": np.asarray(arr(S.split, "hist_event_type")), "esrc": np.asarray(arr(S.split, "hist_event_source")),
                       "hpos": np.asarray(arr(S.split, "hist_hpos")).astype(np.int16)})
    df = df.with_columns(((pl.col("etype") == 1) & (pl.col("esrc") == 1)).alias("fc"), (pl.col("etype") == 1).alias("t1"))
    return df.group_by(["sid", "hfidx", "huser", "hts", "item_id"]).agg(pl.col("fc").any(), pl.col("t1").any(), pl.col("hpos").min())


def signals(S, sid, HT, tag):
    b = pl.DataFrame({"row": np.arange(S.N, dtype=np.int64), "fidx": S.fid.astype(np.int32), "user_id": S.user[S.fid], "ts": S.ts[S.fid], "item_id": S.item})
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
        (pl.col("fc") & (pl.col("hpos") < 8)).sum().alias(f"{tag}_fc_hpos8"))
    out = pl.DataFrame({"row": np.arange(S.N, dtype=np.int64)}).join(agg, on="row", how="left").fill_null(0).sort("row")
    return {c: out[c].to_numpy().astype(np.float32) for c in out.columns if c != "row"}


HV = hist_table(V, 1); HTt = hist_table(T, 2)
sig_v = signals(V, 1, HV, "w"); sig_vt = signals(V, 1, pl.concat([HV, HTt]), "vt")
luo = feat("val", "it_hist_all_luo"); luo1 = feat("val", "it_hist_et1_luo"); trc = feat("val", "it_cand_cnt_train")
train_hist_cnt = luo - sig_vt["vt_all"]
cold_tr = (trc == 0) & (train_hist_cnt <= 0.5)
cold_feed = np.bincount(V.fid, cold_tr, V.n) / V.M >= 0.5
masks = {"cold feeds (>=50% cold rows)": cold_feed, "warm feeds": ~cold_feed, "M<50": V.M < 50, "M>=50": V.M >= 50}
F = {"g3": base, "luo": np.log1p(luo), "luo1": np.log1p(luo1), "trc": np.log1p(trc), "cold": cold_tr.astype(np.float32),
     "M": np.log(V.M[V.fid]).astype(np.float32), "seqpop": np.log1p(feat("val", "seq_cv_pop")),
     "f4": np.asarray(arr("val", "cand_f4")).astype(np.float32)}
for k, v in list(sig_v.items()) + list(sig_vt.items()):
    F[k] = np.log1p(v)
    F[k + "_cold"] = np.log1p(v) * cold_tr

print("\n=== (2) cross-user signals, init-score residual harness ===")
BASE = ["luo", "luo1", "trc", "cold", "M", "seqpop"]
lgb_resid_init(V, F, pf0, BASE, [], base, label="control (A only)", masks=None)
W_ALL = ["w_all", "w_fc", "w_t1", "w_later", "w_later_fc", "w_earlier", "w_earlier_fc", "w_rec24", "w_rec72", "w_rec24_fc", "w_fc_hpos8", "w_later24", "w_earlier24"]
lgb_resid_init(V, F, pf0, BASE, W_ALL, base, label="val-window all", masks=masks)
lgb_resid_init(V, F, pf0, BASE, [w + "_cold" for w in W_ALL], base, label="val-window all, cold rows only", masks=masks)
lgb_resid_init(V, F, pf0, BASE, ["w_earlier", "w_earlier_fc", "w_earlier24"], base, label="val-window earlier-only (past)", masks=masks)
VT = ["vt_all", "vt_fc", "vt_later", "vt_later_fc", "vt_earlier", "vt_earlier_fc", "vt_rec24", "vt_rec24_fc", "vt_later48_fc"]
lgb_resid_init(V, F, pf0, BASE, VT, base, label="val+test windows (optimistic)", masks=masks)
# positive-level diagnostics on cold rows: does the window count separate positives among cold candidates?
from sklearn.metrics import roc_auc_score
for nm in ("w_all", "w_fc", "w_later_fc", "vt_all", "vt_later_fc"):
    s = F[nm]
    print(f"  cold-row AUC of {nm:12s}: {roc_auc_score(y[cold_tr], s[cold_tr]):.4f} (nonzero share {np.mean(s[cold_tr] > 0):.4f}); warm-row AUC {roc_auc_score(y[~cold_tr], s[~cold_tr]):.4f}; "
          f"G3 AUC cold {roc_auc_score(y[cold_tr], base[cold_tr]):.4f}")
# within-feed (grouped) view: nested cold-only blend
for nm in ("w_fc", "w_later_fc", "vt_later_fc"):
    nested_blend(V, base, F[nm + "_cold"], pf0, label=f"{nm} on cold rows only")

print("\n=== (3) co-candidacy / candidate frequency / cohesion, init-score harness ===")
cc = np.load(f"{OUT}/out_cocand_val_v.npy")
keys = sorted(["cnt_other", "v_cc_ov", "v_cc_rec", "v_cc_fc", "v_cc_nfeeds", "v_cc_maxov", "v_coh", "v_coh5"])
for i, k_ in enumerate(keys):
    F[k_] = np.log1p(cc[:, i])
F["ccs"] = feat("val", "it_cand_cnt_split")
lgb_resid_init(V, F, pf0, BASE + ["ccs"], ["v_cc_ov", "v_cc_rec", "v_cc_fc", "v_cc_nfeeds", "v_cc_maxov"], base, label="co-candidacy (val feeds)", masks=masks)
lgb_resid_init(V, F, pf0, BASE + ["ccs"], ["cnt_other", "v_coh", "v_coh5"], base, label="cand freq + cohesion", masks=masks)
lgb_resid_init(V, F, pf0, BASE, ["ccs"], base, label="it_cand_cnt_split alone (already a feature file)", masks=None)

print("\n=== (5a) val labels as item TE (OOF), init-score harness ===")
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
F["vpos"] = np.log1p(oof_pos); F["vcnt"] = np.log1p(oof_cnt); F["vctr"] = oof_pos / (oof_cnt + 2)
F["trpos"] = np.log1p(feat("val", "it_tr_pos")); F["trctr"] = feat("val", "it_tr_ctr")
lgb_resid_init(V, F, pf0, BASE + ["trpos", "trctr"], ["vpos", "vcnt", "vctr"], base, label="val-label TE", masks=masks)
# direct push: items that are val positives elsewhere (OOF) to the top, with weight
for w in (0.1, 0.3, 1.0):
    pf = V.ndcg(base + w * (oof_pos > 0))
    print(f"  G3 + {w} * [item is a val positive elsewhere]: {fmt(pf, pf0)}")
log("done 06")
