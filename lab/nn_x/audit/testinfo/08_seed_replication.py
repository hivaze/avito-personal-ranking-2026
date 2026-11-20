"""Seed replication (fold assignment + LightGBM seed) of the borderline init-score residual results of 06."""
from lib import *
import importlib

V = Split("val"); T = Split("test")
base = g3(V); pf0 = V.ndcg(base); log(f"G3 val {pf0.mean():.5f}")
y = V.y
HV = hist_table(V, 1); HTt = hist_table(T, 2)
sig_v = signals(V, 1, HV, "w"); sig_vt = signals(V, 1, pl.concat([HV, HTt]), "vt")
luo = feat("val", "it_hist_all_luo"); luo1 = feat("val", "it_hist_et1_luo"); trc = feat("val", "it_cand_cnt_train")
cold_tr = (trc == 0) & (luo - sig_vt["vt_all"] <= 0.5)
cold_feed = np.bincount(V.fid, cold_tr, V.n) / V.M >= 0.5
masks = {"cold feeds": cold_feed, "warm feeds": ~cold_feed, "M<50": V.M < 50, "M>=50": V.M >= 50}
F = {"luo": np.log1p(luo), "luo1": np.log1p(luo1), "trc": np.log1p(trc), "cold": cold_tr.astype(np.float32),
     "M": np.log(V.M[V.fid]).astype(np.float32), "seqpop": np.log1p(feat("val", "seq_cv_pop")), "ccs": feat("val", "it_cand_cnt_split"),
     "trpos": np.log1p(feat("val", "it_tr_pos")), "trctr": feat("val", "it_tr_ctr")}
for k, v in list(sig_v.items()) + list(sig_vt.items()):
    F[k] = np.log1p(v)
cc = np.load(f"{OUT}/out_cocand_val_v.npy")
for i, k_ in enumerate(sorted(["cnt_other", "v_cc_ov", "v_cc_rec", "v_cc_fc", "v_cc_nfeeds", "v_cc_maxov", "v_coh", "v_coh5"])):
    F[k_] = np.log1p(cc[:, i])
BASE = ["luo", "luo1", "trc", "cold", "M", "seqpop"]
W_ALL = ["w_all", "w_fc", "w_t1", "w_later", "w_later_fc", "w_earlier", "w_earlier_fc", "w_rec24", "w_rec72", "w_rec24_fc", "w_fc_hpos8", "w_later24", "w_earlier24"]
CONF = {"val-window earlier-only (past)": (BASE, ["w_earlier", "w_earlier_fc", "w_earlier24"]),
        "val-window all": (BASE, W_ALL),
        "co-candidacy (val feeds)": (BASE + ["ccs"], ["v_cc_ov", "v_cc_rec", "v_cc_fc", "v_cc_nfeeds", "v_cc_maxov"]),
        "cand freq + cohesion": (BASE + ["ccs"], ["cnt_other", "v_coh", "v_coh5"])}
# val-label TE needs the OOF built per seed (fold-dependent)
summary = {}
for seed in (1, 2):
    print(f"\n######## seed {seed} ########")
    for lab, (b, a) in CONF.items():
        pa, pb = lgb_resid_init(V, F, pf0, b, a, base, seed=seed, label=f"s{seed} {lab}", masks=masks)
        summary.setdefault(lab, []).append(paired(pb, pa)[:2])
    folds = user_group_folds(V, 5, seed)
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
    pa, pb = lgb_resid_init(V, F, pf0, BASE + ["trpos", "trctr"], ["vpos", "vcnt", "vctr"], base, seed=seed, label=f"s{seed} val-label TE", masks=masks)
    summary.setdefault("val-label TE", []).append(paired(pb, pa)[:2])
print("\n=== B-A per seed (seeds 1,2; seed 0 in 06 log) ===")
for lab, v in summary.items():
    print(f"  {lab:36s} " + "  ".join(f"{d:+.5f}±{s:.5f}" for d, s in v) + f"  mean {np.mean([d for d, s in v]):+.5f}")
log("done 08")
