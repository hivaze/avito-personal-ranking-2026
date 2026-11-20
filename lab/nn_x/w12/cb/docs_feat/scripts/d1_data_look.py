"""D1: look at the raw objects that could be given to CatBoost natively (no model is trained).
Counts cardinalities and train-pool -> val coverage for item_id / emb_idx / user_id / category_id, row order inside train feeds, timestamp order.
Pool = the cb26 exploration pool (p_random > 0.5, M >= 50, minus the 30K holdout) = 134,118 feeds. Label-free except the positive rate.
"""
import os, sys, json, time
assert os.environ.get("CUDA_VISIBLE_DEVICES", "x") == "", "CPU only"
import numpy as np, polars as pl
sys.path.insert(0, "<repo>/nn_x/cb26"); sys.path.insert(0, "<repo>/src")
from cbdata import hold_index, ROOT  # read-only import
t0 = time.time()
out = {}
hold, ftr = hold_index()
M = ftr["M"].to_numpy().astype(np.int64); off = ftr["cand_off"].to_numpy().astype(np.int64)
ts = ftr["timestamp"].to_numpy(); uid = ftr["user_id"].to_numpy(); H = ftr["H"].to_numpy()
pr = np.load(ROOT / "eda/labels_position/out/train_prandom.npz")
p = np.nan_to_num(pr["p_random"].astype(np.float32), nan=0.0); p[M < 50] = 0.0
sel = np.setdiff1d(np.where((p > 0.5) & (M >= 50))[0], hold)
rows = np.repeat(off[sel] - np.r_[0, np.cumsum(M[sel])[:-1]], M[sel]) + np.arange(M[sel].sum())
out["pool_feeds"] = int(len(sel)); out["pool_rows"] = int(len(rows)); out["holdout_overlap"] = int(np.isin(sel, hold).sum())
out["train_ts_sorted_by_fidx"] = bool((np.diff(ts) >= 0).all()); out["train_ts_distinct"] = int(len(np.unique(ts)))
out["pool_users_distinct"] = int(len(np.unique(uid[sel]))); out["pool_user0_feeds"] = int((uid[sel] == 0).sum())
F = ROOT / "data/flat"
L = lambda sp, n: np.load(F / sp / f"{n}.npy", mmap_mode="r")
it = np.asarray(L("train", "cand_item_id")[rows]); ei = np.asarray(L("train", "cand_emb_idx")[rows]); ct = np.asarray(L("train", "cand_category_id")[rows])
y = np.asarray(L("train", "cand_target")[rows]); pos = np.asarray(L("train", "cand_pos")[rows])
out["pool_pos_rate"] = float(y.mean()); out["pool_pos_per_feed"] = float(y.sum() / len(sel))
ui, cnt = np.unique(it, return_counts=True)
out["pool_item_distinct"] = int(len(ui)); out["pool_item_rows_per_item_mean"] = float(cnt.mean())
out["pool_item_count_quantiles_50_90_99_max"] = [int(x) for x in np.quantile(cnt, [0.5, 0.9, 0.99, 1.0])]
out["pool_item_share_singletons"] = float((cnt == 1).mean())
out["pool_rows_share_item_count_ge_5"] = float(cnt[cnt >= 5].sum() / cnt.sum()); out["pool_rows_share_item_count_ge_20"] = float(cnt[cnt >= 20].sum() / cnt.sum())
out["pool_emb_idx_distinct"] = int(len(np.unique(ei))); out["pool_emb_missing_share"] = float((ei < 0).mean())
out["pool_category_distinct"] = int(len(np.unique(ct)))
# within-feed row order in train: is pos increasing inside a feed (flat order = production order)?
g0 = np.r_[0, np.cumsum(M[sel])[:-1]]
inc = np.ones(len(rows), bool); d = np.diff(pos.astype(np.int64)); inc[1:] = d > 0; inc[g0] = True
out["pool_rows_pos_increasing_inside_feed_share"] = float(inc.mean())
out["pool_first_row_target_rate_vs_all"] = [float(y[g0].mean()), float(y.mean())]
# same item twice inside a feed?
fid = np.repeat(np.arange(len(sel)), M[sel]); key = fid.astype(np.int64) * (int(it.max()) + 1) + it
out["pool_item_dup_inside_feed_rows"] = int(len(key) - len(np.unique(key)))
for sp in ["val", "test"]:
    f = pl.read_parquet(F / sp / "feeds.parquet"); vi = np.asarray(L(sp, "cand_item_id")); vu = f["user_id"].to_numpy(); vM = f["M"].to_numpy()
    seen = np.isin(vi, ui)
    out[f"{sp}_rows"] = int(len(vi)); out[f"{sp}_rows_item_seen_in_pool"] = float(seen.mean())
    c5 = np.isin(vi, ui[cnt >= 5]); out[f"{sp}_rows_item_seen_ge5_in_pool"] = float(c5.mean())
    out[f"{sp}_feeds_user_seen_in_pool"] = float(np.isin(vu, np.unique(uid[sel][uid[sel] != 0])).mean())
    out[f"{sp}_feeds_user_seen_in_all_train"] = float(np.isin(vu, np.unique(uid[uid != 0])).mean())
    out[f"{sp}_feeds"] = int(len(vu)); out[f"{sp}_user0_share"] = float((vu == 0).mean())
    if sp == "val":
        yv = np.asarray(L("val", "cand_target")); out["val_pos_rate"] = float(yv.mean())
        out["val_pos_rate_item_seen_vs_unseen_in_pool"] = [float(yv[seen].mean()), float(yv[~seen].mean())]
print(f"{time.time() - t0:.0f}s", flush=True)
# all-train item coverage (heavier: 246.7M ids)
ita = L("train", "cand_item_id")
ua, ca = np.unique(np.asarray(ita), return_counts=True)
out["train_all_item_distinct"] = int(len(ua)); out["train_all_item_singletons_share"] = float((ca == 1).mean())
for sp in ["val", "test"]:
    vi = np.asarray(L(sp, "cand_item_id")); out[f"{sp}_rows_item_seen_in_all_train"] = float(np.isin(vi, ua).mean())
    out[f"{sp}_rows_item_seen_ge5_in_all_train"] = float(np.isin(vi, ua[ca >= 5]).mean())
out["hist_rows_per_feed_pool_mean"] = float(H[sel].mean()); out["hist_tokens_total_pool"] = int(H[sel].sum())
out["sec"] = round(time.time() - t0, 1)
json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "logs", "d1_data_look.json"), "w"), indent=1)
for k, v in out.items():
    print(f"{k:55s} {v}")
