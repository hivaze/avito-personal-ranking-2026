"""Build a per-feed table for the SEGMENTS error lens (val only).

Output: eda/errors/segments/feed_table.parquet with, per val feed:
  segment variables (observable at test time: M, H, uid0, train feeds of user, cold share, hour, day, feed category
  structure, history structure; diagnostic only (need labels): npos, positive category, positive category in history,
  positive's f3/f1 rank), per-feed NDCG of each model and of oracles, leak flag.
Also writes test_segments.parquet with the observable variables for test (for val->test share comparison).
"""
import os
os.environ.setdefault("NUMBA_NUM_THREADS", "16")
import sys, time
import numpy as np, polars as pl
sys.path.insert(0, "<repo>/src")
from metrics import ndcg_per_feed
from stack_val import feed_rank_z

ROOT = "<repo>"
FLAT = f"{ROOT}/data/flat"
PREDS = f"{ROOT}/preds"
OUT = f"{ROOT}/eda/errors/segments"
t0 = time.time()


def log(*a):
    print(f"[{time.time() - t0:6.1f}s]", *a, flush=True)


# ---------- train item bitmap (train candidates or train histories) and train feeds per user ----------
NITEM = 111_400_000
seen_tr = np.zeros(NITEM, bool)
seen_tr[np.load(f"{FLAT}/train/cand_item_id.npy", mmap_mode="r")[:]] = True
seen_tr_cand = seen_tr.copy()
seen_tr[np.load(f"{FLAT}/train/hist_item_id.npy", mmap_mode="r")[:]] = True
log("train item bitmaps", seen_tr_cand.sum(), seen_tr.sum())

ftr = pl.read_parquet(f"{FLAT}/train/feeds.parquet")
tr_cnt = ftr.filter(pl.col("user_id") != 0).group_by("user_id").agg(pl.len().alias("n_train_feeds"))

fva = pl.read_parquet(f"{FLAT}/val/feeds.parquet")
fte = pl.read_parquet(f"{FLAT}/test/feeds.parquet")

# ---------- leak mask: val feeds whose non-zero user has a LATER feed in val or test ----------
later = pl.concat([fva.select("user_id", "timestamp"), fte.select("user_id", "timestamp")])
j = (fva.select("fidx", "user_id", "timestamp").filter(pl.col("user_id") != 0)
     .join(later.rename({"timestamp": "ts2"}), on="user_id")
     .filter(pl.col("ts2") > pl.col("timestamp")).select("fidx").unique())
leak = np.zeros(len(fva), bool); leak[j["fidx"].to_numpy()] = True
log("leak feeds", leak.sum())


def observable(split, f):
    """Segment variables that are observable at inference time."""
    n = len(f)
    M = f["M"].to_numpy().astype(np.int64); off = f["cand_off"].to_numpy()
    H = f["H"].to_numpy().astype(np.int64)
    fid = np.repeat(np.arange(n), M)
    item = np.load(f"{FLAT}/{split}/cand_item_id.npy")
    cat = np.load(f"{FLAT}/{split}/cand_category_id.npy").astype(np.int64)
    hfid = np.load(f"{FLAT}/{split}/hist_fidx.npy").astype(np.int64)
    hcat = np.load(f"{FLAT}/{split}/hist_category_id.npy").astype(np.int64)
    cold = ~seen_tr[item]
    cold_c = ~seen_tr_cand[item]
    d = f.select("fidx", "user_id", "timestamp", "M", "H").join(tr_cnt, on="user_id", how="left").sort("fidx")
    ntr = d["n_train_feeds"].fill_null(0).to_numpy()
    ts = f["timestamp"].to_numpy().astype(np.int64)
    out = {
        "fidx": np.arange(n), "M": M, "H": H, "uid0": (f["user_id"].to_numpy() == 0),
        "n_train_feeds": ntr,
        "cold_share": np.bincount(fid, cold, n) / M,
        "coldcand_share": np.bincount(fid, cold_c, n) / M,
        "hour": ((ts + 3 * 3600) % 86400) // 3600,
        "day": ((ts + 3 * 3600) // 86400) - 20718,
    }
    # feed category structure
    ck = fid * 64 + cat
    uk, inv, ucnt = np.unique(ck, return_inverse=True, return_counts=True)
    out["n_cat"] = np.bincount(uk // 64, minlength=n)
    out["max_cat_share"] = np.maximum.reduceat(ucnt, np.r_[0, np.flatnonzero(np.diff(uk // 64)) + 1]) / M
    # history category structure
    hk = hfid * 64 + hcat
    hu, hcnt = np.unique(hk, return_counts=True)
    hf = hu // 64
    out["hist_n_cat"] = np.bincount(hf, minlength=n)
    top = np.zeros(n); np.maximum.at(top, hf, hcnt)
    out["hist_top_cat_share"] = np.where(H > 0, top / np.maximum(H, 1), 0.0)
    # share of candidates whose category appears in the user's history
    cand_in_hist = np.isin(ck, hu)
    out["cand_cat_in_hist_share"] = np.bincount(fid, cand_in_hist, n) / M
    return out, dict(M=M, off=off, fid=fid, cat=cat, ck=ck, hu=hu, hk=hk, hcnt=hcnt, H=H, item=item, cold=cold)


obs_te, _ = observable("test", fte)
pl.DataFrame(obs_te).write_parquet(f"{OUT}/test_segments.parquet")
log("test observable done")

obs, ctx = observable("val", fva)
M, off, fid, cat, ck = ctx["M"], ctx["off"], ctx["fid"], ctx["cat"], ctx["ck"]
n = len(M)
y = np.load(f"{FLAT}/val/cand_target.npy").astype(np.float64)
f1 = np.load(f"{FLAT}/val/cand_f1.npy").astype(np.float64)
f3 = np.load(f"{FLAT}/val/cand_f3.npy").astype(np.float64)
npos = np.bincount(fid, y, n)
obs["npos"] = npos.astype(np.int64)
obs["leak"] = leak

# ---------- diagnostic (label-based) variables ----------
posk = np.unique(ck[y > 0])
is_poscat = np.isin(ck, posk)
obs["n_pos_cat"] = np.bincount(posk // 64, minlength=n)
# single positive category id (or -1 if 0 or several)
pc = np.full(n, -1); single = obs["n_pos_cat"] == 1
pc[posk // 64] = np.where(single[posk // 64], posk % 64, -1)
obs["pos_cat"] = pc
obs["pos_cat_share_feed"] = np.bincount(fid, is_poscat, n) / M  # share of candidates in a positive category
hu, hk_cnt = ctx["hu"], ctx["hcnt"]
pos_in_hist = np.isin(posk, hu)
obs["pos_cat_in_hist"] = np.bincount(posk // 64, pos_in_hist, n) > 0
# share of history events in the positive categories
idx = np.searchsorted(hu, posk); idx = np.minimum(idx, len(hu) - 1)
cntk = np.where(hu[idx] == posk, hk_cnt[idx], 0)
obs["pos_cat_hist_share"] = np.where(ctx["H"] > 0, np.bincount(posk // 64, cntk, n) / np.maximum(ctx["H"], 1), 0.0)
obs["pos_cold_any"] = np.bincount(fid, (y > 0) & ctx["cold"], n) > 0
obs["pos_cold_all"] = (np.bincount(fid, (y > 0) & ctx["cold"], n) == npos) & (npos > 0)


def best_pos_pct(x):
    """Share of candidates scoring strictly higher than the feed's best positive (0 = best positive at the top)."""
    bp = np.full(n, -np.inf); np.maximum.at(bp, fid[y > 0], x[y > 0])
    above = np.bincount(fid, x > bp[fid], n)
    return np.where(npos > 0, above / np.maximum(M - 1, 1), np.nan)


obs["pos_f3_pct"] = best_pos_pct(f3)
obs["pos_f1_pct"] = best_pos_pct(f1)
# rank of the positive's category among the feed's categories by f3 (f3 is constant within (feed, category))
cat_f3 = {}
kk, first = np.unique(ck, return_index=True)
kf3 = f3[first]; kfeed = kk // 64
# per feed: number of categories with f3 strictly greater than the best positive category's f3
kpos = np.isin(kk, posk)
bp3 = np.full(n, -np.inf); np.maximum.at(bp3, kfeed[kpos], kf3[kpos])
obs["pos_f3_catrank"] = np.where(npos > 0, np.bincount(kfeed, kf3 > bp3[kfeed], n), -1)

# ---------- model predictions ----------
models = {
    "nn_oof": "nn_final2_val_oof", "nn_tr": "nn_final2_trainonly_val",
    # report tables used stk2_res2_r3 (leaky ALS); it was deleted mid-analysis, fall back to the leak-free stk2_res4_clean
    "stk": "stk2_res2_r3_val" if os.path.exists(f"{PREDS}/stk2_res2_r3_val.npy") else "stk2_res4_clean_val",
    "cb": "cb_tr300k_none_val", "lgb": "lgb_tr300k_none_val",
}
Z = {}
for k, fn in models.items():
    s = np.load(f"{PREDS}/{fn}.npy").astype(np.float32)
    Z[k] = feed_rank_z(off, M, s)[:, 1].astype(np.float64)
    obs[f"ndcg_{k}"] = ndcg_per_feed(s, y, off, M)
    log(k, obs[f"ndcg_{k}"].mean(), obs[f"ndcg_{k}"][~leak].mean())
Z["blend"] = 0.5 * Z["nn_oof"] + 0.5 * Z["stk"]
obs["ndcg_blend"] = ndcg_per_feed(Z["blend"], y, off, M)
for k in ["f1", "f3"]:
    x = f1 if k == "f1" else f3
    obs[f"ndcg_{k}"] = ndcg_per_feed(x, y, off, M)

# oracles on top of the NN: positive-category oracle (pos categories first, NN order inside)
for k in ["nn_oof", "nn_tr", "cb"]:
    obs[f"ndcg_catorc_{k}"] = ndcg_per_feed(Z[k] + 1e3 * is_poscat, y, off, M)
    obs[f"pos_{k}_pct"] = best_pos_pct(Z[k])
# category-order oracle the other way: perfect category ranking is above; "NN category order + perfect within category"
# = rank categories by NN max score in category, put positives first inside each category.
def within_orc(z):
    cmax = np.full(len(kk), -np.inf)
    inv = np.searchsorted(kk, ck)
    np.maximum.at(cmax, inv, z)
    o = np.lexsort((-cmax, kfeed))  # categories sorted by feed, then NN max desc
    crank = np.empty(len(kk)); crank[o] = np.arange(len(kk)) - np.searchsorted(kfeed[o], kfeed[o])
    zr = (z - z.min()) / (z.max() - z.min() + 1e-9)
    # category order by NN max score; inside a category positives first, then NN order
    return ndcg_per_feed(-crank[inv] * 100 + y * 10 + zr, y, off, M)
obs["ndcg_withinorc_nn_oof"] = within_orc(Z["nn_oof"])

# NN-vs-CB within-feed agreement (Spearman on z-ranks)
rk = {k: feed_rank_z(off, M, Z[k].astype(np.float32))[:, 0].astype(np.float64) for k in ["nn_oof", "cb"]}
a, b = rk["nn_oof"] - 0.5, rk["cb"] - 0.5
num = np.bincount(fid, a * b, n); den = np.sqrt(np.bincount(fid, a * a, n) * np.bincount(fid, b * b, n)) + 1e-12
obs["rho_nn_cb"] = num / den

df = pl.DataFrame(obs)
df.write_parquet(f"{OUT}/feed_table.parquet")
np.save(f"{OUT}/is_poscat.npy", is_poscat)
log("saved", df.shape)
