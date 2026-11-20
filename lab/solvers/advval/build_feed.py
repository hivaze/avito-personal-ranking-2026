"""Feed-level covariates for adversarial validation: train random-like feeds, val, test.
Causal definitions so that train rows and val/test rows mean the same thing:
  * cold item: never seen in train (as candidate or history event) on a day strictly before the feed's day
    (for val/test feeds: never seen in train at all)
  * n_prior_feeds: same-user train feeds strictly before the feed (val/test: all train feeds of the user); user 0 -> -1
Output: OUT/feed_{train_rl,val,test}.parquet
"""
import time
from common_av import *

t0 = time.time()
def log(*a): print(f"[{time.time() - t0:7.1f}s]", *a, flush=True)

day_of = lambda ts: (np.asarray(ts, np.int64) + MSK) // 86400

# ---------------------------------------------------------------- item first-seen day in train
ftr, Mtr, offtr = meta("train")
ts_tr = ftr["timestamp"].to_numpy().astype(np.int64)
d_tr = day_of(ts_tr)
hoff_tr = ftr["hist_off"].to_numpy().astype(np.int64)
Htr = ftr["H"].to_numpy().astype(np.int64)
c_item_tr = np.load(f"{FLAT}/train/cand_item_id.npy", mmap_mode="r")
h_item_tr = np.load(f"{FLAT}/train/hist_item_id.npy", mmap_mode="r")
mx = 0
for s in ["train", "val", "test"]:
    for k in ["cand_item_id", "hist_item_id"]:
        mx = max(mx, int(np.load(f"{FLAT}/{s}/{k}.npy", mmap_mode="r").max()))
NITEM = mx + 1
INF = 10 ** 9
first_seen = np.full(NITEM, INF, np.int64)
for d in np.unique(d_tr):
    fi = np.where(d_tr == d)[0]
    a, b = fi[0], fi[-1]
    cs = np.asarray(c_item_tr[offtr[a]:offtr[b] + Mtr[b]])
    hs = np.asarray(h_item_tr[hoff_tr[a]:hoff_tr[b] + Htr[b]])
    u = np.unique(np.concatenate([cs, hs]))
    u = u[first_seen[u] == INF]
    first_seen[u] = d
    log("day", d, "new items", len(u))
np.save(f"{SCR}/item_first_seen_day.npy", first_seen)

# prior train feeds per user
order = np.argsort(ts_tr, kind="stable")
uid_tr = ftr["user_id"].to_numpy()
dfu = pl.DataFrame({"i": np.arange(len(uid_tr)), "u": uid_tr, "ts": ts_tr}).sort("ts")
dfu = dfu.with_columns(pl.col("i").cum_count().over("u").alias("k"),
                       pl.col("ts").shift(1).over("u").alias("prev_ts")).sort("i")
prior_tr = dfu["k"].to_numpy().astype(np.int64) - 1  # cum_count starts at 1
prev_ts_tr = dfu["prev_ts"].to_numpy()
tr_users = (ftr.filter(pl.col("user_id") != 0).group_by("user_id")
            .agg(pl.len().alias("ntr"), pl.col("timestamp").max().alias("last_ts")))

# top categories / sources (from val, fixed lists)
cat_val = np.load(f"{FLAT}/val/cand_category_id.npy").astype(np.int64)
TOPCAT = np.argsort(np.bincount(cat_val, minlength=64))[::-1][:15]
es_val = np.load(f"{FLAT}/val/hist_event_source.npy").astype(np.int64)
TOPES = np.argsort(np.bincount(es_val, minlength=32))[::-1][:6]

MI_COUNTS = ["it_hist_all_luo", "seq_cv_invd20", "seq_cv_pop"]
MI_MEAN = ["ctx_nd_max", "ctx_nd_knn_u", "ctx_nd_knn_h", "ctx_nd_top5h", "ctx_nd_hcent", "ctx_nd_ucent", "ctx_hn_top1w",
           "ctx_hn_rrw", "ctx_hs_k128", "ctx_hs_mutual", "maxsim32", "cos_u", "cos_h10", "pre2_fe_z", "pre2_new_z",
           "prew_fe_z", "pre2_fe_s"]
MI_HAS = ["pre2_fe", "prew_fe"]
VT_MEAN = ["it_tr_ctr", "it_tr_coec", "it_cand_cnt_train", "centrality", "pre_fe_z", "pre_new_z", "pre_idmax32",
           "pre_hmax32"]  # fine-tune extras (val/test only)


def ent_from_keys(fid_rows, keys, n, base):
    """per-feed entropy / number of distinct values / top share of a categorical per-row variable"""
    k = fid_rows * base + keys
    uk, cnt = np.unique(k, return_counts=True)
    f = uk // base
    tot = np.bincount(f, cnt, n)
    p = cnt / tot[f]
    ent = np.bincount(f, -p * np.log(p), n)
    ncat = np.bincount(f, minlength=n)
    top = np.zeros(n); np.maximum.at(top, f, p)
    return ent, ncat, top, uk


def build(split, fsel):
    f, M, off = meta(split)
    n = len(fsel)
    ff = f[fsel]
    Ms = M[fsel]
    Hs = ff["H"].to_numpy().astype(np.int64)
    ts = ff["timestamp"].to_numpy().astype(np.int64)
    uid = ff["user_id"].to_numpy()
    dd = day_of(ts)
    crow = rows_of(off, M, fsel)
    fidr = np.repeat(np.arange(n), Ms)
    hoff = f["hist_off"].to_numpy().astype(np.int64)
    hrow = rows_of(hoff, f["H"].to_numpy().astype(np.int64), fsel)
    hfid = np.repeat(np.arange(n), Hs)
    out = {"fidx": fsel, "user_id": uid, "ts": (ts - BOUND) / 86400.0,
           "hour": ((ts + MSK) % 86400) // 3600, "weekday": ((dd + 3) % 7),  # 1970-01-01 was a Thursday
           "M": Ms, "logM": np.log(Ms), "H": Hs, "user0": (uid == 0).astype(np.int8)}
    # user history in train
    if split == "train":
        pr = prior_tr[fsel].astype(float); pv = prev_ts_tr[fsel]
        dsl = np.where(pr > 0, (ts - np.nan_to_num(pv.astype(float), nan=0)) / 86400.0, np.nan)
    else:
        j = ff.select("user_id").with_row_index("r").join(tr_users, on="user_id", how="left").sort("r")
        pr = j["ntr"].fill_null(0).to_numpy().astype(float)
        lt = j["last_ts"].to_numpy()
        dsl = np.where(pr > 0, (ts - np.nan_to_num(lt.astype(float), nan=0)) / 86400.0, np.nan)
    pr[uid == 0] = -1; dsl[uid == 0] = np.nan
    out["n_prior_feeds"] = pr; out["days_since_prior"] = dsl
    out["cold_user"] = ((pr == 0) & (uid != 0)).astype(np.int8)
    # items
    citem = np.load(f"{FLAT}/{split}/cand_item_id.npy", mmap_mode="r")[crow]
    hitem = np.load(f"{FLAT}/{split}/hist_item_id.npy", mmap_mode="r")[hrow]
    fs_c = first_seen[citem]; fs_h = first_seen[hitem]
    cold = fs_c >= dd[fidr] if split == "train" else fs_c == INF
    hcold = fs_h >= dd[hfid] if split == "train" else fs_h == INF
    out["cold_item_share"] = np.bincount(fidr, cold, n) / Ms
    hpos = np.load(f"{FLAT}/{split}/hist_hpos.npy", mmap_mode="r")[hrow]
    r10 = hpos < 10
    n10 = np.bincount(hfid[r10], minlength=n)
    out["hist10_cold_share"] = np.where(n10 > 0, np.bincount(hfid[r10], hcold[r10], n) / np.maximum(n10, 1), np.nan)
    out["hist_cold_share"] = np.where(Hs > 0, np.bincount(hfid, hcold, n) / np.maximum(Hs, 1), np.nan)
    # f1..f5
    for i in range(1, 6):
        x = np.load(f"{FLAT}/{split}/cand_f{i}.npy", mmap_mode="r")[crow].astype(np.float64)
        mu = np.bincount(fidr, x, n) / Ms
        out[f"f{i}_mean"] = mu
        out[f"f{i}_sd"] = np.sqrt(np.maximum(np.bincount(fidr, x * x, n) / Ms - mu ** 2, 0))
    # candidate categories
    ccat = np.load(f"{FLAT}/{split}/cand_category_id.npy", mmap_mode="r")[crow].astype(np.int64)
    ent, ncat, top, uck = ent_from_keys(fidr, ccat, n, 64)
    out["cat_entropy"] = ent; out["cat_n"] = ncat; out["cat_top_share"] = top
    for c in TOPCAT:
        out[f"cat{c}_share"] = np.bincount(fidr, ccat == c, n) / Ms
    cemb = np.load(f"{FLAT}/{split}/cand_emb_idx.npy", mmap_mode="r")[crow]
    out["cand_emb_miss_share"] = np.bincount(fidr, cemb < 0, n) / Ms
    # history structure
    hcat = np.load(f"{FLAT}/{split}/hist_category_id.npy", mmap_mode="r")[hrow].astype(np.int64)
    het = np.load(f"{FLAT}/{split}/hist_event_type.npy", mmap_mode="r")[hrow].astype(np.int64)
    hes = np.load(f"{FLAT}/{split}/hist_event_source.npy", mmap_mode="r")[hrow].astype(np.int64)
    hemb = np.load(f"{FLAT}/{split}/hist_emb_idx.npy", mmap_mode="r")[hrow]
    hent, hncat, htop, uhk = ent_from_keys(hfid, hcat, n, 64)
    Hd = np.maximum(Hs, 1)
    out["hist_cat_entropy"] = np.where(Hs > 0, hent, np.nan); out["hist_cat_n"] = hncat
    out["hist_cat_top_share"] = np.where(Hs > 0, htop, np.nan)
    for e in range(1, 5):
        out[f"hist_et{e}_share"] = np.where(Hs > 0, np.bincount(hfid, het == e, n) / Hd, np.nan)
    for e in TOPES:
        out[f"hist_es{e}_share"] = np.where(Hs > 0, np.bincount(hfid, hes == e, n) / Hd, np.nan)
    out["hist_emb_miss_share"] = np.where(Hs > 0, np.bincount(hfid, hemb < 0, n) / Hd, np.nan)
    ck = fidr * 64 + ccat
    out["cand_cat_in_hist_share"] = np.bincount(fidr, np.isin(ck, uhk), n) / Ms
    last_cat = np.full(n, -1); last_cat[hfid[hpos == 0]] = hcat[hpos == 0]
    out["cand_cat_eq_last_share"] = np.bincount(fidr, ccat == last_cat[fidr], n) / Ms
    # user embedding
    ue = np.load(f"{FLAT}/{split}/user_emb.npy", mmap_mode="r")[fsel].astype(np.float32)
    out["uemb_norm"] = np.linalg.norm(ue, axis=1); out["uemb_mean"] = ue.mean(1); out["uemb_sd"] = ue.std(1)
    out["uemb_zero"] = (np.abs(ue).sum(1) == 0).astype(np.int8)
    # model inputs (per-feed means / zero shares)
    for nm in MI_COUNTS:
        x = load_feat(split, nm, crow).astype(np.float64)
        out[f"{nm}_zero_share"] = np.bincount(fidr, x <= 0, n) / Ms
        out[f"{nm}_mlog"] = np.bincount(fidr, np.log1p(np.maximum(np.nan_to_num(x), 0)), n) / Ms
    for nm in MI_MEAN + (VT_MEAN if split != "train" else []):
        if feat_file(split, nm) is None:
            log("missing", split, nm); continue
        x = load_feat(split, nm, crow).astype(np.float64)
        ok = ~np.isnan(x)
        cnt = np.bincount(fidr, ok, n)
        out[f"{nm}_mean"] = np.where(cnt > 0, np.bincount(fidr, np.where(ok, x, 0), n) / np.maximum(cnt, 1), np.nan)
    for nm in MI_HAS:
        x = load_feat(split, nm, crow)
        out[f"{nm}_has_share"] = np.bincount(fidr, ~np.isnan(x), n) / Ms
    df = pl.DataFrame(out)
    return df, ue


parts = {}
ues = {}
sel_tr = train_random_like()
log("train random-like feeds", len(sel_tr))
for split, fsel in [("val", np.arange(len(meta("val")[0]))), ("test", np.arange(len(meta("test")[0]))),
                    ("train_rl", sel_tr)]:
    sp = "train" if split == "train_rl" else split
    df, ue = build(sp, fsel)
    parts[split] = df; ues[split] = ue
    log(split, df.shape)
# user-embedding PCA fitted on val+test (fixed basis applied to all)
U = np.concatenate([ues["val"], ues["test"]]).astype(np.float64)
mu = U.mean(0)
_, _, Vt = np.linalg.svd(U - mu, full_matrices=False)
for split, df in parts.items():
    P = (ues[split] - mu) @ Vt[:8].T
    df = df.with_columns([pl.Series(f"uemb_pc{i}", P[:, i]) for i in range(8)])
    df.write_parquet(f"{OUT}/feed_{split}.parquet")
    log("wrote", split, df.shape)
