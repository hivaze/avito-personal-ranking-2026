"""audit: per-split context (feed descriptors + candidate descriptors) for val and the 30K holdout; cached in cache/ctx_<sp>.npz."""
from au_common import *
sys.path.insert(0, str(ROOT / "nn_x/cb26"))
import polars as pl
from cbdata import hold_index, trH_rows, load_tab
FLAT = ROOT / "data/flat"
CAND = ["it_hist_all_luo", "cat_cnt_all", "seq_max_all", "seq_max_all_rk", "maxsim32", "cos_u", "cos_hall", "tte_it_imp", "tte_it_ctr",
        "tte_ui_imp", "emb_missing", "it_hist_et1_luo", "cat_min_hpos", "f1", "f5", "pre2_fe_z", "seq_cv_invd20", "centrality"]


def build(sp):
    c = CACHE / f"ctx_{sp}.npz"
    if c.exists(): return dict(np.load(c))
    ftr = pl.read_parquet(FLAT / "train/feeds.parquet")
    tr_uid = ftr["user_id"].to_numpy().astype(np.int64)
    if sp == "val":
        f = pl.read_parquet(FLAT / "val/feeds.parquet"); flat = "val"; rows = None; split = "val"
        ucnt = np.bincount(tr_uid, minlength=int(max(tr_uid.max(), f["user_id"].max())) + 1)
        n_other = ucnt[f["user_id"].to_numpy().astype(np.int64)]
    else:
        hold, _ = hold_index(); f = ftr[hold]; flat = "train"; rows = trH_rows(); split = "trH"
        ucnt = np.bincount(tr_uid); n_other = ucnt[f["user_id"].to_numpy().astype(np.int64)] - 1
    M = f["M"].to_numpy().astype(np.int64); H = f["H"].to_numpy().astype(np.int64); uid = f["user_id"].to_numpy().astype(np.int64)
    ts = f["timestamp"].to_numpy().astype(np.int64); day = (ts + 10800) // 86400
    d = dict(M=M, H=H, uid=uid, day=day, hour=((ts + 10800) % 86400) // 3600, n_other=np.where(uid == 0, -1, n_other))
    X = load_tab(split, CAND, rows)
    for i, n in enumerate(CAND): d["c_" + n] = X[:, i]
    # exact item repeat: candidate item_id present in the feed's own history
    ci = np.load(FLAT / flat / "cand_item_id.npy", mmap_mode="r"); cc = np.load(FLAT / flat / "cand_category_id.npy", mmap_mode="r")
    hi = np.load(FLAT / flat / "hist_item_id.npy", mmap_mode="r")
    co = f["cand_off"].to_numpy().astype(np.int64); ho = f["hist_off"].to_numpy().astype(np.int64)
    rep = np.zeros(int(M.sum()), bool); item = np.empty(int(M.sum()), np.int64); cat = np.empty(int(M.sum()), np.int32); o = 0
    for k in range(len(M)):
        it = np.asarray(ci[co[k]:co[k] + M[k]]); item[o:o + M[k]] = it; cat[o:o + M[k]] = cc[co[k]:co[k] + M[k]]
        if H[k] > 0: rep[o:o + M[k]] = np.isin(it, np.asarray(hi[ho[k]:ho[k] + H[k]]))
        o += M[k]
    d["c_in_hist"] = rep; d["c_item"] = item; d["c_cat"] = cat
    np.savez(c, **d)
    return d


if __name__ == "__main__":
    for sp in ("val", "hold"):
        d = build(sp); print(sp, {k: (v.shape, str(v.dtype)) for k, v in d.items()})
        print("  user0 share", (d["uid"] == 0).mean(), "known share", ((d["uid"] != 0) & (d["n_other"] > 0)).mean(), "days", np.unique(d["day"], return_counts=True))
        print("  in_hist share", d["c_in_hist"].mean(), "cold share", (np.nan_to_num(d["c_it_hist_all_luo"]) <= 0).mean())
