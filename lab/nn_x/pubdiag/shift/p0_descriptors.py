"""Label-free per-feed descriptors for val and test (adversarial validation / reweighting / slices).
Writes desc_val.parquet, desc_test.parquet (row i = feed fidx i). Also the per-candidate cold flag (luo def) for val."""
from shiftlib import *  # noqa: F401,F403

FC = ROOT / "nn_x/r6/feat_cache"
SEG = ROOT / "eda/errors/segments"
OBS = ["M", "H", "uid0", "n_train_feeds", "cold_share", "coldcand_share", "hour", "n_cat", "max_cat_share",
       "hist_n_cat", "hist_top_cat_share", "cand_cat_in_hist_share"]

# top categories by test candidate share (label-free)
cat_te = np.load(FLAT / "test/cand_category_id.npy").astype(np.int64)
TOPC = np.argsort(-np.bincount(cat_te, minlength=64))[:20]
hcat_te = np.load(FLAT / "test/hist_category_id.npy").astype(np.int64)
TOPH = np.argsort(-np.bincount(hcat_te, minlength=64))[:10]
TOPS = sorted(np.argsort(-np.bincount(np.load(FLAT / "test/hist_event_source.npy").astype(np.int64), minlength=32))[:6])
del cat_te, hcat_te


def build(split):
    f, M, off = feed_meta(split)
    n = len(M); fid = np.repeat(np.arange(n), M)
    seg = pl.read_parquet(SEG / ("feed_table.parquet" if split == "val" else "test_segments.parquet"))
    assert (seg["fidx"].to_numpy() == np.arange(n)).all() and (seg["M"].to_numpy() == M).all()
    D = {k: seg[k].to_numpy().astype(np.float64) for k in OBS}
    ts = f["timestamp"].to_numpy().astype(np.int64)
    D["minute_of_day"] = (((ts + 10800) % 86400) // 60).astype(np.float64)
    D["logM"] = np.log(M)
    cat = np.load(FLAT / f"{split}/cand_category_id.npy").astype(np.int64)
    for c in TOPC:
        D[f"cc{c}"] = np.bincount(fid, cat == c, n) / M
    top = np.zeros(n, np.int64)  # feed's top category (mode)
    k = fid * 64 + cat; uk, cnt = np.unique(k, return_counts=True)
    order = np.lexsort((-cnt, uk // 64)); first = np.r_[True, np.diff((uk // 64)[order]) != 0]
    top[(uk // 64)[order][first]] = (uk % 64)[order][first]
    D["top_cat"] = top.astype(np.float64)
    for i in range(1, 6):
        x = np.load(FLAT / f"{split}/cand_f{i}.npy").astype(np.float64)
        D[f"f{i}_mean"] = np.bincount(fid, x, n) / M
        D[f"f{i}_sd"] = np.sqrt(np.maximum(np.bincount(fid, x * x, n) / M - D[f"f{i}_mean"] ** 2, 0))
    f1 = np.load(FLAT / f"{split}/cand_f1.npy")
    D["f1_atmin"] = np.bincount(fid, f1 <= f1.min() + 1e-6, n) / M
    luo = np.load(FC / f"{split}/it_hist_all_luo.npy").astype(np.float64)
    l1 = np.log1p(luo)
    D["luo_cold_share"] = np.bincount(fid, luo == 0, n) / M
    D["luo_logmean"] = np.bincount(fid, l1, n) / M
    D["luo_logmax"] = np.maximum.reduceat(l1, off)
    cc = np.log1p(np.load(FC / f"{split}/it_cand_cnt_train.npy").astype(np.float64))
    D["ccnt_logmean"] = np.bincount(fid, cc, n) / M
    D["ccnt_zero_share"] = np.bincount(fid, cc == 0, n) / M
    e1 = np.load(FC / f"{split}/it_hist_et1_luo.npy").astype(np.float64)
    D["et1_logmean"] = np.bincount(fid, np.log1p(e1), n) / M
    # history composition
    hfid = np.load(FLAT / f"{split}/hist_fidx.npy").astype(np.int64)
    H = f["H"].to_numpy().astype(np.float64); Hs = np.maximum(H, 1)
    het = np.load(FLAT / f"{split}/hist_event_type.npy").astype(np.int64)
    for e in range(1, 11):
        D[f"het{e}"] = np.bincount(hfid, het == e, n) / Hs
    hes = np.load(FLAT / f"{split}/hist_event_source.npy").astype(np.int64)
    for e in TOPS:
        D[f"hes{e}"] = np.bincount(hfid, hes == e, n) / Hs
    hcat = np.load(FLAT / f"{split}/hist_category_id.npy").astype(np.int64)
    for c in TOPH:
        D[f"hc{c}"] = np.bincount(hfid, hcat == c, n) / Hs
    # history item freshness: share of history items never seen as train candidates/history? use luo not available for
    # history -> skip. user_id kept for grouping only.
    D["user_id"] = f["user_id"].to_numpy().astype(np.float64)
    D["timestamp"] = ts.astype(np.float64)
    out = pl.DataFrame(D)
    out.write_parquet(OUT / f"desc_{split}.parquet")
    print(split, out.shape)
    if split == "val":
        np.save(OUT / "val_cand_cold_luo.npy", (luo == 0))
    return out


if __name__ == "__main__":
    build("val"); build("test")
