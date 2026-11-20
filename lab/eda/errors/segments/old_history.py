"""Does the same user's EARLIER (train) snapshot history carry information missing from the val history window?
(earlier snapshots are allowed by the leak rules; later ones are not). Val, leak-masked, users seen in train."""
import numpy as np, polars as pl
ROOT = "<repo>"; FLAT = f"{ROOT}/data/flat"; OUT = f"{ROOT}/eda/errors/segments"
d = pl.read_parquet(f"{OUT}/feed_table.parquet")
fv = pl.read_parquet(f"{FLAT}/val/feeds.parquet"); ft = pl.read_parquet(f"{FLAT}/train/feeds.parquet")
# all train feeds of val users (user != 0)
vu = fv.filter(pl.col("user_id") != 0).select("fidx", "user_id")
tr = ft.join(vu.select("user_id").unique(), on="user_id").select("fidx", "user_id", "H", "hist_off")
print("train feeds of val users", tr.height)
th_item = np.load(f"{FLAT}/train/hist_item_id.npy", mmap_mode="r"); th_cat = np.load(f"{FLAT}/train/hist_category_id.npy", mmap_mode="r")
vh_item = np.load(f"{FLAT}/val/hist_item_id.npy"); vh_cat = np.load(f"{FLAT}/val/hist_category_id.npy").astype(np.int64)
vH = fv["H"].to_numpy(); voff = fv["hist_off"].to_numpy()
vc_cat = np.load(f"{FLAT}/val/cand_category_id.npy").astype(np.int64); y = np.load(f"{FLAT}/val/cand_target.npy")
M = fv["M"].to_numpy(); coff = fv["cand_off"].to_numpy()
by_user = {}
for u, o, h in tr.select("user_id", "hist_off", "H").iter_rows():
    by_user.setdefault(u, []).append((o, h))
keep = ~d["leak"].to_numpy(); Hseg = d["H"].to_numpy()
rows = []
for fi, u in vu.iter_rows():
    if not keep[fi] or u not in by_user:
        continue
    vi = set(vh_item[voff[fi]:voff[fi] + vH[fi]].tolist()); vcats = set(vh_cat[voff[fi]:voff[fi] + vH[fi]].tolist())
    ti, tc = set(), set()
    for o, h in by_user[u]:
        ti.update(th_item[o:o + h].tolist()); tc.update(np.asarray(th_cat[o:o + h]).astype(np.int64).tolist())
    new_items = len(ti - vi); new_cats = tc - vcats
    pc = set(vc_cat[coff[fi]:coff[fi] + M[fi]][y[coff[fi]:coff[fi] + M[fi]] > 0].tolist())
    rows.append(dict(fidx=fi, H=vH[fi], new_items=new_items, n_tr_items=len(ti), new_cats=len(new_cats),
                     poscat_in_val_hist=len(pc & vcats) > 0, poscat_in_old_only=(len(pc & vcats) == 0) and len(pc & new_cats) > 0))
r = pl.DataFrame(rows)
r = r.join(d.select("fidx", "ndcg_nn_oof", "pos_cat_in_hist"), on="fidx")
print(f"val feeds of train-seen users (clean): {r.height} ({r.height / keep.sum():.1%} of clean val feeds)")
for name, m in [("all", pl.lit(True)), ("H256", pl.col("H") == 256), ("H<256", pl.col("H") < 256)]:
    s = r.filter(m)
    print(f"  {name:6s} n={s.height:5d} mean old-only items {s['new_items'].mean():6.1f} (of {s['n_tr_items'].mean():.1f} train-history items); "
          f"feeds with >=1 old-only category {(s['new_cats'] > 0).mean():.3f}; positive cat NOT in val history {(~s['poscat_in_val_hist']).mean():.3f}, "
          f"of which found in old history {s['poscat_in_old_only'].sum()}/{(~s['poscat_in_val_hist']).sum()}")
