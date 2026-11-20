"""Val: snapshot advantage over the pool-only reference (L_ref - L_snap, CE/logM) by positive coldness and category,
to compare with the same segments on production train feeds (04). CPU, read-only. Label-defined slices: descriptive."""
import numpy as np, polars as pl
from pathlib import Path
ROOT = Path("<repo>"); D = ROOT / "nn_x/insights/data_usage/out"
v = pl.read_parquet(D / "val_feed_losses.parquet")
M = v["M"].to_numpy(); n = len(M); fid = np.repeat(np.arange(n), M)
y = np.load(ROOT / "data/flat/val/cand_target.npy") > 0
cat = np.load(ROOT / "data/flat/val/cand_category_id.npy").astype(np.int64)
luo = np.load(ROOT / "data/feats/val/it_hist_all_luo.npy")
npos = np.bincount(fid, weights=y, minlength=n)
coldpos = np.bincount(fid[y], weights=(luo[y] == 0), minlength=n) / np.maximum(npos, 1)
pc = np.full(n, -1); pc[fid[y][::-1]] = cat[y][::-1]  # category of the first positive in storage order
v = v.with_columns(pl.Series("coldpos", coldpos), pl.Series("pcat", pc),
                   ((pl.col("L_snap101") + pl.col("L_snap102") + pl.col("L_snap103") + pl.col("L_snap104")) / 4).alias("L_snap"),
                   ((pl.col("N_snap101") + pl.col("N_snap102") + pl.col("N_snap103") + pl.col("N_snap104")) / 4).alias("N_snap"))
v = v.with_columns(pl.when(pl.col("coldpos") == 0).then(pl.lit("warm")).when(pl.col("coldpos") >= 1).then(pl.lit("allcold")).otherwise(pl.lit("mixed")).alias("coldb"),
                   (pl.col("L_ref") - pl.col("L_snap")).alias("adv"), (pl.col("N_snap") - pl.col("N_ref")).alias("dN"))
pl.Config.set_tbl_rows(50); pl.Config.set_tbl_width_chars(200)
a = v.group_by("coldb").agg(n=pl.len(), adv=pl.col("adv").mean(), se=pl.col("adv").std() / pl.len().sqrt(), dN=pl.col("dN").mean(),
                            dN_se=pl.col("dN").std() / pl.len().sqrt()).sort("coldb")
print("val: snapshot - reference by positive coldness (adv = L_ref - L_snap; dN = NDCG snap - ref)"); print(a)
top = [13, 2, 3, 50, 6, 7, 10, 48, 15, 11, 16, 29, 21, 47, 23]
b = v.filter(pl.col("pcat").is_in(top)).group_by("pcat").agg(n=pl.len(), adv=pl.col("adv").mean(), se=pl.col("adv").std() / pl.len().sqrt(),
                                                          dN=pl.col("dN").mean()).sort("n", descending=True)
p = pl.read_csv(D / "04_seg_pcat_top15.csv").select(["pcat", pl.col("adv").alias("adv_prod")])
b = b.join(p, on="pcat", how="left")
print("val vs production train: advantage by category of the positive"); print(b)
print("corr(adv_val, adv_prod) over the 15 categories:", np.corrcoef(b["adv"].to_numpy(), b["adv_prod"].to_numpy())[0, 1])
a.write_csv(D / "10_val_adv_cold.csv"); b.write_csv(D / "10_val_adv_pcat.csv")
