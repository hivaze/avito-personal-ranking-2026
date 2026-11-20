"""Why are users with train feeds harder? Lift by user type within H and M strata; residual of user-level train feedback
(tte_uc_ctr, tte_uc_imp, tte_u_ctr) on train-user feeds (symmetric local AUC, as s02b)."""
from common_e import meta, ROOT, SCR
import numpy as np
import numba as nb
import polars as pl

f, M, off, fid = meta("val")
n = len(M)
y = np.load(f"{ROOT}/data/flat/val/cand_target.npy").astype(np.float64)
pf = pl.read_parquet(f"{SCR}/out/perfeed.parquet")
nd, ndr = pf["nd"].to_numpy(), pf["ndr"].to_numpy()
H = pf["H"].to_numpy()
ut = np.where(pf["u0"].to_numpy(), "u0", np.where(pf["cold_user"].to_numpy(), "new", "trainU"))
Hb = np.digitize(H, [51, 129, 256])
Mb = np.digitize(M, [25, 50, 100, 200])
print("lift (NDCG - random) by user type within H bucket:")
for h, lab in enumerate(["<51", "51-128", "129-255", "256"]):
    print(f"  H {lab:8s} " + "  ".join(f"{u} {(nd - ndr)[(Hb == h) & (ut == u)].mean():.4f} (n {((Hb == h) & (ut == u)).sum()})" for u in ("new", "trainU")))
# stratified (H x M) difference trainU - new
diffs, ws = [], []
for h in range(4):
    for m in range(5):
        a = (Hb == h) & (Mb == m) & (ut == "trainU")
        b = (Hb == h) & (Mb == m) & (ut == "new")
        if a.sum() > 30 and b.sum() > 30:
            diffs.append((nd - ndr)[a].mean() - (nd - ndr)[b].mean())
            ws.append(a.sum())
print(f"H x M stratified lift difference trainU - new: {np.average(diffs, weights=ws):+.4f}")
# number of train feeds per user and days since last train feed
trf = pl.read_parquet(f"{ROOT}/data/flat/train/feeds.parquet", columns=["user_id", "timestamp"])
g = trf.group_by("user_id").agg(pl.len().alias("ntr"), pl.col("timestamp").max().alias("last"))
d = pf.select(["fidx", "uid", "ts", "nd", "ndr"]).join(g, left_on="uid", right_on="user_id", how="left")
d = d.filter(pl.col("uid") != 0).with_columns(((pl.col("ts") - pl.col("last")) / 86400).alias("gap_d"))
dd = d.filter(pl.col("ntr").is_not_null())
for lo, hi in ((1, 2), (2, 5), (5, 20), (20, 10 ** 6)):
    s = dd.filter((pl.col("ntr") >= lo) & (pl.col("ntr") < hi))
    print(f"  train feeds [{lo},{hi}): n {s.height}, lift {(s['nd'] - s['ndr']).mean():.4f}")
for lo, hi in ((0, 5), (5, 10), (10, 15), (15, 100)):
    s = dd.filter((pl.col("gap_d") >= lo) & (pl.col("gap_d") < hi))
    print(f"  days since last train feed [{lo},{hi}): n {s.height}, lift {(s['nd'] - s['ndr']).mean():.4f}")
