"""Pre-training audit A5 (CPU): length/density of per-item daily time series available from train candidate lists
(exposure) and train labels (clicks) for val/test candidate items: what a time-series model of item popularity would see."""
import numpy as np, polars as pl
from pathlib import Path
R = Path("<repo>"); F = R / "data/flat"; BOUND = 1790497920
def L(s, n): return np.load(F / s / f"{n}.npy", mmap_mode="r")
fe = pl.read_parquet(F / "train/feeds.parquet"); day = ((fe["timestamp"].to_numpy().astype(np.int64) + 3 * 3600) // 86400).astype(np.int32)
d0 = day.min(); M = fe["M"].to_numpy()
it = np.asarray(L("train", "cand_item_id")); tg = np.asarray(L("train", "cand_target")).astype(np.int64)
rd = np.repeat(day - d0, M).astype(np.int64)
G = int(max(it.max(), np.asarray(L("val", "cand_item_id")).max(), np.asarray(L("test", "cand_item_id")).max())) + 1
want = np.zeros(G, np.bool_); want[np.asarray(L("val", "cand_item_id"))] = True; want[np.asarray(L("test", "cand_item_id"))] = True
m = want[it]; it, tg, rd = it[m], tg[m], rd[m]
key = it.astype(np.int64) * 16 + rd
uk, inv = np.unique(key, return_inverse=True)
exp_cnt = np.bincount(inv); clk = np.bincount(inv, weights=tg)
items = uk // 16
ndays = np.bincount(np.searchsorted(np.unique(items), items))
ui = np.unique(items)
nd = np.zeros(G, np.int16); nd[ui] = ndays
tot_exp = np.zeros(G, np.int32); np.add.at(tot_exp, items, exp_cnt)
tot_clk = np.zeros(G, np.int32); np.add.at(tot_clk, items, clk.astype(np.int32))
print(f"train days with data: {sorted(set((day-d0).tolist()))}")
for s in ["val", "test"]:
    ci = np.asarray(L(s, "cand_item_id")); fs = pl.read_parquet(F / s / "feeds.parquet"); ts = np.repeat(fs["timestamp"].to_numpy(), fs["M"].to_numpy())
    for lab, mk in ([("val", np.ones(len(ci), bool))] if s == "val" else [("test public", ts < BOUND), ("test private", ts >= BOUND)]):
        x = ci[mk]; n = nd[x]
        print(f"{lab}: candidate rows with >=1 train exposure day {np.mean(n>0):.4f}; days exposed q50/q90 among exposed {np.median(n[n>0]):.0f}/{np.quantile(n[n>0],0.9):.0f} (of 12); >=7 days {np.mean(n>=7):.4f}; "
              f"total train exposures q50 among exposed {np.median(tot_exp[x][n>0]):.0f}; rows whose item has >=1 train click {np.mean(tot_clk[x]>0):.4f}; >=5 clicks {np.mean(tot_clk[x]>=5):.4f}")
