"""Summarise the GBDT proxy arms: seed-averaged per-feed NDCG, paired deltas vs base.
Two readouts: (a) fixed 200 rounds (near the base peak) on full val and the holdout; (b) round count chosen per arm on
the TUNE half of val (user hash, as in 09), delta read on the CONFIRM half (no selection on the readout feeds).
Writes out/12_gbdt_summary.csv."""
import glob
import json
import re
import numpy as np
import polars as pl
from pathlib import Path

D = Path("<repo>/nn_x/insights/data_usage/out")
G = D / "gb"
mv = np.load(G / "val_meta.npz")
u = mv["user_id"].astype(np.int64); idx = np.arange(len(u))
tune = np.where(u == 0, idx % 2 == 0, u % 2 == 0); conf = ~tune
Mv = mv["M"]
short = Mv < 50
res = {}
for f in sorted(glob.glob(str(G / "res_*_s*.npz"))):
    m = re.match(r".*/res_(.+)_s(\d+)\.npz$", f)
    if not m:
        continue
    arm, s = m.group(1), int(m.group(2))
    z = np.load(f)
    res.setdefault(arm, {})[s] = {k: z[k] for k in z.files if k.startswith("val") or k.startswith("ho")}
base = res["base"]
rounds = [100, 200, 300, 400, 500]
rows = []
for arm, by in res.items():
    seeds = sorted(set(by) & set(base))
    if not seeds:
        continue
    V = {r: np.mean([by[s][f"val{r}"] for s in seeds], 0) for r in rounds}
    H = {r: np.mean([by[s][f"ho{r}"] for s in seeds], 0) for r in rounds}
    BV = {r: np.mean([base[s][f"val{r}"] for s in seeds], 0) for r in rounds}
    BH = {r: np.mean([base[s][f"ho{r}"] for s in seeds], 0) for r in rounds}
    r_arm = max(rounds, key=lambda r: V[r][tune].mean()); r_base = max(rounds, key=lambda r: BV[r][tune].mean())
    d200 = V[200] - BV[200]; h200 = H[200] - BH[200]
    dc = V[r_arm][conf] - BV[r_base][conf]
    dall = V[r_arm] - BV[r_base]
    se = lambda x: x.std() / np.sqrt(len(x))
    rows.append(dict(arm=arm, seeds=len(seeds), val200=V[200].mean(), d_val200=d200.mean(), se200=se(d200), d_ho200=h200.mean(),
                     se_ho200=se(h200), r_arm=r_arm, r_base=r_base, d_confirm=dc.mean(), se_confirm=se(dc),
                     d_short=dall[short].mean(), d_long=dall[~short].mean(),
                     val_best_curve=" ".join(f"{V[r].mean():.4f}" for r in rounds), ho_curve=" ".join(f"{H[r].mean():.4f}" for r in rounds)))
df = pl.DataFrame(rows).sort("d_confirm", descending=True)
pl.Config.set_tbl_rows(100); pl.Config.set_tbl_cols(30); pl.Config.set_tbl_width_chars(320); pl.Config.set_float_precision(5)
print(df)
df.write_csv(D / "12_gbdt_summary.csv")
