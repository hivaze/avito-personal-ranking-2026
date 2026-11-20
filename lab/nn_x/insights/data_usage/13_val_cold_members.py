"""Val NDCG deltas between training stages by positive coldness (label-defined slices; descriptive only):
ref (pool-only, 134K exploration feeds) -> snapshot @77500 (8 ep, production-dominated) -> F7 (cooldown mix 0.1) and
F7x (m02 re-cool). 4 roots averaged; paired per-feed SE."""
import numpy as np, polars as pl
from pathlib import Path
ROOT = Path("<repo>"); D = ROOT / "nn_x/insights/data_usage/out"
v = pl.read_parquet(D / "val_feed_losses.parquet")
M = v["M"].to_numpy(); n = len(M); fid = np.repeat(np.arange(n), M)
y = np.load(ROOT / "data/flat/val/cand_target.npy") > 0
luo = np.load(ROOT / "data/feats/val/it_hist_all_luo.npy")
npos = np.bincount(fid, weights=y, minlength=n)
cp = np.bincount(fid[y], weights=(luo[y] == 0), minlength=n) / np.maximum(npos, 1)
seg = np.where(cp == 0, "warm", np.where(cp >= 1, "allcold", "mixed"))
N = lambda k: np.mean([v[f"N_{k}{'' if k == 'snap' else '_'}{r}"].to_numpy() for r in (101, 102, 103, 104)], 0)
Ns = {"ref": v["N_ref"].to_numpy(), "snap": N("snap"), "f7": N("f7"), "f7x": N("f7x")}
rows = []
for a, b in [("ref", "snap"), ("snap", "f7"), ("snap", "f7x"), ("f7", "f7x")]:
    d = Ns[b] - Ns[a]
    r = dict(step=f"{a}->{b}", all=d.mean(), se=d.std() / np.sqrt(n))
    for s in ["warm", "mixed", "allcold"]:
        m = seg == s; r[s] = d[m].mean(); r[s + "_se"] = d[m].std() / np.sqrt(m.sum())
    r["M<50"] = d[M < 50].mean(); r["M>=50"] = d[M >= 50].mean()
    rows.append(r)
df = pl.DataFrame(rows); pl.Config.set_tbl_width_chars(250); pl.Config.set_float_precision(5); print(df)
print("feed shares:", {s: float(np.mean(seg == s)) for s in ["warm", "mixed", "allcold"]})
df.write_csv(D / "13_val_cold_members.csv")
