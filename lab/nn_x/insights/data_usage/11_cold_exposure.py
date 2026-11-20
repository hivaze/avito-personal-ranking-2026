"""Cold (never-seen, it_hist_all_luo == 0) candidates under production vs random-like display (train, M>=50):
share by shown rank, and click-rate ratio cold/warm. CPU, read-only."""
import numpy as np, polars as pl
from pathlib import Path
ROOT = Path("<repo>"); D = ROOT / "nn_x/insights/data_usage/out"
t = pl.read_parquet(D / "train_feeds.parquet")
M = t["M"].to_numpy().astype(np.int64); cls = t["cls"].to_numpy(); n = len(M)
fid = np.repeat(np.arange(n), M)
pos = np.load(ROOT / "data/flat/train/cand_pos.npy").astype(np.int64)
off = np.concatenate([[0], np.cumsum(M)[:-1]])
o = np.lexsort((pos, fid)); rank = np.empty(len(pos), np.int32); rank[o] = (np.arange(len(pos)) - off[fid[o]]); del o
rkend = M[fid] - 1 - rank
y = np.load(ROOT / "data/flat/train/cand_target.npy").astype(np.int8)
cold = np.load(ROOT / "data/feats/train/it_hist_all_luo.npy", mmap_mode="r")
cold = np.asarray(cold) == 0
cf = cls[fid]; Mc = M[fid]
rows = []
for nm, mk in [("prod M>=50", (cf == 0) & (Mc >= 50)), ("random-like M>=50", (cf >= 1)), ("prod M<50", (cf == 0) & (Mc < 50))]:
    mk2 = mk & (rkend >= 1)
    c, yy, rr = cold[mk2], y[mk2], rank[mk2]
    d = dict(cls=nm, cold_share=c.mean(), rate_cold=yy[c].mean(), rate_warm=yy[~c].mean())
    d["cold/warm rate"] = d["rate_cold"] / d["rate_warm"]
    for lo, hi in [(0, 1), (1, 3), (3, 10), (10, 50), (50, 300)]:
        s = (rr >= lo) & (rr < hi)
        d[f"cold_share r[{lo},{hi})"] = c[s].mean()
        d[f"cold/warm r[{lo},{hi})"] = yy[s & c].mean() / max(yy[s & ~c].mean(), 1e-9)
    rows.append(d)
df = pl.DataFrame(rows)
pl.Config.set_tbl_cols(30); pl.Config.set_tbl_width_chars(300); pl.Config.set_float_precision(4)
print(df.transpose(include_header=True, column_names=[r["cls"] for r in rows]))
df.write_csv(D / "11_cold_exposure.csv")
# val: cold share and cold/warm rate (random order)
yv = np.load(ROOT / "data/flat/val/cand_target.npy") > 0
cv = np.load(ROOT / "data/feats/val/it_hist_all_luo.npy") == 0
print(f"val: cold share {cv.mean():.4f}, rate cold {yv[cv].mean():.5f}, warm {yv[~cv].mean():.5f}, ratio {yv[cv].mean() / yv[~cv].mean():.3f}")
