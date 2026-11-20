"""web_shift probe s4: the ONE labelled read of the s3 predictions, by the rule in PREDECLARED.txt."""
import sys
from pathlib import Path
import numpy as np
import polars as pl
sys.path.insert(0, "<repo>/src"); sys.path.insert(0, "<repo>/feat_new")
from metrics import ndcg_per_feed
import hz_common as H
ROOT = Path("<repo>"); HERE = ROOT / "reports/TRICKS_1115/web_shift_probe"
V = [f"w5_e10_kd05w192_s{s}" for s in (951, 952, 953, 954)]
f = pl.read_parquet(ROOT / "data/flat/val/feeds.parquet")
M = f["M"].to_numpy().astype(np.int64); off = f["cand_off"].to_numpy().astype(np.int64); ts = f["timestamp"].to_numpy().astype(np.int64)
y = np.load(ROOT / "data/flat/val/cand_target.npy").astype(np.float64)
fid = np.repeat(np.arange(len(M)), M)
cnt = np.load(ROOT / "data/feats/val/it_hist_all_luo.npy")
dv = np.load(HERE / "cf/delta_val.npy")
def z(s):
    s = s.astype(np.float64); mu = np.bincount(fid, s, len(M)) / M
    sd = np.sqrt(np.maximum(np.bincount(fid, s * s, len(M)) / M - mu ** 2, 0)) + 1e-9
    return (s - mu[fid]) / sd[fid]
have = [v for v in V if all((HERE / "preds" / f"cpu_{v}_{a}.npy").exists() for a in ("base", "A", "B"))]
print("members read:", have)
P = {a: [np.load(HERE / "preds" / f"cpu_{v}_{a}.npy") for v in have] for a in ("base", "A", "B")}
pool = {a: np.mean([z(p) for p in P[a]], 0) for a in P}
nd = {a: ndcg_per_feed(pool[a], y, off, M) for a in pool}
day = H.day_of(ts); days = np.unique(day[np.bincount(day - day.min()).argsort()[-2:] + day.min()] if False else day)
big = [d for d in np.unique(day) if (day == d).sum() > 1000]
npos = np.bincount(fid, y, len(M)); coldpos = np.bincount(fid, y * (cnt <= 0), len(M))
chg = np.bincount(fid, (dv > 0.05).astype(float), len(M)) / M
def rd(d, m=None, name=""):
    x = d if m is None else d[m]
    return f"{name}{x.mean():+.5f} +- {x.std(ddof=1)/np.sqrt(len(x)):.5f} (z {x.mean()/(x.std(ddof=1)/np.sqrt(len(x))):+.2f}, n {len(x)})"
print(f"4-pool baseline (CPU) val NDCG {nd['base'].mean():.5f}")
for a in ("A", "B"):
    d = nd[a] - nd["base"]
    print(f"\nARM {a}: pool {nd[a].mean():.5f} | delta {rd(d)}")
    print("   M<50   ", rd(d, M < 50)); print("   M>=50  ", rd(d, M >= 50))
    for dd in big:
        print(f"   day {dd} ", rd(d, day == dd))
    if len(big) == 2:
        a1, a2 = d[day == big[0]], d[day == big[1]]
        se = np.sqrt(a1.var(ddof=1) / len(a1) + a2.var(ddof=1) / len(a2))
        print(f"   later day minus earlier day: {a2.mean() - a1.mean():+.5f} +- {se:.5f}")
    print("   feeds with all positives cold ", rd(d, (npos > 0) & (coldpos == npos)))
    print("   feeds with no cold positive   ", rd(d, (npos > 0) & (coldpos == 0)))
    print("   feeds with >=30% rows changed ", rd(d, chg >= 0.30)); print("   feeds with <10% rows changed  ", rd(d, chg < 0.10))
    top = lambda s: np.array([off[i] + np.argmax(s[off[i]:off[i] + M[i]]) for i in range(len(M))])
    print(f"   top-1 changed in {np.mean(top(pool[a]) != top(pool['base'])):.3f} of feeds")
    for v, pb, pa in zip(have, P["base"], P[a]):
        dm = ndcg_per_feed(pa, y, off, M) - ndcg_per_feed(pb, y, off, M)
        print(f"   member {v}: {rd(dm)}")
