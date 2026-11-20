"""Skeptic T5: re-count the T5 evaluation sample, the training data a <=20710 truncation removes,
M<50 random-like availability, and per-feed contrast sd on the holdout's late days. CPU only."""
import os
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "POLARS_MAX_THREADS"):
    os.environ[k] = "12"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np, polars as pl
from pathlib import Path
ROOT = Path("<repo>")
VD = Path("/tmp/scratchpad/dpa/validation/out")
f = pl.read_parquet(ROOT / "data/flat/train/feeds.parquet")
print("train cols", f.columns)
M = f["M"].to_numpy().astype(np.int64)
ts = f["timestamp"].to_numpy().astype(np.int64) if "timestamp" in f.columns else f["ts"].to_numpy().astype(np.int64)
d = (ts + 10800) // 86400
pr = np.load(ROOT / "eda/labels_position/out/train_prandom.npz")
p_raw = pr["p_random"].astype(np.float32)
print("p_random NaN share overall %.4f, among M<50 %.4f, among M>=50 %.4f" % (np.isnan(p_raw).mean(), np.isnan(p_raw[M < 50]).mean(), np.isnan(p_raw[M >= 50]).mean()))
p = np.nan_to_num(p_raw, nan=0.0)
rl = (p > 0.9) & (M >= 50)
rl_small = (p > 0.9) & (M < 50)
cand = np.where(rl)[0]
hold = np.sort(np.random.default_rng(12345).choice(cand, 30000, replace=False))
Mh, offh = np.load(ROOT / "preds/holdout/holdout_meta.npy")
assert (M[hold] == Mh).all()
ish = np.zeros(len(M), bool); ish[hold] = True
print(f"train feeds {len(M):,}; candidates {M.sum():,}")
print("day | feeds | cands | M>=50 rl | of which holdout | M<50 with p>0.9 | M<50 feeds")
for dd in np.unique(d):
    m = d == dd
    print(f"{dd} | {m.sum():8,d} | {M[m].sum():10,d} | {(rl & m).sum():6,d} | {(ish & m).sum():5,d} | {(rl_small & m).sum():6,d} | {((M < 50) & m).sum():8,d}")
late = d >= 20713
keep = d <= 20710
print(f"T5 eval sample (M>=50 rl, days>=20713): {(rl & late).sum():,} = holdout {(ish & late).sum():,} + rest {(rl & late & ~ish).sum():,}")
print(f"truncation <=20710 keeps feeds {keep.sum():,} ({keep.mean():.3f}), candidates {M[keep].sum() / M.sum():.3f}; removes {(~keep).sum():,} feeds")
print(f"public-like truncation <=20708 keeps feeds {(d <= 20708).mean():.3f}, candidates {M[d <= 20708].sum() / M.sum():.3f}")
# per-feed contrast sd on holdout late days, for the MDE claim
hd = d[hold]
G3 = np.load(VD / "pfh_G3.npy")
for c in ["Q2a", "Q1a", "M2", "PF7x", "Cd"]:
    x = np.load(VD / f"pfh_{c}.npy") - G3
    for lab, msk in [("all", np.ones(len(hd), bool)), ("late3d", hd >= 20713)]:
        sd = x[msk].std(ddof=1)
        print(f"  {c}-G3 holdout {lab:6s} n {msk.sum():6d} mean {x[msk].mean():+.5f} sd {sd:.4f} SE {sd/np.sqrt(msk.sum()):.5f} | MDE(2.8SE) at n=24,991: {2.8*sd/np.sqrt(24991):.5f}")
