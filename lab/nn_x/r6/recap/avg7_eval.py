"""Evaluate recap_avg7 outputs (CPU fp32 preds in nn_x/r6/preds): single-member pooled paired deltas avg3 - cooled,
ensembles (7 avg3 vs 7 cooled vs 14 pool), seed correlations; reference = stored GPU preds of the same cooled ckpts."""
import os
import sys

import numpy as np
import polars as pl

sys.path.insert(0, "<repo>/src")
from metrics import ndcg_per_feed  # noqa: E402

R = "<repo>"
P6 = f"{R}/nn_x/r6/preds"
f = pl.read_parquet(f"{R}/data/flat/val/feeds.parquet")
M = f["M"].to_numpy().astype(np.int64)
off = f["cand_off"].to_numpy().astype(np.int64)
y = np.load(f"{R}/data/flat/val/cand_target.npy").astype(np.float64)
day = (f["timestamp"].to_numpy() + 3 * 3600) // 86400
day = np.where(day <= day.min() + 1, 0, 1)
fid = np.repeat(np.arange(len(M)), M)
HM, HO = np.load(f"{R}/preds/holdout/holdout_meta.npy")
HY = np.load(f"{R}/preds/holdout/holdout_y.npy").astype(np.float64)
hfid = np.repeat(np.arange(len(HM)), HM)
SL = {"M<50": M < 50, "M>=50": M >= 50, "Tue": day == 0, "Thu": day == 1}


def fz(s, fid_, M_):
    s = s.astype(np.float64)
    mu = np.bincount(fid_, s, len(M_)) / M_
    sd = np.sqrt(np.maximum(np.bincount(fid_, s * s, len(M_)) / M_ - mu ** 2, 0)) + 1e-6
    return (s - mu[fid_]) / sd[fid_]


seeds = [s for s in range(101, 108) if os.path.exists(f"{P6}/nn_r6avg3_s{s}_holdout.npy")
         and os.path.exists(f"{P6}/nn_r6cool_s{s}_holdout.npy")]
print("seeds with both files:", seeds)
V, H = {}, {}
for tag in ("r6avg3", "r6cool"):
    for s in seeds:
        v = np.load(f"{P6}/nn_{tag}_s{s}_trainonly_val.npy")
        h = np.load(f"{P6}/nn_{tag}_s{s}_holdout.npy")
        V[(tag, s)] = (ndcg_per_feed(v, y, off, M), fz(v, fid, M))
        H[(tag, s)] = (ndcg_per_feed(h, HY, HO, HM), fz(h, hfid, HM))
# GPU-stored cooled preds (bf16) for the precision offset
for s in seeds:
    g = np.load(f"{R}/preds/nn_v17_pre2s_s{s}_trainonly_val.npy")
    print(f"s{s}: avg3 val {V[('r6avg3', s)][0].mean():.5f} hold {H[('r6avg3', s)][0].mean():.5f} | cooled(cpu) val "
          f"{V[('r6cool', s)][0].mean():.5f} hold {H[('r6cool', s)][0].mean():.5f} | cooled(gpu-bf16) val "
          f"{ndcg_per_feed(g, y, off, M).mean():.5f} | delta val {V[('r6avg3', s)][0].mean() - V[('r6cool', s)][0].mean():+.5f}"
          f" hold {H[('r6avg3', s)][0].mean() - H[('r6cool', s)][0].mean():+.5f}")
dv = np.mean([V[("r6avg3", s)][0] - V[("r6cool", s)][0] for s in seeds], 0)
dh = np.mean([H[("r6avg3", s)][0] - H[("r6cool", s)][0] for s in seeds], 0)
n = len(seeds)
se_h = dh.std() / np.sqrt(len(dh))
print(f"\nPOOLED single member avg3 - cooled ({n} seeds): val {dv.mean():+.5f} +- {dv.std() / np.sqrt(len(dv)):.5f} "
      f"(z {dv.mean() / (dv.std() / np.sqrt(len(dv))):.2f}) | " + " ".join(f"{k} {dv[m].mean():+.5f}" for k, m in SL.items()))
print(f"  holdout {dh.mean():+.5f} +- {se_h:.5f} feed-SE; calibrated SE {np.sqrt(se_h ** 2 + 0.00085 ** 2 / n):.5f}")
print(f"  seeds positive on val: {sum(V[('r6avg3', s)][0].mean() > V[('r6cool', s)][0].mean() for s in seeds)}/{n}; "
      f"on holdout: {sum(H[('r6avg3', s)][0].mean() > H[('r6cool', s)][0].mean() for s in seeds)}/{n}")


def ens(keys, D, yy, oo, MM):
    return ndcg_per_feed(np.mean([D[k][1] for k in keys], 0), yy, oo, MM)


ka = [("r6avg3", s) for s in seeds]
kc = [("r6cool", s) for s in seeds]
ea, ec, ep = ens(ka, V, y, off, M), ens(kc, V, y, off, M), ens(ka + kc, V, y, off, M)
ha, hc, hp = ens(ka, H, HY, HO, HM), ens(kc, H, HY, HO, HM), ens(ka + kc, H, HY, HO, HM)
print(f"\nENSEMBLE ({n} members each): val avg3 {ea.mean():.5f} | cooled {ec.mean():.5f} | pool {ep.mean():.5f}; "
      f"avg3-cooled {np.mean(ea - ec):+.5f} +- {np.std(ea - ec) / np.sqrt(len(ea)):.5f}; pool-cooled {np.mean(ep - ec):+.5f} +- "
      f"{np.std(ep - ec) / np.sqrt(len(ep)):.5f}")
print(f"  holdout avg3 {ha.mean():.5f} | cooled {hc.mean():.5f} | pool {hp.mean():.5f}; avg3-cooled {np.mean(ha - hc):+.5f} +- "
      f"{np.std(ha - hc) / np.sqrt(len(ha)):.5f}; pool-cooled {np.mean(hp - hc):+.5f}")
print("  slices avg3-cooled ens val: " + " ".join(f"{k} {(ea - ec)[m].mean():+.5f}" for k, m in SL.items()))


def corr(keys):
    zs = [V[k][1] for k in keys]
    c = [np.corrcoef(zs[i], zs[j])[0, 1] for i in range(len(zs)) for j in range(i + 1, len(zs))]
    return np.mean(c), np.std(c, ddof=1) if len(c) > 1 else 0
print(f"\nwithin-feed seed corr: avg3 {corr(ka)[0]:.4f} (sd {corr(ka)[1]:.4f}); cooled {corr(kc)[0]:.4f} (sd {corr(kc)[1]:.4f}); "
      f"avg3 vs own cooled: {np.mean([np.corrcoef(V[('r6avg3', s)][1], V[('r6cool', s)][1])[0, 1] for s in seeds]):.4f}")
