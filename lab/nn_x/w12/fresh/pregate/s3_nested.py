"""Step 3: nested residual read (PRIMARY family, DECLARATION.txt section C). 5 user-grouped folds x 3 fold seeds.

  control = [b, b*lm, cold, cold*b, cold*lm]; arm = control + 8 columns per window; arms A (as-of), B (b, c, d), ALL, L (later-only).
Writes out/s3_nested.json, out/s3_perfeed.npz (per-feed d_f of every arm, per-seed control NDCG) and prints the slices.
  cd <repo> && CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 NUMBA_NUM_THREADS=12 .venv/bin/python nn_x/w12/fresh/pregate/s3_nested.py
"""
import json
import sys
import time

import numpy as np

sys.path.insert(0, "<repo>/nn_x/w12/fresh/pregate")
from lib import OUT
from nested import ARMS, FOLD_SEEDS, Val, gain, nested_scores, scale, wstat

sys.path.insert(0, "<repo>/src")
import pmix

t0 = time.time()
V = Val()
print(f"SANITY pool score b: plain val NDCG {V.base_ndcg.mean():.5f} (FD16P20 build_r4: 0.40271)", flush=True)
assert abs(V.base_ndcg.mean() - 0.40271) < 2e-5
W = pmix.weights()
import polars as pl
ftab = pl.read_parquet("<repo>/eda/errors/segments/feed_table.parquet")
assert np.abs(ftab["coldcand_share"].to_numpy() - V.cold_share).max() < 1e-6, "no-train-impression share differs from the feed table's coldcand_share"
cs_pmix = ftab["cold_share"].to_numpy()  # pmix variable: never seen in train as candidate or in a history
print(f"pmix: mean feed cold share val {W['cs_val']:.3f} public {W['cs_pub']:.3f} private {W['cs_priv']:.3f}", flush=True)
C = {"main": np.load(f"{OUT}/cnt_main.npy")[:V.N], "nzsrc": np.load(f"{OUT}/cnt_nzsrc.npy")[:V.N]}

# slices (per feed)
csb = np.digitize(V.cold_share, [0.15, 0.30, 0.45], right=True)  # pmix bins (e[i-1], e[i]]
tod = (V.ts + 10800) % 86400
third = np.zeros(V.nf, np.int64)
for d in (20718, 20720):
    m = V.day == d
    q = np.quantile(tod[m], [1 / 3, 2 / 3]); third[m] = np.digitize(tod[m], q)
csp = np.digitize(cs_pmix, [0.15, 0.30, 0.45], right=True)
SL = {"cold<15%": csb == 0, "cold15-30%": csb == 1, "cold30-45%": csb == 2, "cold>=45%": csb == 3,
      "pmixcold<15%": csp == 0, "pmixcold15-30%": csp == 1, "pmixcold30-45%": csp == 2, "pmixcold>=45%": csp == 3,
      "day20718": V.day <= 20719, "day20720": V.day >= 20720, "M<50": V.M < 50, "M>=50": V.M >= 50,
      "uid0": V.uid == 0, "uid!=0": V.uid != 0,
      "d18:third1": (V.day == 20718) & (third == 0), "d18:third2": (V.day == 20718) & (third == 1), "d18:third3": (V.day == 20718) & (third == 2),
      "d20:third1": (V.day == 20720) & (third == 0), "d20:third2": (V.day == 20720) & (third == 1), "d20:third3": (V.day == 20720) & (third == 2)}
print("slice sizes: " + ", ".join(f"{k} {int(v.sum())}" for k, v in SL.items()), flush=True)

Xc, cn = V.control(); Xc, _ = scale(Xc)
mats = {"pool_only": None, "control": Xc}
names = {"control": cn}
for tag in ("main", "nzsrc"):
    for arm in ARMS:
        X, nm = V.arm_matrix(C[tag], arm); X, _ = scale(X)
        mats[f"{tag}:{arm}"] = X; names[f"{tag}:{arm}"] = nm
print(f"design built {time.time() - t0:.0f}s; columns: " + ", ".join(f"{k} {0 if v is None else v.shape[1]}" for k, v in mats.items()), flush=True)
out, thetas = nested_scores(V, mats, verbose=True)
print(f"nested fits done {time.time() - t0:.0f}s", flush=True)

res = {"base_ndcg": float(V.base_ndcg.mean()), "control_ndcg_by_seed": [float(x.mean()) for x in out["control"]], "arms": {}}
d0, g0, se0, z0 = gain(out["control"], out["pool_only"])
dpo, gpo, sepo, zpo = gain(out["pool_only"], [V.base_ndcg] * len(FOLD_SEEDS))
print(f"\npool_only (beta*b, monotone in b) vs raw b: {gpo:+.6f} (must be 0)   |   control - pool_only: {g0:+.5f} +- {se0:.5f} (z {z0:+.2f}) "
      f"[what M / cold flag alone add; not a result of this task]")
res["control_minus_pool"] = [g0, se0, z0]
save = {"control_seed%d" % s: out["control"][i] for i, s in enumerate(FOLD_SEEDS)}
hdr = f"{'arm':12s} {'GAIN':>9s} {'SE':>8s} {'z':>6s} | {'seed0':>8s} {'seed1':>8s} {'seed2':>8s} | {'pmix-pub':>18s} {'pmix-priv':>18s}"
print("\n=== NESTED GAIN vs control (plain val, paired per-feed SE; 14,423 feeds; mean of 3 fold seeds per feed)")
print(hdr)
for k in mats:
    if k in ("pool_only", "control"):
        continue
    d, g, se, z = gain(out[k], out["control"])
    ps = [float((a - c).mean()) for a, c in zip(out[k], out["control"])]
    pub, priv = wstat(d, W["public"]), wstat(d, W["private"])
    sl = {s: [float(d[m].mean()), float(d[m].std() / np.sqrt(m.sum())), int(m.sum())] for s, m in SL.items()}
    res["arms"][k] = dict(gain=g, se=se, z=z, per_seed=ps, pmix_public=pub, pmix_private=priv, slices=sl,
                          n_cols=int(mats[k].shape[1]))
    save["d_" + k.replace(":", "_")] = d
    print(f"{k:12s} {g:+9.5f} {se:8.5f} {z:+6.2f} | {ps[0]:+8.5f} {ps[1]:+8.5f} {ps[2]:+8.5f} | {pub[0]:+.5f}+-{pub[1]:.5f} {priv[0]:+.5f}+-{priv[1]:.5f}")
print("\n=== SLICES (gain +- SE [feeds]) ")
for grp in (["cold<15%", "cold15-30%", "cold30-45%", "cold>=45%"], ["pmixcold<15%", "pmixcold15-30%", "pmixcold30-45%", "pmixcold>=45%"], ["day20718", "day20720"], ["M<50", "M>=50"], ["uid0", "uid!=0"],
            ["d18:third1", "d18:third2", "d18:third3"], ["d20:third1", "d20:third2", "d20:third3"]):
    print("  " + " | ".join(f"{s} [{int(SL[s].sum())}]" for s in grp))
    for k in res["arms"]:
        print(f"    {k:12s} " + " | ".join(f"{res['arms'][k]['slices'][s][0]:+.5f}+-{res['arms'][k]['slices'][s][1]:.5f}" for s in grp))
# later-feed share (step 6 G1)
print("\n=== LATER-FEED SHARE (step 6): differences of per-feed gains, paired")
for tag in ("main", "nzsrc"):
    for x, y_ in (("B", "A"), ("ALL", "A"), ("ALL", "B")):
        dd = save[f"d_{tag}_{x}"] - save[f"d_{tag}_{y_}"]
        r = [float(dd.mean()), float(dd.std() / np.sqrt(len(dd)))]
        res.setdefault("later_share", {})[f"{tag}:{x}-{y_}"] = r
        print(f"  {tag}: GAIN({x}) - GAIN({y_}) = {r[0]:+.5f} +- {r[1]:.5f} (z {r[0] / r[1]:+.2f})")
# coefficients of the full-val refit (information): mean theta over the 15 fits
print("\n=== mean coefficients over the 15 fits (scaled columns), main arms")
for k in ("control", "main:A", "main:B", "main:ALL", "main:L"):
    th = np.mean(thetas[k], 0); sdv = np.std(thetas[k], 0)
    res.setdefault("theta", {})[k] = dict(names=["beta"] + names[k], mean=th.tolist(), sd=sdv.tolist())
    print(f"  {k}: " + ", ".join(f"{n} {v:+.3f}" for n, v in zip(["beta"] + names[k], th)))
np.savez(f"{OUT}/s3_perfeed.npz", **save)
np.savez(f"{OUT}/s3_thetas.npz", **{k.replace(":", "_"): np.array(v) for k, v in thetas.items()})
json.dump(res, open(f"{OUT}/s3_nested.json", "w"), indent=1)
print(f"\nDONE {time.time() - t0:.0f}s")
