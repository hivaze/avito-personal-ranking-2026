"""Skeptic check for P4 FPCD-lite (no Sep 6-8 feeds in the m02 cooldown). CPU only, read-only on project files.
(1) Is the 'popularity two-sidedness' specific to Sep 6-8, or a smooth time gradient?  base - xdrop_pop / pop1 per-feed
    holdout NDCG deltas (4 seed-paired members) split Sep 6-8 / Sep 9-14 / Sep 17-19, with feed SEs and contrasts.
(2) How far does dropping Sep 6-8 move the cooldown's LUO-zero share toward val / test?  (20K random train-feed sample
    built by the pipeline lens, col 3 = it_hist_all_luo, raw.)
(3) How does dropping Sep 6-8 change the week-1 / week-2 production mix (Sep 12 production step, external findings 3.4)?
"""
import os, sys
for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "POLARS_MAX_THREADS"]:
    os.environ[k] = "12"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
from pathlib import Path
sys.path.insert(0, "<repo>/src")
from metrics import ndcg_per_feed

ROOT = Path("<repo>")
P = Path("/tmp/scratchpad/dpa/pipeline/out")

day_of = lambda ts: (ts.astype(np.int64) + 10800) // 86400

# ---------------- (1)
mh = np.load(P / "hold_meta.npz"); M, off, y = mh["M"], mh["off"], mh["y"].astype(np.float64)
d0 = day_of(mh["ts"]); dmin = d0.min(); dh = d0 - dmin
grp = {"Sep6-8": dh <= 2, "Sep9-14": (dh >= 3) & (dh <= 8), "Sep17-19": dh >= 11}


def per(fam):
    return np.mean([ndcg_per_feed(np.load(ROOT / f"preds/holdout/nn_{fam}_s{s}_holdout.npy"), y, off, M)
                    for s in (951, 952, 953, 954)], 0)


base = per("w5_e10_base")
for alt in ("abl_e10_xdrop_pop", "abl_e10_pop1"):
    d = base - per(alt)
    print(f"\n(1) holdout base - {alt}: all {d.mean():+.5f} ({d.std() / np.sqrt(len(d)):.5f})")
    st = {}
    for g, m in grp.items():
        st[g] = (d[m].mean(), d[m].std() / np.sqrt(m.sum()), m.sum())
        print(f"    {g:9s} n {m.sum():5d}  {st[g][0]:+.5f} ({st[g][1]:.5f})")
    for a, b in (("Sep6-8", "Sep9-14"), ("Sep6-8", "Sep17-19"), ("Sep9-14", "Sep17-19")):
        diff = st[a][0] - st[b][0]; se = np.hypot(st[a][1], st[b][1])
        print(f"    contrast {a} - {b}: {diff:+.5f} ({se:.5f}) z {diff / se:+.2f}")

# ---------------- (2)
rows = {}
for sub in ("trn", "hold", "val", "test"):
    m = np.load(P / f"{sub}_meta.npz"); R = np.load(P / f"{sub}_raw.npy", mmap_mode="r")
    luo = np.asarray(R[:, 3]); fid = np.repeat(np.arange(len(m["M"])), m["M"])
    rows[sub] = (m, luo, fid)
m, luo, fid = rows["trn"]
dt = day_of(m["ts"]) - dmin  # same day origin as the holdout (Sep 6 = 0)
dcand = dt[fid]
z = (luo == 0)
print("\n(2) LUO-zero share (candidate level), train sample by day (Sep 6 = 0): " +
      " ".join(f"d{d}:{z[dcand == d].mean():.3f}" for d in np.unique(dt)))
allz = z.mean(); noE = z[dcand >= 3].mean(); Ez = z[dcand <= 2].mean()
mv, luov, _ = rows["val"]; vz = (luov == 0).mean()
mt, luot, fidt = rows["test"]; pub = mt["ts"] < 1790497920
tz_pub = (luot[pub[fidt]] == 0).mean(); tz_pri = (luot[~pub[fidt]] == 0).mean()
mhh, luoh, _ = rows["hold"]; hz = (luoh == 0).mean()
print(f"    train all {allz:.4f} | Sep 6-8 {Ez:.4f} | without Sep 6-8 {noE:.4f} (shift {noE - allz:+.4f})")
print(f"    holdout {hz:.4f} | val {vz:.4f} | test public {tz_pub:.4f} | test private {tz_pri:.4f}")
print(f"    share of the train->val gap closed by FPCD: {(noE - allz) / (vz - allz):.3f}; train->private gap: {(noE - allz) / (tz_pri - allz):.3f}")
# feed-level counts by day
cnt = np.bincount(dt, minlength=dt.max() + 1)
print("    train-sample feeds by day: " + " ".join(f"d{d}:{c}" for d, c in enumerate(cnt) if c))

# ---------------- (3) production / exploration mix by week
pr = np.load(ROOT / "eda/labels_position/out/train_prandom.npz")
prand = np.nan_to_num(pr["p_random"].astype(np.float32), nan=0.0)[m["fidx"]]
prod = prand <= 0.9
w1 = dt <= 5; w2 = dt >= 6  # Sep 6-11 vs Sep 12-19 (production step on Sat Sep 12, external findings 3.4)
for lab, keep in (("all train", np.ones_like(dt, bool)), ("without Sep 6-8", dt >= 3)):
    k = keep & prod
    print(f"(3) {lab:16s}: production feeds week1 (Sep 6-11) {np.mean(w1[k]):.3f} / week2 (Sep 12-19) {np.mean(w2[k]):.3f};"
          f" production share of feeds {np.mean(prod[keep]):.3f}")
