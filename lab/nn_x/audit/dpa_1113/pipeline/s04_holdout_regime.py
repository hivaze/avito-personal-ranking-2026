"""Input pipeline audit, step 4: does the holdout's Sep 6-8 'pre2-missing' regime (an input state that never occurs on
val / test) distort what the holdout measures?  For every member with holdout preds and train-only val preds:
NDCG on holdout feeds Sep 6-8 (pre2 NaN; 26%) vs Sep 9-19 (pre2 present), Spearman of member orderings between the two
subsets and against val; family-level means; G3 on each subset.  Also: holdout H=0 feeds.  CPU only."""
import os, sys, glob
for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS"]:
    os.environ[k] = "12"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr
sys.path.insert(0, "<repo>/src")
from metrics import ndcg_per_feed

ROOT = Path("<repo>"); OUT = Path(__file__).resolve().parent / "out"
mh = np.load(OUT / "hold_meta.npz"); M, off = mh["M"], mh["off"]; y = mh["y"].astype(np.float64)
R = np.load(OUT / "hold_raw.npy", mmap_mode="r")
pre2nan = np.isnan(np.asarray(R[off, 30]))  # pre2_fe_s at the first row of each feed (feed-constant regime)
H = mh["H"]
mv = np.load(OUT / "val_meta.npz"); yv = mv["y"].astype(np.float64)
rows = []
for p in sorted(glob.glob(str(ROOT / "preds/holdout/nn_*_holdout.npy"))):
    m = Path(p).name[3:-12]
    if "@" in m:
        continue
    pv = ROOT / f"preds/nn_{m}_trainonly_val.npy"
    if not pv.exists():
        continue
    s = np.load(p)
    if len(s) != len(y):
        continue
    nd = ndcg_per_feed(s, y, off, M)
    v = ndcg_per_feed(np.load(pv), yv, mv["off"], mv["M"]).mean()
    rows.append((m, nd[pre2nan].mean(), nd[~pre2nan].mean(), nd.mean(), v))
names = [r[0] for r in rows]; a = np.array([r[1:] for r in rows])
print(f"holdout feeds: {len(M)}; Sep 6-8 (pre2 NaN) {pre2nan.sum()} ({pre2nan.mean():.3f}); H=0 {np.mean(H == 0):.4f} "
      f"(Sep 6-8 {np.mean(H[pre2nan] == 0):.4f}, Sep 9-19 {np.mean(H[~pre2nan] == 0):.4f}); M mean {M[pre2nan].mean():.1f} vs {M[~pre2nan].mean():.1f}")
print(f"members with holdout + train-only val preds: {len(rows)}")
print(f"Spearman over members: hold(Sep6-8) vs hold(Sep9-19) {spearmanr(a[:,0], a[:,1])[0]:.3f}; "
      f"val vs hold(Sep6-8) {spearmanr(a[:,3], a[:,0])[0]:.3f}; val vs hold(Sep9-19) {spearmanr(a[:,3], a[:,1])[0]:.3f}; val vs hold(all) {spearmanr(a[:,3], a[:,2])[0]:.3f}")
# modern members only (pre2s inputs, 10-epoch era): names containing pre2s / m02 / cd / w5 / abl
mod = np.array([("pre2s" in n) or ("m02" in n) or ("cd_" in n) or n.startswith(("w5_", "abl_", "v27", "v28", "v29", "v30")) for n in names])
b = a[mod]
print(f"modern members ({mod.sum()}): Spearman Sep6-8 vs Sep9-19 {spearmanr(b[:,0], b[:,1])[0]:.3f}; val vs Sep6-8 {spearmanr(b[:,3], b[:,0])[0]:.3f}; "
      f"val vs Sep9-19 {spearmanr(b[:,3], b[:,1])[0]:.3f}; val vs all {spearmanr(b[:,3], b[:,2])[0]:.3f}")
fam = {}
for n, r in zip(names, a):
    f = n.rsplit("_", 1)[0]; fam.setdefault(f, []).append(r)
print(f"{'family':28s} n  hold6-8  hold9-19  hold_all  val_TO   (9-19 minus 6-8)")
for f, rs in sorted(fam.items(), key=lambda kv: -np.mean([r[2] for r in kv[1]])):
    r = np.mean(rs, 0)
    print(f"{f:28s} {len(rs):2d} {r[0]:.5f} {r[1]:.5f} {r[2]:.5f} {r[3]:.5f}  {r[1]-r[0]:+.5f}")
