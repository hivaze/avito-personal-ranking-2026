"""critique2 N5: the G1 probe's decision rule compares the PUBLIC cost of zeroing the popularity family with the raw VAL cost
(ratio >= 0.8 -> 'carries over'; <= 0.5 -> 'over-reliance, promote hzcd'). The null for that ratio is not 1: public has a
different M x cold-share composition (73.5% cold>.3 feeds vs 42.7% on val), and the per-feed cost of zeroing popularity is
composition-dependent. This script takes the stored CPU replay of the popularity family zeroed (nn_x/audit/features/out/
replay/<member>_gpop_z.npy vs _base.npy) and computes the expected public / private cost under the M-mix (calib) and the
M x cold-share mix (pmix) reweightings. Also the stale-popularity replay (_st_pop.npy) for the same members.
CPU only. Writes out/c2_04_popcost_mix.log (stdout)."""
import os, glob
for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "POLARS_MAX_THREADS"]:
    os.environ[k] = "12"
import sys
from pathlib import Path
import numpy as np, polars as pl
ROOT = Path("<repo>"); sys.path.insert(0, str(ROOT / "src"))
from common import feed_meta, FLAT
from metrics import ndcg_per_feed
import calib, pmix
RP = ROOT / "nn_x/audit/features/out/replay"
fv, Mv, ov = feed_meta("val"); y = np.load(FLAT / "val/cand_target.npy")
Wpub, Wpriv = calib.weights("public"), calib.weights("private"); PM = pmix.weights(); Wpub2, Wpriv2 = PM["public"], PM["private"]
seg = pl.read_parquet(ROOT / "eda/errors/segments/feed_table.parquet"); cs = seg["cold_share"].to_numpy()
def wmean(d, w): return float(np.average(d, weights=w))
members = sorted({os.path.basename(p).replace("_base.npy", "") for p in glob.glob(str(RP / "*_base.npy"))})
print("members with replay files:", members)
for m in members:
    base = ndcg_per_feed(np.load(RP / f"{m}_base.npy"), y, ov, Mv)
    for tag in ["gpop_z", "gpop_p", "st_pop", "st_both"]:
        p = RP / f"{m}_{tag}.npy"
        if not p.exists():
            continue
        d = ndcg_per_feed(np.load(p), y, ov, Mv) - base
        se = d.std(ddof=1) / np.sqrt(len(d))
        v, pu, pr, pu2, pr2 = d.mean(), wmean(d, Wpub), wmean(d, Wpriv), wmean(d, Wpub2), wmean(d, Wpriv2)
        print(f"  {m} {tag:8s}: val {v:+.5f} ± {se:.5f} | M-mix public {pu:+.5f} (ratio {pu/v:.2f}) private {pr:+.5f} ({pr/v:.2f}) | "
              f"pmix public {pu2:+.5f} (ratio {pu2/v:.2f}) private {pr2:+.5f} ({pr2/v:.2f}) | cold>.3 {d[cs > 0.3].mean():+.5f} cold<=.3 {d[cs <= 0.3].mean():+.5f} M<50 {d[Mv < 50].mean():+.5f} M>=50 {d[Mv >= 50].mean():+.5f}")
print("Reading: G1's 'public cost / val cost' thresholds (0.8 / 0.5) must be compared with the composition-only null printed as the pmix public ratio, not with 1.")
