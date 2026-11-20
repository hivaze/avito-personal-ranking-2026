"""Context for the val -> holdout/public transfer gap (not a bug check): per-feed paired val deltas of the r6 families vs
F7 (root-averaged over 7 roots), split by segments that differ between val / holdout / test (M>=50 = holdout support,
cold share, day), and the src/pmix.py public/private reweighting."""
import sys

import numpy as np

sys.path.insert(0, "<repo>/src")
sys.path.insert(0, "<repo>/eda/errors/segments")
import polars as pl  # noqa: E402
from common import feed_meta  # noqa: E402
from metrics import ndcg_per_feed  # noqa: E402
from pmix import weights, wmean_se  # noqa: E402
from segdefs import seg_defs  # noqa: E402

R = "<repo>"
P = f"{R}/preds"
roots = range(101, 108)
FAM = {"F7": "v17_pre2s_s{r}", "F7x": "r6_cd_m02_r{r}", "F7x70": "r6_cd_m02_p70k_r{r}", "F7b": "r6_cd_c0_r{r}"}
_, M, off = feed_meta("val")
y = np.load(f"{R}/data/flat/val/cand_target.npy").astype(np.float64)
d = pl.read_parquet(f"{R}/eda/errors/segments/feed_table.parquet")
S = seg_defs(d)
W = weights()
segs = {"all": np.ones(len(M), bool), "M>=50": M >= 50, "M<50": M < 50}
for k in np.unique(S["cold_share"]):
    segs[k] = S["cold_share"] == k
for k in np.unique(S["day"]):
    segs[k] = S["day"] == k
for kind, fn in (("to", "trainonly_val"), ("ft", "n2cpv3_val_oof")):
    nd = {(f, r): ndcg_per_feed(np.load(f"{P}/nn_{p.format(r=r)}_{fn}.npy"), y, off, M) for f, p in FAM.items() for r in roots}
    print(f"\n== kind={kind}: root-averaged paired per-feed delta vs F7 (mean +- SE; n feeds)")
    for fam in ("F7b", "F7x", "F7x70"):
        dlt = np.mean([nd[(fam, r)] - nd[("F7", r)] for r in roots], 0)
        parts = [f"{s} {dlt[m].mean():+.5f}+-{dlt[m].std() / np.sqrt(m.sum()):.5f} (n{m.sum()})" for s, m in segs.items()]
        pm = [f"pmix-{k} {wmean_se(dlt, W[k])[0]:+.5f}+-{wmean_se(dlt, W[k])[1]:.5f}" for k in ("public", "private")]
        print(f"  {fam:6s} " + " | ".join(parts + pm))
    dlt = np.mean([nd[("F7x", r)] - nd[("F7b", r)] for r in roots], 0)
    print(f"  F7x-F7b (m02 increment): all {dlt.mean():+.5f}+-{dlt.std() / np.sqrt(len(dlt)):.5f} | M>=50 "
          f"{dlt[M >= 50].mean():+.5f}+-{dlt[M >= 50].std() / np.sqrt((M >= 50).sum()):.5f}")
