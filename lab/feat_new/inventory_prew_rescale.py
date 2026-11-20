"""Clip-safe rescaled copies of the train-stage prew_* raw columns (featrev inventory, INVENTORY-1).

Why: nn/data.py SplitData.attach_xt clips every xt column that is not *_rk / *_z / seq_cv* / it_hist* to [-10, 10]
before the in-model within-feed rank / z (XT_REL) are computed. prew_fe and prew_new have median ~15 and p99 ~23, so
86% / 79% of present rows are clipped to exactly 10 and their in-model rank / z channels (the ONLY way cvnd_prew uses
them, XT_RELONLY in nn/train.py:51) collapse to ties. prew_hmax32 (range [-1, 1]) is unaffected.

Construction (identical for train / val / test, no labels, no cross-row information):
    prew_fe_s  = clip((prew_fe  + 20) / 6, 1e-3, 9.99)     NaN stays NaN
    prew_new_s = clip((prew_new + 20) / 6, 1e-3, 9.99)     NaN stays NaN
The affine map is fixed (not fitted), keeps every present value inside (0, 10) on all splits (raw ranges: train
[-15.35, 30.72], val [-9.60, 27.56], test [-10.82, 28.06] -> [0.78, 8.45]; checked at run time: the
clip bounds are never hit), and keeps NaN rows (which attach_xt maps to 0) BELOW all present rows, exactly the
ordering the raw columns had. Within-feed rank is therefore identical to the unclipped raw column; within-feed z is
an affine function of the raw z over present rows.

Output: data/feats_new/{train,val,test}/prew_fe_s.npy, prew_new_s.npy (float32, aligned with data/flat/<split>/cand_*).
Run: CUDA_VISIBLE_DEVICES="" python feat_new/inventory_prew_rescale.py
"""
import numpy as np
from pathlib import Path

ROOT = Path("<repo>")
SRC = ROOT / "data" / "feats"
DST = ROOT / "data" / "feats_new"

for split in ("train", "val", "test"):
    n_rows = len(np.load(ROOT / "data" / "flat" / split / "cand_item_id.npy", mmap_mode="r"))
    for name in ("prew_fe", "prew_new"):
        x = np.load(SRC / split / f"{name}.npy", mmap_mode="r")
        assert len(x) == n_rows, (split, name, len(x), n_rows)
        out = np.empty(n_rows, np.float32)
        lo, hi, nan = np.inf, -np.inf, 0
        step = 20_000_000
        for s in range(0, n_rows, step):
            v = np.asarray(x[s:s + step], np.float32)
            pres = ~np.isnan(v)
            nan += int((~pres).sum())
            if pres.any():
                lo, hi = min(lo, float(v[pres].min())), max(hi, float(v[pres].max()))
            y = (v + 20.0) / 6.0
            out[s:s + step] = np.where(pres, np.clip(y, 1e-3, 9.99), np.nan)
        (DST / split).mkdir(parents=True, exist_ok=True)
        np.save(DST / split / f"{name}_s.npy", out)
        print(f"{split} {name}_s rows {n_rows} NaN {nan / n_rows:.4f} raw range [{lo:.3f}, {hi:.3f}] -> "
              f"[{(lo + 20) / 6:.3f}, {(hi + 20) / 6:.3f}] clip-bound hit: {(lo + 20) / 6 < 1e-3 or (hi + 20) / 6 > 9.99}",
              flush=True)
