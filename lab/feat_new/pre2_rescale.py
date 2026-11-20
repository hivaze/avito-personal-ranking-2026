"""Clip-safe rescaled copies of the train-stage pre2_* raw columns (FEATURES_REVIEW, finding FR-1).

Why: nn/data.py SplitData.attach_xt clips every xt column that is not *_rk / *_z / seq_cv* / it_hist* to [-10, 10]
before the in-model within-feed rank / z (XT_REL) are computed. The cvnd_pre2 preset (nn/train.py:43-46) feeds raw
pre2_fe / pre2_new through XT_RELONLY, i.e. ONLY via that rank / z. Their present values have median ~15-17 and
p99 ~22-24, so 92.5% / 87.9% of present train rows and 83.8% / 75.9% of val rows sit at exactly 10: the rank channel
runs on ties, and the tie share differs between train and val (the PREP-1 / INVENTORY-1 / RESEARCH-1 prew bug again).

Construction (identical for train / val / test, no labels, no cross-row information; same map as
feat_new/inventory_prew_rescale.py so that the prew and pre2 presets stay comparable):
    pre2_fe_s  = clip((pre2_fe  + 20) / 6, 1e-3, 9.99)     NaN stays NaN
    pre2_new_s = clip((pre2_new + 20) / 6, 1e-3, 9.99)     NaN stays NaN
The map is fixed (not fitted). The run asserts that the clip bounds are never hit on any split, so the within-feed
rank equals the rank of the unclipped raw column, the within-feed z is the raw z over present rows, and NaN rows
(attach_xt maps NaN -> 0) stay BELOW every present row, the ordering the raw columns had.

Output: data/feats_new/{train,val,test}/pre2_fe_s.npy, pre2_new_s.npy (float32, aligned with data/flat/<split>/cand_*).
Run: CUDA_VISIBLE_DEVICES="" .venv/bin/python feat_new/pre2_rescale.py
"""
import numpy as np
from pathlib import Path

ROOT = Path("<repo>")
SRC = ROOT / "data" / "feats"
DST = ROOT / "data" / "feats_new"

for split in ("train", "val", "test"):
    n_rows = len(np.load(ROOT / "data" / "flat" / split / "cand_item_id.npy", mmap_mode="r"))
    for name in ("pre2_fe", "pre2_new"):
        x = np.load(SRC / split / f"{name}.npy", mmap_mode="r")
        assert len(x) == n_rows, (split, name, len(x), n_rows)
        out = np.empty(n_rows, np.float32)
        lo, hi, nan, sat = np.inf, -np.inf, 0, 0
        step = 20_000_000
        for s in range(0, n_rows, step):
            v = np.asarray(x[s:s + step], np.float32)
            pres = ~np.isnan(v)
            nan += int((~pres).sum())
            sat += int((v[pres] >= 10).sum())
            if pres.any():
                lo, hi = min(lo, float(v[pres].min())), max(hi, float(v[pres].max()))
            y = (v + 20.0) / 6.0
            out[s:s + step] = np.where(pres, np.clip(y, 1e-3, 9.99), np.nan)
        assert (lo + 20) / 6 > 1e-3 and (hi + 20) / 6 < 9.99, (split, name, lo, hi)
        assert np.array_equal(np.isnan(out), np.isnan(np.asarray(x))), (split, name)
        (DST / split).mkdir(parents=True, exist_ok=True)
        np.save(DST / split / f"{name}_s.npy", out)
        n_pres = n_rows - nan
        print(f"{split} {name}_s rows {n_rows} NaN {nan / n_rows:.4f} raw range [{lo:.3f}, {hi:.3f}] -> "
              f"[{(lo + 20) / 6:.3f}, {(hi + 20) / 6:.3f}]; raw present rows >= 10 (clipped by attach_xt today): "
              f"{sat / max(n_pres, 1):.4f}", flush=True)
