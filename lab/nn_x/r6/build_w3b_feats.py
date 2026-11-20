raise SystemExit("disabled in the nn_x/r6 copy: writes into data/feats_new and nn/feat_cache (shared paths)")
"""W3b re-FT hygiene: derived val/test copies of fine-tune extra features under NEW names (never overwrite a name).

  pre_idmax32_has      1 where pre_idmax32 is present (55-60% NaN, otherwise only mean-imputed without a flag)
  pre_hmax32_c / _8_c  pre_hmax32 / pre_hmax8 with the -2 "no history" sentinel mapped to NaN (mean-imputed downstream)
  pre_hmax32_sent / _8 1 where the sentinel was
  it_hist_all_luo_zsl  signed log1p of it_hist_all_luo_z (heavy right tail up to ~17)
Pure per-row maps, no labels, no cross-row information. Written to data/feats_new/<split>/ and hard-linked into
nn/feat_cache/<split>/ (the only place nn/data.feat_path looks besides data/feats).
Run: CUDA_VISIBLE_DEVICES="" .venv/bin/python nn/build_w3b_feats.py
"""
import os

import numpy as np

ROOT = "<repo>"
for split in ("val", "test"):
    src = lambda n: np.load(f"{ROOT}/data/feats/{split}/{n}.npy").astype(np.float32)  # noqa: E731
    out = {}
    out["pre_idmax32_has"] = np.isfinite(src("pre_idmax32")).astype(np.float32)
    for h in ("pre_hmax32", "pre_hmax8"):
        x = src(h)
        s = x == -2.0
        assert np.isfinite(x).all() and x[~s].min() > -1.5, h
        out[f"{h}_c"] = np.where(s, np.nan, x).astype(np.float32)
        out[f"{h}_sent"] = s.astype(np.float32)
    z = src("it_hist_all_luo_z")
    out["it_hist_all_luo_zsl"] = (np.sign(z) * np.log1p(np.abs(z))).astype(np.float32)
    os.makedirs(f"{ROOT}/data/feats_new/{split}", exist_ok=True)
    os.makedirs(f"{ROOT}/nn/feat_cache/{split}", exist_ok=True)
    for n, x in out.items():
        dst, link = f"{ROOT}/data/feats_new/{split}/{n}.npy", f"{ROOT}/nn/feat_cache/{split}/{n}.npy"
        assert not os.path.exists(dst) and not os.path.exists(link) and not os.path.exists(
            f"{ROOT}/data/feats/{split}/{n}.npy"), n
        np.save(dst, x)
        os.link(dst, link)
        print(split, n, len(x), "nan share", round(float(np.isnan(x).mean()), 4), "mean", round(float(np.nanmean(x)), 4))
