"""Step 0 (numeric part, label-free): how much of the existing it_hist_*_luo inputs comes from val / test history snapshots,
and what the banned in-split candidate count is. No labels are read.
  cd <repo> && CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 .venv/bin/python nn_x/w12/fresh/pregate/s0_facts.py
"""
import json
import sys

import numpy as np
import polars as pl

sys.path.insert(0, "<repo>/nn_x/w12/fresh/pregate")
from lib import FEATS, FLAT, OUT

NI = 111_400_000
res = {}
cnt = {}
for s in ("train", "val", "test"):
    hi = np.load(f"{FLAT}/{s}/hist_item_id.npy")
    cnt[s] = np.bincount(hi, minlength=NI).astype(np.float32)
    f = pl.read_parquet(f"{FLAT}/{s}/feeds.parquet")
    res[f"hist_rows_{s}"] = int(len(hi)); res[f"feeds_{s}"] = int(len(f)); res[f"mean_H_{s}"] = float(f["H"].mean())
    print(s, "history rows", len(hi), "feeds", len(f), "mean H", f["H"].mean(), flush=True)
fresh = cnt["val"] + cnt["test"]
for s in ("val", "test"):
    ci = np.load(f"{FLAT}/{s}/cand_item_id.npy")
    base = np.load(f"{FEATS}/{s}/it_hist_all.npy"); luo = np.load(f"{FEATS}/{s}/it_hist_all_luo.npy")
    tr = cnt["train"][ci]; fr = fresh[ci]
    assert np.allclose(base, tr + fr), "it_hist_all != train + val + test history counts"
    cold = np.load(f"{FEATS}/{s}/tte_it_imp.npy") == 0
    r = dict(rows=int(len(ci)), it_hist_all_is_train_plus_valtest_hist=True,
             mean_it_hist_all=float(base.mean()), mean_luo=float(luo.mean()), mean_from_train_hist=float(tr.mean()),
             mean_from_valtest_hist=float(fr.mean()), mass_share_valtest_hist=float(fr.sum() / base.sum()),
             rows_with_valtest_hist_ge1=float((fr >= 1).mean()), rows_luo_ge1=float((luo >= 1).mean()),
             rows_only_valtest_hist=float(((fr >= 1) & (tr == 0)).mean()),
             cold_rows_luo_ge1=float((luo[cold] >= 1).mean()), cold_rows_valtest_hist_ge1=float((fr[cold] >= 1).mean()),
             cold_rows_only_valtest_hist=float(((fr[cold] >= 1) & (tr[cold] == 0)).mean()),
             rows_luo_differs_from_all=float((luo != base).mean()))
    cc = np.load(f"{FEATS}/{s}/it_cand_cnt_split.npy")
    r["it_cand_cnt_split_zero_or_self_only"] = float((cc * (res[f"feeds_{s}"] / 1e4) <= 1.0001).mean())
    res[s] = r
    print(s, json.dumps(r, indent=1), flush=True)
json.dump(res, open(f"{OUT}/s0_facts.json", "w"), indent=1)
