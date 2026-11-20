"""Dry run of cbx.run_arm (the standard arm protocol) on 3,000 feeds: a KD arm (RMSE on the teacher logit, selection by NDCG on clicks
of the train-side split) and a hard-label arm. Writes to out/smoke_arm/. usage: CUDA_VISIBLE_DEVICES="" CBX_THREADS=4 python -u smoke_arm.py"""
import sys
sys.path.insert(0, "<repo>/nn_x/w12/cb/infra")
import cbx
from catboost import CatBoostRanker
tr = cbx.load("xm"); ev = cbx.Evaluator(); out = cbx.HERE / "out/smoke_arm"
common = dict(depth=6, learning_rate=0.2, iterations=30, eval_metric="NDCG:type=Base", od_type="Iter", od_wait=20, allow_writing_files=False,
              thread_count=cbx.THREADS, random_seed=0)
a = cbx.run_arm("dry_kd_rmse", CatBoostRanker(loss_function="RMSE", **common), tr, tr.teacher("mean11", "centred"), out, max_feeds=3000, ev=ev, verbose=10)
b = cbx.run_arm("dry_hard_yeti", CatBoostRanker(loss_function="YetiRank", **common), tr, tr.y, out, max_feeds=3000, ev=ev, verbose=10)
import numpy as np
ev.compare(np.load(out / "dry_kd_rmse_val.npy"), np.load(out / "dry_kd_rmse_hold.npy"), np.load(out / "dry_hard_yeti_val.npy"), np.load(out / "dry_hard_yeti_hold.npy"),
           label="dry_kd_rmse - dry_hard_yeti")
print("SMOKE_ARM OK")
