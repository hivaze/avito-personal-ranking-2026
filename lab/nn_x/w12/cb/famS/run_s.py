"""famS arm runner (cbx.run_arm protocol: train-side fit / selection split, no val label before the final evaluation).
usage: CUDA_VISIBLE_DEVICES="" python -u run_s.py --tag T [--loss rmse|qsm|yeti] [--lr 0.1] [--iters 3000] [--depth 8] [--seed 0]
       [--block none|raw|pca|cand] [--grow SymmetricTree|Depthwise|Lossguide] [--mdl 0] [--budget_min 120] [--max_feeds 0] [--od_wait 0]"""
import argparse, os, sys, time, warnings
import numpy as np
sys.path.insert(0, "<repo>/nn_x/w12/cb/famS")
import fs_common as fs
from fs_common import cbx, log
warnings.filterwarnings("ignore", message=".*DataFrame is highly fragmented.*")
try:
    import pandas as pd
    warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
except Exception:  # noqa: BLE001
    pass

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--loss", default="rmse", choices=["rmse", "qsm", "yeti"])
ap.add_argument("--lr", type=float, default=0.1)
ap.add_argument("--iters", type=int, default=3000)
ap.add_argument("--depth", type=int, default=8)
ap.add_argument("--l2", type=float, default=5.0)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--block", default="none", choices=["none", "raw", "pca", "cand"])
ap.add_argument("--grow", default="SymmetricTree")
ap.add_argument("--mdl", type=int, default=0, help="min_data_in_leaf (Depthwise / Lossguide only)")
ap.add_argument("--max_leaves", type=int, default=0)
ap.add_argument("--budget_min", type=float, default=None)
ap.add_argument("--max_feeds", type=int, default=0)
ap.add_argument("--od_wait", type=int, default=0, help="0 = no early stop (od_wait = iterations)")
ap.add_argument("--kd_w", type=float, default=0.5, help="qsm only: label = (1 - kd_w) y / sum(y) + kd_w softmax(teacher)")
ap.add_argument("--rsm", type=float, default=0.0, help="0 = CatBoost default (1.0)")
ap.add_argument("--out", default=str(fs.PREDS))
a = ap.parse_args()
assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
from catboost import CatBoostRanker

tr = cbx.load("xm")
p = dict(depth=a.depth, learning_rate=a.lr, l2_leaf_reg=a.l2, border_count=254, iterations=a.iters, od_type="Iter",
         od_wait=a.od_wait or a.iters, eval_metric="NDCG:type=Base", use_best_model=False, random_seed=a.seed,
         allow_writing_files=False, thread_count=cbx.THREADS)
if a.grow != "SymmetricTree":
    p["grow_policy"] = a.grow
    if a.mdl:
        p["min_data_in_leaf"] = a.mdl
    if a.max_leaves:
        p["max_leaves"] = a.max_leaves
if a.rsm:
    p["rsm"] = a.rsm
extra = dict(loss=a.loss, block=a.block, args=vars(a))
if a.loss == "rmse":
    p["loss_function"] = "RMSE"
    label = tr.teacher("mean11", "centred")
elif a.loss == "yeti":
    p["loss_function"] = "YetiRank"
    label = tr.teacher("mean11", "centred")
else:
    p["loss_function"] = "QuerySoftMax"
    p["leaf_estimation_method"] = "Newton"
    label, n0 = fs.kd_soft_label(tr, a.kd_w, 1.0)
    extra["feeds_without_click"] = n0
    assert label.min() >= 0
pool_kw = dict(cat="native")
if a.block != "none":
    pool_kw["extra"] = fs.extra_cols(a.block)
log(f"[{a.tag}] params {p} block {a.block} label mean {float(np.mean(label)):.5f} sd {float(np.std(label)):.4f}")
m = CatBoostRanker(**p)
r = cbx.run_arm(a.tag, m, tr, label, a.out, pool_kw=pool_kw, budget_min=a.budget_min, verbose=50, extra=extra,
                max_feeds=a.max_feeds or None)
gp = m.get_all_params()
log(f"[{a.tag}] resolved: loss {gp.get('loss_function')} leaf_est {gp.get('leaf_estimation_method')} l2 {gp.get('l2_leaf_reg')} "
    f"bootstrap {gp.get('bootstrap_type')} subsample {gp.get('subsample')} grow {gp.get('grow_policy')} depth {gp.get('depth')} "
    f"mdl {gp.get('min_data_in_leaf')} rsm {gp.get('rsm')} max_ctr_complexity {gp.get('max_ctr_complexity')}")
if a.block != "none":
    imp = m.get_feature_importance(thread_count=cbx.THREADS)  # PredictionValuesChange, no data needed
    names = m.feature_names_
    nb = set(fs.block_names(a.block))
    share = float(sum(v for n, v in zip(names, imp) if n in nb) / max(imp.sum(), 1e-12))
    log(f"[{a.tag}] importance share of the new block ({len(nb)} columns): {share:.4f}; top new: "
        f"{sorted(((float(v), n) for n, v in zip(names, imp) if n in nb), reverse=True)[:8]}")
log(f"[{a.tag}] DONE")
