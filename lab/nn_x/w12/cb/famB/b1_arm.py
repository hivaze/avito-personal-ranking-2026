"""B1 / B2: native CatBoost CTRs on CANDIDATE-level keys, B0 recipe otherwise (YetiRank d8 lr 0.2 l2 5, train-side selection, cbx.run_arm).
  --keys clu      : cl256, cl1024, cl4096 as native categoricals next to category_id (B1)
  --keys item     : item_id as a native categorical (B2; use --set xm+xs so that the statistics see more rows)
  has_time=True, feeds in time order (file order), rows SHUFFLED inside each feed (ShufDS): the ordered statistic of a row sees earlier
  feeds and a random part of its own feed. No feed- or user-constant key is native (docs_feat 5.3).
usage: python -u b1_arm.py --tag b1_mcc4_s0 --keys clu --mcc 4 [--seed 0] [--set xm] [--max_feeds 3000] [--budget_min 95] [--iters 520]"""
import argparse, json, sys
sys.path.insert(0, "<repo>/nn_x/w12/cb/famB")
from fb_common import *

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True); ap.add_argument("--keys", choices=["clu", "item", "clu+item"], required=True)
ap.add_argument("--mcc", type=int, default=4); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--set", default="xm")
ap.add_argument("--max_feeds", type=int, default=0); ap.add_argument("--budget_min", type=float, default=None); ap.add_argument("--iters", type=int, default=ITERS)
ap.add_argument("--msr", type=float, default=None, help="model_size_reg (B2: 0)"); ap.add_argument("--leaf_limit", type=int, default=None, help="ctr_leaf_count_limit")
ap.add_argument("--out", default=str(PREDS)); ap.add_argument("--ctr", choices=["255", "default"], default="255")
a = ap.parse_args()

base = cbx.load("xm") if a.set == "xm" else TimeCat([cbx.load(s) for s in a.set.split("+")])
sh = ShufDS(base, seed=a.seed)
extra, extra_cat = {}, []
if "clu" in a.keys:
    for K in (256, 1024, 4096):
        extra[f"cl{K}"] = (lambda d, K=K: clu(d, K)); extra_cat.append(f"cl{K}")
if "item" in a.keys:
    extra["item_id"] = (lambda d: d.item_id); extra_cat.append("item_id")
kw = dict(has_time=True, max_ctr_complexity=a.mcc)
if a.ctr == "255":
    kw.update(simple_ctr=CTR255, combinations_ctr=CTR255)
if a.msr is not None:
    kw["model_size_reg"] = a.msr
if a.leaf_limit is not None:
    kw["ctr_leaf_count_limit"] = a.leaf_limit
m = b0_model(a.seed, iterations=a.iters, **kw)
log(f"[{a.tag}] set {sh.name} feeds {sh.n_feeds} rows {sh.n_rows} keys {extra_cat} params {kw}")
r = cbx.run_arm(a.tag, m, sh, sh.y, a.out, pool_kw=dict(cat="native", extra=extra, extra_cat=extra_cat), budget_min=a.budget_min,
                max_feeds=a.max_feeds or None, verbose=25, extra=dict(args=vars(a), leak="native CTR keys are candidate-level only; has_time with "
                "feeds in time order and rows shuffled inside feeds; statistics come from the training pool only (fit feeds: no holdout feed, no val / test row)"))
imp = sorted(zip(m.get_feature_importance(type="PredictionValuesChange"), m.feature_names_), reverse=True)[:30]
log("top importance:", json.dumps([(n, round(float(x), 2)) for x, n in imp]))
log("extra-key importance:", json.dumps({n: round(float(x), 3) for x, n in zip(m.get_feature_importance(type="PredictionValuesChange"), m.feature_names_) if n in extra_cat + ["category_id"]}))
if not a.max_feeds:
    nats_line(a.tag, np.load(f"{a.out}/{a.tag}_val.npy"), np.load(f"{a.out}/{a.tag}_hold.npy"))
    ev = cbx.Evaluator()
    ev.compare(np.load(f"{a.out}/{a.tag}_val.npy"), np.load(f"{a.out}/{a.tag}_hold.npy"), np.load(B0V), np.load(B0H), label=f"{a.tag} - B0(ts_lr020_s0)")
log("DONE", a.tag)
