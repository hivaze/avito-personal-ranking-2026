# Family B, arm B3: the reference CatBoost ranker (YetiRank, depth 8, learning rate 0.2, the 177
# features) plus the hand-made label statistics of the chosen blocks; the number of trees is chosen
# on a train-side split, never on val.
# How it was run (docstring): python -u b3_arm.py --tag b3_abc_s0 --blocks a b c [--seed 0]
# [--budget_min 100]   (CPU, 12 threads, 101 min for seed 0). Its helper fb_common.py is not copied.
# What it produced: predictions for val, holdout and test and the screen line in
# results/famB_final_lines.txt (single model: val 0.38362, holdout 0.30480; member rule pooled z
# +0.30, the rule needs +2).
# Kept as it ran: paths and names refer to the lab tree (see lab/ and docs/glossary.md).
# Original: lab/nn_x/w12/cb/famB/b3_arm.py
"""B3: the B0 recipe (YetiRank d8 lr 0.2 l2 5, 177 features, category_id native with its default statistics, train-side selection through
cbx.run_arm) + hand-made label-aware columns of the chosen blocks (feat/<block>_<set>.npy, built by b3_build_{a,b,c}.py).
usage: python -u b3_arm.py --tag b3_abc_s0 --blocks a b c [--drop col,col] [--seed 0] [--max_feeds 3000] [--budget_min 100]"""
import argparse, json, sys
sys.path.insert(0, "<repo>/nn_x/w12/cb/famB")
from fb_common import *

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True); ap.add_argument("--blocks", nargs="+", required=True); ap.add_argument("--drop", default="")
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--max_feeds", type=int, default=0); ap.add_argument("--budget_min", type=float, default=None)
ap.add_argument("--iters", type=int, default=ITERS); ap.add_argument("--out", default=str(PREDS))
ap.add_argument("--native_clu", action="store_true", help="B13: also the B1 inputs (cl256 / cl1024 / cl4096 native, has_time, rows shuffled inside feeds, CTR255)")
ap.add_argument("--mcc", type=int, default=4)
a = ap.parse_args()
drop = set(x for x in a.drop.split(",") if x)
_cache = {}


def col(block, j):
    def f(d):
        if hasattr(d, "take"):                      # ShufDS: the base set's column through the inside-feed permutation
            return d.take(f)
        key = (block, d.name)
        if key not in _cache:
            _cache[key] = np.load(FEAT / f"{block}_{d.name}.npy")
            assert _cache[key].shape[0] == d.n_rows
        return _cache[key][:, j]
    return f


extra = {}
for b in a.blocks:
    for j, n in enumerate(json.load(open(FEAT / f"{b}_names.json"))):
        if n not in drop:
            extra[n] = col(b, j)
tr = cbx.load("xm"); new_cols = list(extra); extra_cat = []; kw = {}
if a.native_clu:
    tr = ShufDS(tr, seed=a.seed)
    for K in (256, 1024, 4096):
        extra[f"cl{K}"] = (lambda d, K=K: clu(d, K)); extra_cat.append(f"cl{K}")
    kw = dict(has_time=True, max_ctr_complexity=a.mcc, simple_ctr=CTR255, combinations_ctr=CTR255)
m = b0_model(a.seed, iterations=a.iters, **kw)
log(f"[{a.tag}] blocks {a.blocks}: {len(new_cols)} new columns (dropped {sorted(drop)}) on top of the 177; native keys {extra_cat} {kw}")
r = cbx.run_arm(a.tag, m, tr, tr.y, a.out, pool_kw=dict(cat="native", extra=extra, extra_cat=extra_cat), budget_min=a.budget_min, max_feeds=a.max_feeds or None, verbose=25,
                extra=dict(args=vars(a), new_columns=new_cols, native_keys=extra_cat, leak="hand-made statistics from non-holdout TRAIN labels only; block a: the user's strictly earlier "
                           "feeds; blocks b, c: other user folds and earlier days only; val / test / holdout labels never read"))
fi = m.get_feature_importance(type="PredictionValuesChange"); fn = m.feature_names_
imp = sorted(zip(fi, fn), reverse=True)
log("top importance:", json.dumps([(n, round(float(x), 2)) for x, n in imp[:30]]))
log("importance share of the new columns:", round(float(sum(x for x, n in imp if n in new_cols)), 2), "of 100;",
    json.dumps({b: round(float(sum(x for x, n in imp if n in set(json.load(open(FEAT / f"{b}_names.json"))) and n in new_cols)), 2) for b in a.blocks}),
    "native keys:", json.dumps({n: round(float(x), 2) for x, n in imp if n in extra_cat + ["category_id"]}))
log("new columns by importance:", json.dumps([(n, round(float(x), 2)) for x, n in imp if n in new_cols][:25]))
if not a.max_feeds:
    nats_line(a.tag, np.load(f"{a.out}/{a.tag}_val.npy"), np.load(f"{a.out}/{a.tag}_hold.npy"))
    cbx.Evaluator().compare(np.load(f"{a.out}/{a.tag}_val.npy"), np.load(f"{a.out}/{a.tag}_hold.npy"), np.load(B0V), np.load(B0H), label=f"{a.tag} - B0(ts_lr020_s0)")
log("DONE", a.tag)
