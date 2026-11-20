"""B0 extra seeds: the reference recipe (infra ts_lr020_s0: YetiRank d8 lr 0.2 l2 5, 177 features, category_id native, train-side selection),
written into famB/preds (the infra directory is read-only).   usage: python -u b0_arm.py --seed 1 [--max_feeds 3000]"""
import argparse, sys
sys.path.insert(0, "<repo>/nn_x/w12/cb/famB")
from fb_common import *
ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, required=True); ap.add_argument("--max_feeds", type=int, default=0)
ap.add_argument("--budget_min", type=float, default=None); ap.add_argument("--tag", default=None)
a = ap.parse_args(); tag = a.tag or f"b0_s{a.seed}"
tr = cbx.load("xm")
cbx.run_arm(tag, b0_model(a.seed), tr, tr.y, PREDS, pool_kw=dict(cat="native"), budget_min=a.budget_min, max_feeds=a.max_feeds or None, verbose=25,
            extra=dict(args=vars(a), leak="B0: the 177 features of record, hard labels, no val label in selection"))
if not a.max_feeds:
    nats_line(tag, np.load(PREDS / f"{tag}_val.npy"), np.load(PREDS / f"{tag}_hold.npy"))
log("DONE", tag)
