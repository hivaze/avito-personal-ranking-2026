"""Truncated copy of a famS model at a tree cap T0 (addendum 2 (c) of PREDECLARED_famS.txt): the prediction at the best train-side selection
iteration among the FIRST T0 trees of <src_tag> = what a T0-tree run of that seed gives (boosting is sequential, the first T0 trees do not depend on
the later ones). No val / holdout label is used for the choice; the evaluation at the end is the standard cbx.Evaluator read.
usage: CUDA_VISIBLE_DEVICES="" python -u fs_trunc.py <src_tag> <dst_tag> <T0> [block none|pca|raw|cand]
writes preds/<dst_tag>_{val,hold,test}.npy, <dst_tag>.eval.json, <dst_tag>.curve.json (first T0 rows of the source curve)."""
import sys, json
import numpy as np
sys.path.insert(0, "<repo>/nn_x/w12/cb/famS")
import fs_common as fs
from fs_common import cbx, log
from catboost import CatBoost

src, dst, T0 = sys.argv[1], sys.argv[2], int(sys.argv[3])
blk = sys.argv[4] if len(sys.argv) > 4 else "none"
assert src != dst
c = json.load(open(fs.PREDS / f"{src}.curve.json"))
assert len(c) >= T0, (len(c), T0)
c0 = c[:T0]
v = np.array([m["validation:NDCG:type=Base"] for _, _, m in c0])
best = int(np.argmax(v)); k = best + 1
m = CatBoost(); m.load_model(str(fs.PREDS / f"{src}.cbm"))
assert m.tree_count_ >= T0
kw = dict(cat="native")
if blk != "none":
    kw["extra"] = fs.extra_cols(blk)
pv, ph, pt = (np.asarray(m.predict(cbx.make_pool(cbx.load(n), **kw), ntree_end=k, thread_count=cbx.THREADS), np.float32) for n in ("val", "hold", "test"))
(fs.PREDS / f"{dst}.curve.json").write_text(json.dumps(c0))
info = dict(best_iter=best, trees_built=T0, fit_seconds=float(c0[-1][1]), truncated_from=src, source_trees=len(c), sel_ndcg=float(v[best]), selection="trainside", block=blk)
log(f"[{dst}] = {src} truncated at {T0} trees: best selection iteration {best} (sel NDCG {v[best]:.5f}), fit time to tree {T0}: {c0[-1][1] / 60:.1f} min")
cbx.Evaluator().evaluate(dst, pv, ph, pt, out_dir=fs.PREDS, extra=info)
log(f"[{dst}] DONE")
