"""Cross-fitted residual LightGBM on top of the model score (init_score = c * score).

usage: python resid_cv.py <objective> <num_leaves> <lr> <min_data> <feature_set> [max_trees]
Evaluates the out-of-fold NDCG change at several tree counts; appends a line per (variant, trees) to cv_log.txt.
"""
from __future__ import annotations

import sys
import time

import lightgbm as lgb
import numpy as np
import polars as pl

import prep
from prep import kit

obj, nl, lr, mdl, fset = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
max_trees = int(sys.argv[6]) if len(sys.argv) > 6 else 400
l2 = float(sys.argv[7]) if len(sys.argv) > 7 else 10.0
CHECK = [t for t in (10, 25, 50, 100, 200, 400, 800) if t <= max_trees]

df = prep.derive(kit.load())
names = prep.feature_names(df)
if fset == "nobranch":      # drop s_ft / br_diff (NaN on the holdout)
    names = [c for c in names if c not in ("s_ft", "br_diff")]
elif fset == "raw":
    names = [c for c in prep.RAW if c not in ("s_ft",)]
elif fset == "nofm":
    names = [c for c in names if not c.startswith("fm_") and c not in ("s_ft", "br_diff")]
X = prep.matrix(df, names)
y = df["label"].to_numpy().astype(np.float64)
s = df["score"].to_numpy().astype(np.float64)
off, sz = kit.feed_layout(df)
fold = kit.folds(df)
ffold = fold[off]
oof = {t: np.zeros(len(y)) for t in CHECK}
t0 = time.time()
for k in range(5):
    trf = ffold != k
    tr = fold != k
    toff = np.r_[0, np.cumsum(sz[trf])[:-1]]
    c = prep.softmax_scale(s[tr], y[tr], toff, sz[trf])
    if obj == "binary":
        # logistic calibration of c*score + a on the training folds
        from sklearn.linear_model import LogisticRegression
        lrm = LogisticRegression(C=1e6).fit(s[tr, None], y[tr])
        init = lrm.intercept_[0] + lrm.coef_[0, 0] * s
    else:
        init = c * s
    params = dict(objective=obj, num_leaves=nl, learning_rate=lr, min_data_in_leaf=mdl, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=l2, num_threads=12, verbose=-1, seed=k,
                  max_depth=-1)
    if obj == "softmax":
        fid_tr = np.repeat(np.arange(trf.sum()), sz[trf])
        npos_tr = np.add.reduceat(y[tr], toff)
        def fobj(pred, data, fid=fid_tr, o=toff, npos=npos_tr, yy=y[tr]):
            m = np.maximum.reduceat(pred, o)[fid]
            e = np.exp(pred - m)
            pz = e / np.add.reduceat(e, o)[fid]
            w = 1.0 / np.maximum(npos, 1)[fid]
            return (pz * npos[fid] - yy) * w, np.maximum(pz * (1 - pz) * npos[fid] * w, 1e-6)
        params.update(objective=fobj, min_sum_hessian_in_leaf=float(sys.argv[8]) if len(sys.argv) > 8 else 1.0)
    if obj in ("lambdarank", "rank_xendcg"):
        params.update(lambdarank_truncation_level=300, label_gain=[0, 1], eval_at=[300])
    ds = lgb.Dataset(X[tr], y[tr], init_score=init[tr], group=sz[trf] if obj != "binary" else None,
                     free_raw_data=False)
    bst = lgb.train(params, ds, num_boost_round=max(CHECK))
    te = fold == k
    for t in CHECK:
        oof[t][te] = init[te] + bst.predict(X[te], num_iteration=t, raw_score=True)
    print(f"fold {k} c={c:.2f} {time.time()-t0:.0f}s", flush=True)

with open("cv_log.txt", "a") as fh:
    for t in CHECK:
        r = kit.evaluate(df, oof[t])
        line = (f"{obj} nl={nl} lr={lr} mdl={mdl} l2={l2} fset={fset} trees={t}: delta={r['delta']:+.5f} "
                f"se={r['se']:.5f} z={r['z']:.2f}")
        print(line)
        fh.write(line + "\n")
np.save(f"oof_{obj}_{nl}_{lr}_{mdl}_{fset}.npy", oof[max(CHECK)])
