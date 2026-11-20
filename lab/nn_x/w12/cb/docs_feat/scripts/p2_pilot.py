"""P2: SMALL real-data pilot (not an arm of the track): do CatBoost embedding calcers on the pair embedding u*e add anything to the cb26 177-feature set?
Train: N random feeds of the exploration pool (holdout excluded), fixed number of trees (no early stopping, val labels are used for scoring only).
usage: p2_pilot.py <arm> <seed> [n_feeds=4000] [iters=300]
arms: base_Y | emb_Y | knn50_Y | base_L | emb_L | lda_L | knn50_L | handlda_Y | base_S | knn50_S | emb_S   (_S = RMSE on the centred out-of-fold teacher logit)
  *_Y = YetiRank, *_L = Logloss. emb = default calcers (LDA + KNN:k=5). handlda_Y = LDA direction fitted by hand on the train rows, given as ONE numeric column.
Writes cache/p2_<arm>_s<seed>_n<n>_val.npy (per-row score) and appends a line to logs/p2_pilot.jsonl.
"""
import os, sys, time, json
assert os.environ.get("CUDA_VISIBLE_DEVICES", "x") == "", "CPU only"
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, "<repo>/nn_x/cb26"); sys.path.insert(0, "<repo>/src")
from realdata import pool_feeds, pair_embedding, ROOT
from cbdata import load_tab, meta
from cb_member import member_features
from metrics import ndcg_per_feed
from catboost import CatBoost, Pool

arm, seed = sys.argv[1], int(sys.argv[2])
n_feeds = int(sys.argv[3]) if len(sys.argv) > 3 else 4000
iters = int(sys.argv[4]) if len(sys.argv) > 4 else 300
HERE = os.path.dirname(__file__)
t0 = time.time(); log = lambda *a: print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)
names = member_features(); log(len(names), "features")
sel, rows, gid, y = pool_feeds(n_feeds, seed=1)          # the SAME feeds for every arm and seed
Xtr = load_tab("train", names, rows); fv, Mv, ov, yv = meta("val"); Xv = load_tab("val", names)
gv = np.repeat(np.arange(len(Mv)), Mv)
log("train", Xtr.shape, "val", Xv.shape, "pos rate", float(y.mean()))
use_emb = arm.split("_")[0] in ("emb", "knn50", "lda")
loss = "YetiRank" if arm.endswith("_Y") else "RMSE" if arm.endswith("_S") else "Logloss"
ylab = y
if arm.endswith("_S"):  # soft target = out-of-fold teacher logit (sum of 3 fold-excluded models), centred per feed, /3
    t = np.asarray(np.load(ROOT / "nn_x/w12/kd/out/teacher_oof.npy", mmap_mode="r")[rows], np.float64)
    cnt = np.bincount(np.unique(gid, return_inverse=True)[1]); o = np.r_[0, np.cumsum(cnt)[:-1]]
    ylab = ((t - np.repeat(np.add.reduceat(t, o) / cnt, cnt)) / 3.0).astype(np.float32)
    log("soft target sd", float(ylab.std()), "corr with click", float(np.corrcoef(ylab, y)[0, 1]))
Etr = Ev = None
if use_emb or arm.startswith("handlda"):
    Etr = pair_embedding("train", rows); Ev = pair_embedding("val"); log("pair embeddings", Etr.shape, Ev.shape)


def df_of(X):
    d = pd.DataFrame(X, columns=names, copy=False)
    d["category_id"] = np.nan_to_num(d["category_id"].to_numpy(), nan=-1).astype(np.int32)
    return d


dtr, dv = df_of(Xtr), df_of(Xv)
if arm.startswith("handlda"):
    # Fisher direction w = Sw^-1 (mu1 - mu0) on the train rows only (labels of the pool feeds; no holdout / val label)
    E64 = Etr.astype(np.float64); m1, m0 = E64[y > 0].mean(0), E64[y == 0].mean(0)
    Sw = np.cov(E64[y > 0].T) * (y > 0).sum() + np.cov(E64[y == 0].T) * (y == 0).sum(); Sw = Sw / len(y) + 1e-6 * np.eye(128)
    w = np.linalg.solve(Sw, m1 - m0)
    dtr["hand_lda"] = (Etr @ w).astype(np.float32); dv["hand_lda"] = (Ev @ w).astype(np.float32)
    log("hand LDA: corr(w, ones)", float(np.corrcoef(w, np.ones(128))[0, 1]))
kw_tr, kw_v = {}, {}
if use_emb:
    kw_tr = dict(embedding_features=["pair"], embedding_features_data={"pair": Etr}); kw_v = dict(embedding_features=["pair"], embedding_features_data={"pair": Ev})
ptr = Pool(dtr, label=ylab, group_id=gid, cat_features=["category_id"], thread_count=12, **kw_tr)
pv = Pool(dv, group_id=gv, cat_features=["category_id"], thread_count=12, **kw_v)
p = dict(loss_function=loss, iterations=iters, learning_rate=0.1, depth=6, l2_leaf_reg=5, thread_count=12, task_type="CPU", random_seed=seed,
         allow_writing_files=False, verbose=50)
if arm.startswith("knn50"):
    p["embedding_calcers"] = ["KNN:k=50"]
elif arm.startswith("lda"):
    p["embedding_calcers"] = ["LDA:components=1"]
log("fit", arm, loss, p.get("embedding_calcers", "default calcers" if use_emb else "no embedding"))
m = CatBoost(p); m.fit(ptr); t_fit = time.time() - t0
pr = np.asarray(m.predict(pv, thread_count=12), np.float32)
nd = ndcg_per_feed(pr, yv, ov, Mv)
tag = f"p2_{arm}_s{seed}_n{n_feeds}"
np.save(os.path.join(HERE, "..", "cache", f"{tag}_val.npy"), pr)
imp = sorted(zip(m.get_feature_importance(type="PredictionValuesChange"), m.feature_names_), reverse=True)
top = [(n, round(float(v), 2)) for v, n in imp[:8]]; extra = [(n, round(float(v), 2)) for v, n in imp if n in ("pair", "hand_lda")]
out = dict(arm=arm, seed=seed, n_feeds=int(len(sel)), rows=int(len(rows)), iters=iters, loss=loss, val_ndcg=float(nd.mean()), val_se=float(nd.std() / np.sqrt(len(nd))), n_val=int(len(nd)),
           fit_min=round(t_fit / 60, 1), total_min=round((time.time() - t0) / 60, 1), top=top, emb_importance=extra)
log(json.dumps(out))
with open(os.path.join(HERE, "..", "logs", "p2_pilot.jsonl"), "a") as fh:
    fh.write(json.dumps(out) + "\n")
