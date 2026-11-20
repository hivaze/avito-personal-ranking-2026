"""P1: cost of CatBoost embedding calcers on REAL pair embeddings (u * e, 128-d) of the exploration pool. Timing only (5 trees), CPU, 12 threads.
usage: p1_emb_cost.py <n_feeds> <calcer: KNN|LDA|none> [dim: 128|32]
Prints wall time of Pool build, fit, predict on 200K val rows, model size, max RSS.
"""
import os, sys, time, json, resource
assert os.environ.get("CUDA_VISIBLE_DEVICES", "x") == "", "CPU only"
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from realdata import pool_feeds, pair_embedding, ROOT
from catboost import CatBoost, Pool

n_feeds, calcer = int(sys.argv[1]), sys.argv[2]
dim = int(sys.argv[3]) if len(sys.argv) > 3 else 128
t0 = time.time()
sel, rows, gid, y = pool_feeds(n_feeds, seed=0)
E = pair_embedding("train", rows)
V, vg, vy = pair_embedding("val", None, max_rows=200_000, with_meta=True)
if dim < 128:
    mu = E.mean(0); _, _, Vt = np.linalg.svd((E[:200_000] - mu).astype(np.float64), full_matrices=False)
    P = Vt[:dim].T.astype(np.float32); E = (E - mu) @ P; V = (V - mu) @ P
t1 = time.time()
x = np.asarray(E.sum(1) if dim == 128 else E[:, 0], np.float32)  # one numeric column so that a pool without calcers is valid
def mk(Em, xx, yy, gg):
    df = pd.DataFrame({"x": xx})
    if calcer == "none":
        return Pool(df, label=yy, group_id=gg)
    return Pool(df, label=yy, group_id=gg, embedding_features=["emb"], embedding_features_data={"emb": Em})
ptr = mk(E, x, y, gid)
t2 = time.time()
p = dict(loss_function="Logloss", iterations=5, depth=4, learning_rate=0.3, thread_count=12, task_type="CPU", random_seed=0, allow_writing_files=False, verbose=0)
if calcer == "KNN":
    p["embedding_calcers"] = ["KNN:k=5"]
elif calcer == "LDA":
    p["embedding_calcers"] = ["LDA:components=1"]
m = CatBoost(p); m.fit(ptr)
t3 = time.time()
xv = np.asarray(V.sum(1) if dim == 128 else V[:, 0], np.float32)
pv = m.predict(mk(V, xv, vy, vg), thread_count=12)
t4 = time.time()
f = os.path.join(os.path.dirname(__file__), "..", "cache", f"p1_{calcer}_{n_feeds}_{dim}.cbm"); m.save_model(f); sz = os.path.getsize(f); os.remove(f)
out = dict(n_feeds=n_feeds, rows=int(len(rows)), dim=dim, calcer=calcer, load_s=round(t1 - t0, 1), pool_s=round(t2 - t1, 1), fit_s=round(t3 - t2, 1),
           predict_200k_s=round(t4 - t3, 1), model_mb=round(sz / 1e6, 1), max_rss_gb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2),
           imp=dict(zip(m.feature_names_, [round(float(v), 1) for v in m.get_feature_importance(type="PredictionValuesChange")])))
print(json.dumps(out), flush=True)
with open(os.path.join(os.path.dirname(__file__), "..", "logs", "p1_emb_cost.jsonl"), "a") as fh:
    fh.write(json.dumps(out) + "\n")
