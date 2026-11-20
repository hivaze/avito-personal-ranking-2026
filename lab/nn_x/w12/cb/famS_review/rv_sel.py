"""famS_review: is the curve used for the tree count really the click NDCG of the xm train-side SELECTION feeds (not val / holdout)?
Recompute it with own NDCG for saved models at their selected tree count; also the val / holdout argmax over staged counts would be a different thing (not computed)."""
import os, sys, json
os.environ["CUDA_VISIBLE_DEVICES"] = ""; os.environ["CBX_THREADS"] = "4"; os.environ["CBX_CACHE_EXTRA"] = "<repo>/nn_x/w12/cb/famS_review/cache"
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS"): os.environ[k] = "4"
import numpy as np
sys.path.insert(0, "<repo>/nn_x/w12/cb/infra"); sys.path.insert(0, "<repo>/nn_x/w12/cb/famS_review")
import cbx, rv_own as o
from catboost import CatBoost
P = cbx.ROOT / "nn_x/w12/cb/famS/preds"
tr = cbx.load("xm"); fit, sel = cbx.trainside_split(tr)
ft, Mt, hold, xm = o.train_hold_index()
assert (tr.fidx == xm).all()
uid = ft["user_id"].to_numpy()
us, uf, uh = set(uid[xm[sel]].tolist()), set(uid[xm[fit]].tolist()), set(uid[hold].tolist())
print(f"selection feeds {int(sel.sum())}; users: selection {len(us)}, fit {len(uf)}, overlap selection/fit {len(us & uf)}; holdout users {len(uh)}, holdout users also in the fit set {len(uh & uf)} ({len(uh & uf) / len(uh):.3f})")
rm = np.repeat(sel, tr.M); Ms = tr.M[sel]; w = dict(M=Ms, off=np.r_[0, np.cumsum(Ms)[:-1]].astype(np.int64), y=np.asarray(tr.y, np.float64)[rm])
pool = cbx.make_pool(tr, feeds=sel, cat="native")
for tag in ("s1f_rmse_d8_lr030_s0", "s1f_rmse_d8_lr030_s2", "s4b_qsm_soft_lr030_s0", "s1_rmse_d8_lr010_s0"):
    c = json.load(open(P / f"{tag}.curve.json")); v = np.array([m["validation:NDCG:type=Base"] for _, _, m in c]); k = int(np.argmax(v)) + 1
    m = CatBoost(); m.load_model(str(P / f"{tag}.cbm"))
    p = np.asarray(m.predict(pool, ntree_end=k, thread_count=4), np.float32)
    print(f"[{tag}] curve value at tree {k}: {v[k - 1]:.5f} | own click NDCG on the 13,527 xm selection feeds: {o.ndcg(p, w).mean():.5f}", flush=True)
