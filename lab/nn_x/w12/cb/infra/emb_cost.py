"""Light probe (4 threads): what CatBoost 1.2.10 does with embedding features by default and what it costs.
usage: CUDA_VISIBLE_DEVICES="" CBX_THREADS=4 python -u emb_cost.py"""
import sys, time, json
import numpy as np
sys.path.insert(0, "<repo>/nn_x/w12/cb/infra")
import cbx
from catboost import CatBoostRanker
tr = cbx.load("xm"); fit, sel = cbx.trainside_split(tr)
idx = np.where(fit)[0]
def run(n_feeds, kinds, **kw):
    m = np.zeros(tr.n_feeds, bool); m[idx[:n_feeds]] = True
    t0 = time.time(); p = cbx.make_pool(tr, feeds=m, label=tr.y, embeddings=kinds) if kinds else cbx.make_pool(tr, feeds=m, label=tr.y)
    t1 = time.time()
    mod = CatBoostRanker(loss_function="YetiRank", iterations=5, depth=6, thread_count=cbx.THREADS, allow_writing_files=False, verbose=0, **kw).fit(p)
    t2 = time.time()
    ap = mod.get_all_params()
    imp = dict(zip(mod.feature_names_, mod.get_feature_importance(type="PredictionValuesChange", thread_count=cbx.THREADS)))
    print(f"feeds {n_feeds} rows {p.num_row()} emb {kinds} {kw}: pool {t1 - t0:.1f}s fit(5 trees) {t2 - t1:.1f}s | "
          f"emb importance {[round(imp.get('emb_' + k, 0), 3) for k in (kinds or [])]}", flush=True)
    return ap
run(300, None)
ap = run(300, ["cand"])
print("embedding-related params of the fitted model:", json.dumps({k: v for k, v in ap.items() if "embed" in k.lower() or "calcer" in k.lower() or "token" in k.lower()}, default=str)[:1500], flush=True)
run(1200, ["cand"])
run(1200, None)
for spec in (["LDA"], ["KNN"]):
    try:
        run(300, ["cand"], embedding_features=None) if False else None
    except Exception as e:
        print("err", e)
print("EMB_COST DONE")
