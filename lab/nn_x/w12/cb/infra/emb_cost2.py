"""Light probe 2 (4 threads): cost of the LDA and KNN embedding calcers separately (CatBoost 1.2.10, CPU).
usage: CUDA_VISIBLE_DEVICES="" CBX_THREADS=4 python -u emb_cost2.py"""
import sys, time, json
import numpy as np
sys.path.insert(0, "<repo>/nn_x/w12/cb/infra")
import cbx
from catboost import CatBoostRanker
tr = cbx.load("xm"); fit, sel = cbx.trainside_split(tr)
idx = np.where(fit)[0]
def run(n_feeds, label, **kw):
    m = np.zeros(tr.n_feeds, bool); m[idx[:n_feeds]] = True
    try:
        p = cbx.make_pool(tr, feeds=m, label=tr.y, embeddings=["cand"])
        t1 = time.time()
        mod = CatBoostRanker(loss_function="YetiRank", iterations=20, depth=6, thread_count=cbx.THREADS, allow_writing_files=False, verbose=0, **kw).fit(p)
        t2 = time.time()
        imp = dict(zip(mod.feature_names_, mod.get_feature_importance(type="PredictionValuesChange", thread_count=cbx.THREADS)))
        print(f"{label}: feeds {n_feeds} rows {p.num_row()} fit(20 trees) {t2 - t1:.1f}s | emb_cand importance {imp.get('emb_cand', 0):.3f} | "
              f"{json.dumps(mod.get_all_params().get('embedding_processing'), default=str)[:200]}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"{label}: FAILED {type(e).__name__}: {str(e)[:300]}", flush=True)
for lab, kw in (("LDA only (dict form)", dict(embedding_processing={"embedding_processing": {"default": ["LDA"]}})),
                ("KNN only (dict form)", dict(embedding_processing={"embedding_processing": {"default": ["KNN"]}})),
                ("LDA only (calcers arg)", dict(embedding_calcers=["LDA"])),
                ("default", {})):
    run(300, lab, **kw)
run(1200, "LDA only (dict form)", embedding_processing={"embedding_processing": {"default": ["LDA"]}})
print("EMB_COST2 DONE")
