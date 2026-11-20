"""docs_rank: sharper test of within-feed label visibility through ordered target statistics (CPU, 1.2.10).
Pure-noise numeric features, 1..3 positives per feed placed at random, ONE categorical = feed-unique id, simple_ctr = Borders only
(no Counter, so the feed length cannot enter). If the learn metric computed DURING training rises above the random level while the
final model applied to the same train pool is at the random level, then the online CTR of a row saw labels of earlier rows of the
same feed (a pattern that does not exist at inference)."""
import os, json, sys
assert os.environ.get("CUDA_VISIBLE_DEVICES", "x") == ""
import numpy as np, pandas as pd
from catboost import CatBoostRanker, Pool
from catboost.utils import eval_metric
TH = 4
r = np.random.default_rng(0); nf = 1500; M = r.integers(20, 120, nf); n = int(M.sum()); off = np.r_[0, np.cumsum(M)[:-1]]; gid = np.repeat(np.arange(nf), M)
X = r.normal(size=(n, 6)).astype(np.float32); y = np.zeros(n, np.float32)
for g in range(nf):
    y[off[g] + r.choice(M[g], size=1 + r.integers(0, 3), replace=False)] = 1
user = np.repeat(r.integers(0, nf // 3, nf), M)
rand = eval_metric(y, r.normal(size=n), "NDCG:type=Base", group_id=gid, thread_count=TH)[0]
out = [dict(random_level=round(rand, 4))]; print(out[0])
for name, cat, kw in [("no categorical", None, {}), ("feed-unique id, Borders CTR only", gid, dict(simple_ctr=["Borders:Prior=0.5/1"])),
                      ("feed-unique id, default CTRs", gid, {}), ("user id (about 3 feeds), Borders CTR only", user, dict(simple_ctr=["Borders:Prior=0.5/1"])),
                      ("feed-unique id, Borders only, boosting_type=Ordered", gid, dict(simple_ctr=["Borders:Prior=0.5/1"], boosting_type="Ordered")),
                      ("feed-unique id, Borders only, has_time=True", gid, dict(simple_ctr=["Borders:Prior=0.5/1"], has_time=True))]:
    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(6)])
    if cat is not None: df["c"] = cat.astype(str)
    p = Pool(df, y, group_id=gid, cat_features=["c"] if cat is not None else None, thread_count=TH)
    m = CatBoostRanker(loss_function="YetiRank", eval_metric="NDCG:type=Base;hints=skip_train~false", iterations=200, learning_rate=0.1, depth=6,
                       thread_count=TH, random_seed=0, verbose=False, allow_writing_files=False, **kw)
    m.fit(p)
    learn = m.get_evals_result()["learn"]["NDCG:type=Base"]
    final = eval_metric(y, m.predict(p, thread_count=TH), "NDCG:type=Base", group_id=gid, thread_count=TH)[0]
    fi = dict(zip(m.feature_names_, m.get_feature_importance(type="PredictionValuesChange")))
    d = dict(arm=name, learn_ndcg_during_training_it1=round(learn[0], 4), learn_ndcg_during_training_last=round(learn[-1], 4),
             final_model_on_train_pool=round(final, 4), importance_of_cat=round(float(fi.get("c", float("nan"))), 2))
    out.append(d); print(json.dumps(d), flush=True)
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "verify_ctr_group2.json"), "w"), indent=1)
