"""T1: loss function x feature type matrix on CatBoost 1.2.10 CPU.
For every loss: (a) numeric only hard labels, (b) soft real-valued labels, (c) embedding feature only (is it USED?),
(d) text feature only, (e) categorical high-cardinality only, (f) baseline in Pool, (g) init_model continuation.
Writes logs/t1_loss_matrix.json ; prints one line per check: PASS / FAIL + exact error text.
"""
import json, os, sys, time, traceback
assert os.environ.get("CUDA_VISIBLE_DEVICES", "x") == "", "CPU only"
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from synth import make, frame, split, ndcg
from catboost import CatBoost, Pool

OUT = os.path.join(os.path.dirname(__file__), "..", "logs", "t1_loss_matrix.json")
D = make()
tr, ev = split(D, 90)
LOSSES = ["Logloss", "CrossEntropy", "RMSE", "YetiRank", "YetiRank:mode=NDCG", "YetiRankPairwise", "QuerySoftMax", "QueryRMSE",
          "LambdaMart", "StochasticRank:metric=NDCG", "PairLogit", "PairLogitPairwise", "QueryCrossEntropy"]
BASE = dict(iterations=40, learning_rate=0.2, depth=4, thread_count=4, task_type="CPU", random_seed=0, allow_writing_files=False, verbose=0)
res = {}


def pool(mask, label, num=True, cats=(), texts=(), embs=(), baseline=None):
    df = frame(D, num, cats, texts).loc[mask].reset_index(drop=True)
    kw = {}
    if embs:
        kw = dict(embedding_features=[f"E_{e}" for e in embs], embedding_features_data={f"E_{e}": D[e][mask] for e in embs})
    return Pool(df, label=label[mask], group_id=D["gid"][mask], cat_features=list(cats) or None, text_features=list(texts) or None,
                baseline=None if baseline is None else baseline[mask], **kw)


def run(name, loss, label, extra=None, init=False, **pk):
    key = f"{name}|{loss}"
    t0 = time.time()
    try:
        p = dict(BASE, loss_function=loss, **(extra or {}))
        m = CatBoost(p)
        ptr, pev = pool(tr, label, **pk), pool(ev, label, **pk)
        m.fit(ptr)
        if init:
            m2 = CatBoost(p); m2.fit(ptr, init_model=m); m = m2
        pr = m.predict(pev)
        nd = ndcg(np.asarray(pr, np.float64), D["y"][ev], D["gid"][ev])
        ntr = ndcg(np.asarray(m.predict(ptr), np.float64), D["y"][tr], D["gid"][tr])
        imp = dict(zip(m.feature_names_, [round(float(v), 2) for v in m.get_feature_importance(type="PredictionValuesChange")]))
        res[key] = dict(status="PASS", eval_ndcg=round(nd[0], 4), se=round(nd[1], 4), n=nd[2], train_ndcg=round(ntr[0], 4), trees=m.tree_count_,
                        imp=imp, sec=round(time.time() - t0, 1))
        print(f"PASS {key:55s} eval NDCG {nd[0]:.4f} +- {nd[1]:.4f} (n={nd[2]}) train {ntr[0]:.4f} trees {m.tree_count_} imp {imp}", flush=True)
    except Exception as e:  # noqa
        msg = str(e).strip().replace("\n", " | ")
        res[key] = dict(status="FAIL", error=msg[:600], sec=round(time.time() - t0, 1))
        print(f"FAIL {key:55s} {type(e).__name__}: {msg[:400]}", flush=True)
    json.dump(res, open(OUT, "w"), indent=1)


rnd = ndcg(np.random.default_rng(1).normal(size=ev.sum()), D["y"][ev], D["gid"][ev])
print(f"random-score eval NDCG {rnd[0]:.4f} +- {rnd[1]:.4f}; oracle soft-score NDCG {ndcg(D['soft'][ev].astype(float), D['y'][ev], D['gid'][ev])[0]:.4f}", flush=True)
res["_random"] = rnd; res["_oracle_soft"] = ndcg(D['soft'][ev].astype(float), D['y'][ev], D['gid'][ev])
soft01 = (1 / (1 + np.exp(-D["soft"]))).astype(np.float32)
for loss in LOSSES:
    run("a_numeric_hard", loss, D["y"])
    run("b_numeric_soft_real", loss, D["soft"])                 # real-valued, negative values present
    run("b2_numeric_soft_01", loss, soft01)                      # real-valued in (0,1)
    run("c_emb_prod_only", loss, D["y"], num=False, embs=("emb_prod",))
    run("c2_emb_prod_only_soft", loss, D["soft"], num=False, embs=("emb_prod",))
    run("d_text_cross_only", loss, D["y"], num=False, texts=("text_cross",))
    run("e_cat_item_only", loss, D["y"], num=False, cats=("item_id", "cat_id"))
    run("f_baseline", loss, D["y"], baseline=(0.5 * D["x"][:, 0]).astype(np.float64))
    run("g_init_model", loss, D["y"], init=True)
print("written", OUT)
