"""T2: embedding_features on CatBoost 1.2.10 CPU, loss x calcer matrix + option spelling + what is produced.
Checks: (1) does each loss accept an embedding column, (2) is it USED (model trained on the embedding column alone must beat random),
(3) which calcer contributes under classification vs regression/ranking targets (KNN only / LDA only),
(4) how options are spelled (embedding_calcers / embedding_processing), (5) tree split descriptions (names of the estimated features),
(6) several embedding columns, (7) save/load, sum_models, calc_leaf_indexes, export formats with embedding features.
Writes logs/t2_embedding.json.
"""
import json, os, sys, time, tempfile
assert os.environ.get("CUDA_VISIBLE_DEVICES", "x") == "", "CPU only"
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from synth import make, frame, split, ndcg
from catboost import CatBoost, Pool, sum_models

OUT = os.path.join(os.path.dirname(__file__), "..", "logs", "t2_embedding.json")
D = make()
tr, ev = split(D, 90)
BASE = dict(iterations=40, learning_rate=0.2, depth=4, thread_count=4, task_type="CPU", random_seed=0, allow_writing_files=False, verbose=0)
res = {}


def pool(mask, label, embs, num=False, cats=()):
    df = frame(D, num, cats).loc[mask].reset_index(drop=True)
    for e in embs:
        df[e] = list(D[e][mask])
    return Pool(df, label=label[mask], group_id=D["gid"][mask], embedding_features=list(embs), cat_features=list(cats) or None)


def splits_of(m, p, k=3):
    out = []
    for t in range(min(k, m.tree_count_)):
        try:
            out += list(m._get_tree_splits(t, p))
        except Exception as e:  # noqa
            out.append(f"<{type(e).__name__}: {str(e)[:120]}>")
    return out


def run(name, loss, label, embs=("emb_prod",), extra=None, num=False, keep=False):
    key = f"{name}|{loss}"
    t0 = time.time()
    try:
        p = dict(BASE, loss_function=loss, **(extra or {}))
        m = CatBoost(p)
        ptr, pev = pool(tr, label, embs, num), pool(ev, label, embs, num)
        m.fit(ptr)
        nd = ndcg(np.asarray(m.predict(pev), np.float64), D["y"][ev], D["gid"][ev])
        ntr = ndcg(np.asarray(m.predict(ptr), np.float64), D["y"][tr], D["gid"][tr])
        imp = dict(zip(m.feature_names_, [round(float(v), 2) for v in m.get_feature_importance(type="PredictionValuesChange")]))
        sp = sorted(set(s.split(", bin=")[0].split(" border=")[0] for s in splits_of(m, ptr, 40)))
        res[key] = dict(status="PASS", eval_ndcg=round(nd[0], 4), se=round(nd[1], 4), n=nd[2], train_ndcg=round(ntr[0], 4), trees=m.tree_count_,
                        imp=imp, split_kinds=sp[:40], sec=round(time.time() - t0, 1))
        print(f"PASS {key:60s} eval {nd[0]:.4f} +- {nd[1]:.4f} train {ntr[0]:.4f} trees {m.tree_count_} imp {imp}\n       splits: {sp[:12]}", flush=True)
        json.dump(res, open(OUT, "w"), indent=1)
        return (m, ptr, pev) if keep else None
    except Exception as e:  # noqa
        msg = str(e).strip().replace("\n", " | ")
        res[key] = dict(status="FAIL", error=msg[:700], sec=round(time.time() - t0, 1))
        print(f"FAIL {key:60s} {type(e).__name__}: {msg[:500]}", flush=True)
        json.dump(res, open(OUT, "w"), indent=1)
        return None


rnd = ndcg(np.random.default_rng(1).normal(size=ev.sum()), D["y"][ev], D["gid"][ev])
dotn = ndcg(D["emb_prod"][ev].sum(1).astype(np.float64), D["y"][ev], D["gid"][ev])
print(f"random eval NDCG {rnd[0]:.4f} +- {rnd[1]:.4f}; plain dot product <u,e> as a score {dotn[0]:.4f} +- {dotn[1]:.4f}", flush=True)
res["_random"] = rnd; res["_dot"] = dotn
soft01 = (1 / (1 + np.exp(-D["soft"]))).astype(np.float32)
LOSSES = ["Logloss", "CrossEntropy", "RMSE", "YetiRank", "YetiRank:mode=NDCG", "YetiRankPairwise", "QuerySoftMax", "QueryRMSE",
          "LambdaMart", "StochasticRank:metric=NDCG", "PairLogit", "PairLogitPairwise"]
for loss in LOSSES:
    lab = soft01 if loss == "CrossEntropy" else D["y"]
    run("emb_default", loss, lab)
    run("emb_KNN_only", loss, lab, extra=dict(embedding_calcers=["KNN:k=5"]))
    run("emb_LDA_only", loss, lab, extra=dict(embedding_calcers=["LDA:components=1"]))
# soft labels under ranking / regression: KNN = mean teacher target of neighbours
for loss in ["RMSE", "YetiRank", "QueryRMSE", "LambdaMart"]:
    run("emb_default_softreal", loss, D["soft"])
# option spelling
run("opt_knn_k25", "Logloss", D["y"], extra=dict(embedding_calcers=["KNN:k=25"]))
run("opt_lda_c4_reg_like", "Logloss", D["y"], extra=dict(embedding_calcers=["LDA:components=1:reg=0.01:likelihood=true"]))
run("opt_lda_components_too_big", "Logloss", D["y"], extra=dict(embedding_calcers=["LDA:components=4"]))
run("opt_embedding_processing_dict", "Logloss", D["y"], embs=("emb_prod", "emb_item"), extra=dict(
    embedding_processing={"default": ["LDA:components=1"], "0": ["KNN:k=7"], "1": ["LDA:components=1", "KNN:k=3"]}))
run("opt_both_given", "Logloss", D["y"], extra=dict(embedding_calcers=["KNN:k=5"], embedding_processing={"default": ["KNN:k=5"]}))
run("opt_unknown_calcer", "Logloss", D["y"], extra=dict(embedding_calcers=["PCA:components=2"]))
run("opt_text_calcer_on_embedding", "Logloss", D["y"], extra=dict(embedding_calcers=["BoW"]))
# several embedding columns, with numeric features
run("three_emb_cols_num", "Logloss", D["y"], embs=("emb_item", "emb_user", "emb_prod"), num=True)
run("three_emb_cols_num", "YetiRank", D["y"], embs=("emb_item", "emb_user", "emb_prod"), num=True)
run("item_emb_only", "Logloss", D["y"], embs=("emb_item",))
run("user_emb_only", "Logloss", D["y"], embs=("emb_user",))
# boosting_type Ordered vs Plain with embedding features
run("emb_ordered_boost", "Logloss", D["y"], extra=dict(boosting_type="Ordered"))
run("emb_plain_boost", "Logloss", D["y"], extra=dict(boosting_type="Plain"))
run("emb_has_time", "Logloss", D["y"], extra=dict(has_time=True))

# model-output API with embedding features
k = run("api_model", "Logloss", D["y"], embs=("emb_prod",), num=True, keep=True)
if k:
    m, ptr, pev = k
    api = {}
    def chk(n, f):
        try:
            v = f(); api[n] = "PASS " + str(v)[:160]
        except Exception as e:  # noqa
            api[n] = "FAIL " + type(e).__name__ + ": " + str(e).strip().replace("\n", " | ")[:300]
        print(f"  api {n:28s} {api[n]}", flush=True)
    tmp = tempfile.mkdtemp(dir=os.path.join(os.path.dirname(__file__), "..", "cache"))
    chk("calc_leaf_indexes", lambda: m.calc_leaf_indexes(pev).shape)
    chk("staged_predict", lambda: len(list(m.staged_predict(pev, eval_period=10))))
    chk("save_cbm", lambda: (m.save_model(f"{tmp}/m.cbm"), os.path.getsize(f"{tmp}/m.cbm"))[1])
    chk("load_cbm_predict_equal", lambda: float(np.abs(CatBoost().load_model(f"{tmp}/m.cbm").predict(pev) - m.predict(pev)).max()))
    for fmt in ["json", "onnx", "cpp", "python", "coreml", "pmml"]:
        chk(f"save_{fmt}", lambda fmt=fmt: (m.save_model(f"{tmp}/m.{fmt}", format=fmt, pool=ptr), os.path.getsize(f"{tmp}/m.{fmt}"))[1])
    chk("sum_models", lambda: sum_models([m, m], weights=[0.5, 0.5]).tree_count_)
    chk("shap", lambda: m.get_feature_importance(pev, type="ShapValues").shape)
    chk("loss_function_change", lambda: m.get_feature_importance(pev, type="LossFunctionChange").round(4).tolist())
    chk("interaction", lambda: np.asarray(m.get_feature_importance(type="Interaction"))[:3].tolist())
    chk("get_all_params_embedding", lambda: {kk: vv for kk, vv in m.get_all_params().items() if "embed" in kk or "calcer" in kk})
    chk("feature_names", lambda: m.feature_names_)
    res["api"] = api
    json.dump(res, open(OUT, "w"), indent=1)
print("written", OUT)
