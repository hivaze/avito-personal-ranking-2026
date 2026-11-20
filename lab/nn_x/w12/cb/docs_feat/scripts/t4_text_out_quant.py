"""T4: text features (as a carrier for id SETS), T5: getting things out of a model, T6: numeric quantisation. CatBoost 1.2.10 CPU, synthetic.
usage: t4_text_out_quant.py text|out|quant   -> logs/t4_<section>.json
"""
import json, os, sys, time, tempfile
assert os.environ.get("CUDA_VISIBLE_DEVICES", "x") == "", "CPU only"
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from synth import make, frame, split, ndcg
from catboost import CatBoost, CatBoostRanker, Pool, sum_models, EFstrType

SEC = sys.argv[1]
OUT = os.path.join(os.path.dirname(__file__), "..", "logs", f"t4_{SEC}.json")
D = make(n_groups=300, gsize=50, n_items=1500, seed=3)
tr, ev = split(D, 200)
BASE = dict(iterations=60, learning_rate=0.2, depth=4, thread_count=6, task_type="CPU", random_seed=0, allow_writing_files=False, verbose=0)
res = {}


def pool(mask, label, num=True, cats=(), texts=(), with_group=True):
    df = frame(D, num, cats, texts).loc[mask].reset_index(drop=True)
    return Pool(df, label=label[mask], group_id=D["gid"][mask] if with_group else None, cat_features=list(cats) or None, text_features=list(texts) or None)


def run(name, loss, extra=None, label=None, keep=False, **pk):
    key = f"{name}|{loss}"
    label = D["y"] if label is None else label
    t0 = time.time()
    try:
        m = CatBoost(dict(BASE, loss_function=loss, **(extra or {})))
        ptr, pev = pool(tr, label, **pk), pool(ev, label, **pk)
        m.fit(ptr)
        nd = ndcg(np.asarray(m.predict(pev), np.float64), D["y"][ev], D["gid"][ev])
        imp = dict(zip(m.feature_names_, [round(float(v), 1) for v in m.get_feature_importance(type="PredictionValuesChange")]))
        sp = []
        for t in range(m.tree_count_):
            try:
                sp += list(m._get_tree_splits(t, ptr))
            except Exception as e:  # noqa
                sp.append(f"<{str(e)[:80]}>"); break
        kinds = sorted(set(s.split(", bin=")[0].split(" border=")[0].split(", value")[0] for s in sp))
        calcers = sorted(set(k.split("calcer_id=")[1].split(" ")[0] for k in kinds if "calcer_id=" in k))
        res[key] = dict(status="PASS", eval_ndcg=round(nd[0], 4), se=round(nd[1], 4), n=nd[2], imp=imp, n_distinct_split_features=len(kinds), n_calcers_used=len(calcers),
                        sec=round(time.time() - t0, 1))
        print(f"PASS {key:60s} eval {nd[0]:.4f} +- {nd[1]:.4f} imp {imp} split-features {len(kinds)} calcers-used {len(calcers)} sec {time.time() - t0:.1f}", flush=True)
        json.dump(res, open(OUT, "w"), indent=1)
        return (m, ptr, pev) if keep else None
    except Exception as e:  # noqa
        msg = str(e).strip().replace("\n", " | ")
        res[key] = dict(status="FAIL", error=msg[:600])
        print(f"FAIL {key:60s} {type(e).__name__}: {msg[:420]}", flush=True)
        json.dump(res, open(OUT, "w"), indent=1)


def chk(store, n, f):
    try:
        v = f(); store[n] = "PASS " + str(v)[:220]
    except Exception as e:  # noqa
        store[n] = "FAIL " + type(e).__name__ + ": " + str(e).strip().replace("\n", " | ")[:330]
    print(f"  {n:44s} {store[n]}", flush=True)
    json.dump(res, open(OUT, "w"), indent=1)


if SEC == "text":
    DIC1 = [{"dictionary_id": "W", "occurrence_lower_bound": "1", "max_dictionary_size": "50000", "gram_order": "1"}]
    TOK = [{"tokenizer_id": "Space", "separator_type": "ByDelimiter", "delimiter": " "}]
    for loss in ["Logloss", "YetiRank", "RMSE", "QueryRMSE", "LambdaMart"]:
        for tname in ["text_cross", "text_hist"]:
            run(f"{tname}_default", loss, num=False, texts=(tname,))
        run("text_cross_BoW", loss, dict(tokenizers=TOK, dictionaries=DIC1, feature_calcers=["BoW:top_tokens_count=2000"]), num=False, texts=("text_cross",))
        run("text_cross_NaiveBayes", loss, dict(tokenizers=TOK, dictionaries=DIC1, feature_calcers=["NaiveBayes"]), num=False, texts=("text_cross",))
        run("text_cross_BM25", loss, dict(tokenizers=TOK, dictionaries=DIC1, feature_calcers=["BM25"]), num=False, texts=("text_cross",))
        run("text_cross_all3", loss, dict(tokenizers=TOK, dictionaries=DIC1, feature_calcers=["BoW:top_tokens_count=2000", "NaiveBayes", "BM25"]), num=False, texts=("text_cross",))
    run("text_processing_dict", "Logloss", dict(text_processing={
        "tokenizers": TOK, "dictionaries": DIC1 + [{"dictionary_id": "Bi", "occurrence_lower_bound": "1", "gram_order": "2"}],
        "feature_processing": {"default": [{"dictionaries_names": ["W", "Bi"], "feature_calcers": ["BoW"], "tokenizers_names": ["Space"]},
                                           {"dictionaries_names": ["W"], "feature_calcers": ["NaiveBayes", "BM25"], "tokenizers_names": ["Space"]}]}}),
        num=False, texts=("text_cross",))
    run("num_only_reference", "Logloss", num=True)
    run("num_plus_text_cross", "Logloss", num=True, texts=("text_cross",))
    run("num_plus_text_cross", "YetiRank", num=True, texts=("text_cross",))
    # leak test: group-constant text + pure-noise labels
    r = np.random.default_rng(11); G, S = 800, 10
    gid = np.repeat(np.arange(G), S); y = np.concatenate([r.permutation(np.r_[np.ones(5), np.zeros(5)]) for _ in range(G)]).astype(np.float32)
    txt = np.array([f"u{g} w{g % 7}" for g in gid], dtype=object); trm = gid < 600
    for nm, ex in [("default", {}), ("NaiveBayes", dict(tokenizers=TOK, dictionaries=DIC1, feature_calcers=["NaiveBayes"])),
                   ("BM25", dict(tokenizers=TOK, dictionaries=DIC1, feature_calcers=["BM25"])), ("BoW", dict(tokenizers=TOK, dictionaries=DIC1, feature_calcers=["BoW"]))]:
        try:
            ptr = Pool(pd.DataFrame({"t": txt[trm]}), label=y[trm], group_id=gid[trm], text_features=["t"])
            pev = Pool(pd.DataFrame({"t": txt[~trm]}), label=y[~trm], group_id=gid[~trm], text_features=["t"])
            m = CatBoost(dict(iterations=200, learning_rate=0.2, depth=4, thread_count=6, task_type="CPU", random_seed=0, allow_writing_files=False, verbose=0, loss_function="Logloss", **ex))
            m.fit(ptr, eval_set=pev); er = m.get_evals_result()
            res[f"leak_text_{nm}"] = dict(learn_first=round(er["learn"]["Logloss"][0], 4), learn_last=round(er["learn"]["Logloss"][-1], 4), eval_last=round(er["validation"]["Logloss"][-1], 4))
        except Exception as e:  # noqa
            res[f"leak_text_{nm}"] = "FAIL " + str(e).strip().replace("\n", " | ")[:300]
        print(f"LEAK-TEST group-constant text, noise labels, {nm:10s}: {res[f'leak_text_{nm}']}  (0.6931 = no information)", flush=True)
    k = run("api_text_model", "Logloss", num=True, texts=("text_cross",), keep=True)
    if k:
        m, ptr, pev = k; api = res.setdefault("api_text", {}); tmp = tempfile.mkdtemp(dir=os.path.join(os.path.dirname(__file__), "..", "cache"))
        chk(api, "calc_leaf_indexes(text model)", lambda: m.calc_leaf_indexes(pev).shape)
        chk(api, "sum_models(text model)", lambda: sum_models([m, m]).tree_count_)
        chk(api, "save_json(text model)", lambda: m.save_model(f"{tmp}/t.json", format="json", pool=ptr))
        chk(api, "save_cbm(text model) bytes", lambda: (m.save_model(f"{tmp}/t.cbm"), os.path.getsize(f"{tmp}/t.cbm"))[1])
        chk(api, "shap(text model)", lambda: m.get_feature_importance(pev, type="ShapValues").shape)

if SEC == "out":
    api = res.setdefault("api", {}); tmp = tempfile.mkdtemp(dir=os.path.join(os.path.dirname(__file__), "..", "cache"))
    CATS = ("cat_id", "item_id")
    m = CatBoostRanker(**dict(BASE, loss_function="YetiRank")); ptr, pev = pool(tr, D["y"], cats=CATS), pool(ev, D["y"], cats=CATS); m.fit(ptr)
    m2 = CatBoostRanker(**dict(BASE, loss_function="YetiRank", random_seed=1)); m2.fit(ptr)
    mn = CatBoostRanker(**dict(BASE, loss_function="YetiRank")); pntr, pnev = pool(tr, D["y"]), pool(ev, D["y"]); mn.fit(pntr)
    mn2 = CatBoostRanker(**dict(BASE, loss_function="QuerySoftMax")); mn2.fit(pntr)
    base_nd = ndcg(m.predict(pev).astype(float), D["y"][ev], D["gid"][ev]); print("model with cat features: eval NDCG", base_nd, flush=True)
    chk(api, "calc_leaf_indexes (ctr model) shape/max", lambda: (m.calc_leaf_indexes(pev).shape, int(m.calc_leaf_indexes(pev).max())))
    chk(api, "calc_leaf_indexes ntree range", lambda: m.calc_leaf_indexes(pev, ntree_start=10, ntree_end=20).shape)
    chk(api, "iterate_leaf_indexes first", lambda: next(iter(m.iterate_leaf_indexes(pev)))[:5])
    chk(api, "staged_predict n stages (eval_period=20)", lambda: len(list(m.staged_predict(pev, eval_period=20))))
    chk(api, "predict ntree_end=30 differs", lambda: float(np.abs(m.predict(pev, ntree_end=30) - m.predict(pev)).mean()))
    for pol in ["IntersectingCountersAverage", "LeaveMostDiversifiedTable", "FailIfCtrIntersects", "KeepAllTables"]:
        chk(api, f"sum_models ctr models, {pol}", lambda pol=pol: (lambda s: (s.tree_count_, float(np.abs(s.predict(pev) - 0.5 * (m.predict(pev) + m2.predict(pev))).max())))(sum_models([m, m2], weights=[0.5, 0.5], ctr_merge_policy=pol)))
    chk(api, "sum_models numeric YetiRank+QuerySoftMax max|diff|", lambda: (lambda s: float(np.abs(s.predict(pnev) - (0.7 * mn.predict(pnev) + 0.3 * mn2.predict(pnev))).max()))(sum_models([mn, mn2], weights=[0.7, 0.3])))
    for fmt in ["cbm", "json", "cpp", "python", "onnx", "coreml", "pmml"]:
        chk(api, f"save {fmt} (ctr model) bytes", lambda fmt=fmt: (m.save_model(f"{tmp}/c.{fmt}", format=fmt, pool=ptr), os.path.getsize(f"{tmp}/c.{fmt}"))[1])
        chk(api, f"save {fmt} (numeric model) bytes", lambda fmt=fmt: (mn.save_model(f"{tmp}/n.{fmt}", format=fmt, pool=pntr), os.path.getsize(f"{tmp}/n.{fmt}"))[1])
    for t in ["PredictionValuesChange", "LossFunctionChange", "ShapValues", "ShapInteractionValues", "Interaction", "PredictionDiff", "SageValues", "FeatureImportance"]:
        def f(t=t):
            if t == "PredictionDiff":
                v = mn.get_feature_importance(pnev.slice([0, 1]) if hasattr(pnev, "slice") else pnev, type=t)
            elif t in ("ShapInteractionValues",):
                v = mn.get_feature_importance(pool(ev & (D["gid"] < 203), D["y"]), type=t)
            elif t == "SageValues":
                v = mn.get_feature_importance(pool(ev & (D["gid"] < 210), D["y"]), type=t)
            else:
                v = m.get_feature_importance(pev, type=t)
            v = np.asarray(v)
            return (v.shape, np.round(v.reshape(-1)[:7], 3).tolist())
        chk(api, f"fstr {t} (YetiRank)", f)
    chk(api, "fstr default type for YetiRank model (no type given, with pool)", lambda: np.round(m.get_feature_importance(pev), 4).tolist())
    chk(api, "fstr default type without pool (ranking)", lambda: np.round(m.get_feature_importance(), 4).tolist())
    def sel(algo, loss="YetiRank"):
        ms = CatBoostRanker(**dict(BASE, loss_function=loss, iterations=40))
        s = ms.select_features(pntr, eval_set=pnev, features_for_select=[0, 1, 2, 3], num_features_to_select=2, algorithm=algo, steps=2, train_final_model=False, logging_level="Silent")
        return dict(selected=s["selected_features_names"], eliminated=s["eliminated_features_names"], keys=sorted(s.keys()))
    for algo in ["RecursiveByShapValues", "RecursiveByLossFunctionChange", "RecursiveByPredictionValuesChange"]:
        chk(api, f"select_features {algo} (YetiRank)", lambda algo=algo: sel(algo))
    def sel_cat():
        ms = CatBoostRanker(**dict(BASE, loss_function="YetiRank", iterations=40))
        s = ms.select_features(ptr, eval_set=pev, features_for_select=list(range(6)), num_features_to_select=3, algorithm="RecursiveByLossFunctionChange", steps=2, train_final_model=False, logging_level="Silent")
        return dict(selected=s["selected_features_names"], eliminated=s["eliminated_features_names"])
    chk(api, "select_features with cat features (YetiRank)", sel_cat)
    chk(api, "get_borders n features", lambda: len(mn.get_borders()))
    chk(api, "shrink + drop_unused_features", lambda: (lambda c: (c.shrink(20), c.drop_unused_features(), c.tree_count_)[2])(mn.copy()))
    chk(api, "set_scale_and_bias", lambda: (lambda c: (c.set_scale_and_bias(2.0, 1.0), float(np.abs(c.predict(pnev) - (2 * mn.predict(pnev) + 1)).max()))[1])(mn.copy()))
    chk(api, "get_leaf_values / get_tree_leaf_counts", lambda: (len(mn.get_leaf_values()), mn.get_tree_leaf_counts()[:3].tolist()))
    # leaf indexes of a tree model as categorical inputs of a second model (GBDT -> GBDT stacking through leaves)
    def leaf_stack():
        Ltr, Lev = m.calc_leaf_indexes(ptr)[:, :20], m.calc_leaf_indexes(pev)[:, :20]
        dtr = pd.DataFrame(Ltr.astype(np.int32), columns=[f"leaf{i}" for i in range(20)]); dev = pd.DataFrame(Lev.astype(np.int32), columns=dtr.columns)
        s = CatBoostRanker(**dict(BASE, loss_function="YetiRank", one_hot_max_size=255))
        s.fit(Pool(dtr, label=D["y"][tr], group_id=D["gid"][tr], cat_features=list(dtr.columns)))
        return ndcg(s.predict(Pool(dev, group_id=D["gid"][ev], cat_features=list(dev.columns))).astype(float), D["y"][ev], D["gid"][ev])
    chk(api, "leaf indexes -> categorical features of a 2nd model (eval NDCG, se, n)", leaf_stack)
    chk(api, "CatBoostRanker(embedding_calcers=...) kwarg", lambda: CatBoostRanker(embedding_calcers=["KNN"]))
    chk(api, "CatBoostRanker(ctr_history_unit='Group') kwarg accepted by constructor", lambda: CatBoostRanker(ctr_history_unit="Group").get_params())

if SEC == "quant":
    X = D["x"].copy(); X[np.random.default_rng(0).random(X.shape) < 0.05] = np.nan
    def q(name, loss, extra):
        key = f"{name}|{loss}"
        try:
            m = CatBoost(dict(BASE, loss_function=loss, **extra))
            df = pd.DataFrame(X, columns=["x0", "x1", "x2", "x3"])
            m.fit(Pool(df[tr].reset_index(drop=True), label=D["y"][tr], group_id=D["gid"][tr]))
            nd = ndcg(np.asarray(m.predict(Pool(df[ev].reset_index(drop=True), group_id=D["gid"][ev])), np.float64), D["y"][ev], D["gid"][ev])
            nb = [len(b) for b in m.get_borders().values()]
            res[key] = dict(status="PASS", eval_ndcg=round(nd[0], 4), se=round(nd[1], 4), borders_per_feature=nb)
            print(f"PASS {key:55s} eval {nd[0]:.4f} +- {nd[1]:.4f} borders/feature {nb}", flush=True)
        except Exception as e:  # noqa
            res[key] = dict(status="FAIL", error=str(e).strip().replace("\n", " | ")[:400])
            print(f"FAIL {key:55s} {str(e).strip()[:300]}", flush=True)
        json.dump(res, open(OUT, "w"), indent=1)
    for loss in ["YetiRank", "Logloss"]:
        q("default", loss, {})
        for bc in [32, 254, 1024, 4096, 65535]:
            q(f"border_count_{bc}", loss, dict(border_count=bc))
        for bt in ["Median", "Uniform", "UniformAndQuantiles", "MaxLogSum", "MinEntropy", "GreedyLogSum"]:
            q(f"feature_border_type_{bt}", loss, dict(feature_border_type=bt))
        q("per_float_golden_x0_1024", loss, dict(per_float_feature_quantization=["0:border_count=1024"]))
        q("per_float_full_spec", loss, dict(per_float_feature_quantization=["0:border_count=1024,nan_mode=Max,border_type=Median", "1:border_count=16"]))
        q("per_float_colon_spec", loss, dict(per_float_feature_quantization=["0:border_count=1024:nan_mode=Max:border_type=Median"]))
        for nm in ["Min", "Max", "Forbidden"]:
            q(f"nan_mode_{nm}", loss, dict(nan_mode=nm))
print("written", OUT)
