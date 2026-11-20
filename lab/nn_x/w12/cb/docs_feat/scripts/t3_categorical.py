"""T3: categorical-feature levers on CatBoost 1.2.10 CPU.
(a) parameter acceptance under YetiRank / Logloss / RMSE(soft): simple_ctr, combinations_ctr, per_feature_ctr, ctr types, ctr_target_border_count,
    counter_calc_method, max_ctr_complexity, one_hot_max_size, ctr_leaf_count_limit, store_all_simple_ctr, has_time, final_ctr_computation_mode,
    ctr_history_unit, model_size_reg, boosting_type.
(b) within-group prefix leak: labels are pure noise (5 positives in each group of 10); the only feature is a GROUP-CONSTANT categorical id (user id)
    or a group-constant embedding. An honest feature cannot reduce the learn loss; a prefix statistic that sees same-group labels can.
(c) has_time=True with label-sorted rows inside each group: does the input order inside a group reach the statistics?
Writes logs/t3_categorical.json
"""
import json, os, sys, time
assert os.environ.get("CUDA_VISIBLE_DEVICES", "x") == "", "CPU only"
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from synth import make, frame, split, ndcg
from catboost import CatBoost, Pool

SEC = sys.argv[1] if len(sys.argv) > 1 else "a"
OUT = os.path.join(os.path.dirname(__file__), "..", "logs", f"t3_categorical_{SEC}.json")
res = {}
D = make(n_groups=300, gsize=50, n_items=1500, seed=3)
tr, ev = split(D, 200)
BASE = dict(iterations=60, learning_rate=0.2, depth=4, thread_count=6, task_type="CPU", random_seed=0, allow_writing_files=False, verbose=0)
CATS = ("cat_id", "item_id", "user_id")


def pool(mask, label, cats=CATS, num=True):
    df = frame(D, num, cats).loc[mask].reset_index(drop=True)
    return Pool(df, label=label[mask], group_id=D["gid"][mask], cat_features=list(cats))


def run(name, loss, extra, label=None, cats=CATS, num=True, predict=True):
    key = f"{name}|{loss}"
    label = D["y"] if label is None else label
    t0 = time.time()
    try:
        m = CatBoost(dict(BASE, loss_function=loss, **extra))
        ptr, pev = pool(tr, label, cats, num), pool(ev, label, cats, num)
        m.fit(ptr)
        r = dict(status="PASS", trees=m.tree_count_, sec=round(time.time() - t0, 1))
        if predict:
            nd = ndcg(np.asarray(m.predict(pev), np.float64), D["y"][ev], D["gid"][ev])
            imp = dict(zip(m.feature_names_, [round(float(v), 1) for v in m.get_feature_importance(type="PredictionValuesChange")]))
            r.update(eval_ndcg=round(nd[0], 4), se=round(nd[1], 4), n=nd[2], imp=imp)
            try:
                ii = m.get_feature_importance(type="Interaction")
                r["interaction_top"] = [f"{m.feature_names_[int(a)]}x{m.feature_names_[int(b)]}:{c:.1f}" for a, b, c in ii[:3]]
            except Exception as e:  # noqa
                r["interaction_top"] = f"<{str(e)[:80]}>"
        res[key] = r
        print(f"PASS {key:58s} " + " ".join(f"{k}={v}" for k, v in r.items() if k not in ("status",)), flush=True)
    except Exception as e:  # noqa
        msg = str(e).strip().replace("\n", " | ")
        res[key] = dict(status="FAIL", error=msg[:700])
        print(f"FAIL {key:58s} {type(e).__name__}: {msg[:420]}", flush=True)
    json.dump(res, open(OUT, "w"), indent=1)


print("== (a) parameter acceptance ==", flush=True)
A = [
    ("default", {}),
    ("simple_Borders_priors_slash", dict(simple_ctr=["Borders:CtrBorderCount=15:Prior=0/1:Prior=0.5/1:Prior=1/1"])),
    ("simple_Borders_priors_plain", dict(simple_ctr=["Borders:CtrBorderCount=15:Prior=0:Prior=0.5:Prior=1"])),
    ("simple_Buckets", dict(simple_ctr=["Buckets"])),
    ("simple_BinarizedTargetMeanValue", dict(simple_ctr=["BinarizedTargetMeanValue"])),
    ("simple_Counter", dict(simple_ctr=["Counter"])),
    ("simple_Borders+Counter", dict(simple_ctr=["Borders", "Counter"])),
    ("simple_FeatureFreq", dict(simple_ctr=["FeatureFreq"])),
    ("simple_FloatTargetMeanValue", dict(simple_ctr=["FloatTargetMeanValue"])),
    ("combinations_Borders+Counter", dict(combinations_ctr=["Borders", "Counter"])),
    ("per_feature_ctr", dict(per_feature_ctr=["4:Borders:Prior=0.5", "5:Counter"])),
    ("ctr_target_border_count_5", dict(ctr_target_border_count=5)),
    ("simple_TargetBorderCount5_Uniform", dict(simple_ctr=["Borders:TargetBorderCount=5:TargetBorderType=Uniform"])),
    ("counter_calc_SkipTest", dict(counter_calc_method="SkipTest", simple_ctr=["Borders", "Counter"])),
    ("counter_calc_Full", dict(counter_calc_method="Full", simple_ctr=["Borders", "Counter"])),
    ("counter_calc_PrefixTest", dict(counter_calc_method="PrefixTest")),
    ("max_ctr_complexity_1", dict(max_ctr_complexity=1)),
    ("max_ctr_complexity_2", dict(max_ctr_complexity=2)),
    ("max_ctr_complexity_6", dict(max_ctr_complexity=6, depth=8)),
    ("one_hot_max_size_20", dict(one_hot_max_size=20)),
    ("one_hot_max_size_255", dict(one_hot_max_size=255)),
    ("one_hot_max_size_2000", dict(one_hot_max_size=2000)),
    ("ctr_leaf_count_limit_100", dict(ctr_leaf_count_limit=100)),
    ("ctr_leaf_count_limit_100_store_all", dict(ctr_leaf_count_limit=100, store_all_simple_ctr=True)),
    ("has_time", dict(has_time=True)),
    ("ctr_history_unit_Sample", dict(ctr_history_unit="Sample")),
    ("ctr_history_unit_Group", dict(ctr_history_unit="Group")),
    ("model_size_reg_0", dict(model_size_reg=0)),
    ("model_size_reg_5", dict(model_size_reg=5)),
    ("boosting_Ordered", dict(boosting_type="Ordered")),
    ("fold_permutation_block_4", dict(fold_permutation_block=4)),
    ("approx_on_full_history", dict(boosting_type="Ordered", approx_on_full_history=True)),
    ("grow_Lossguide", dict(grow_policy="Lossguide", max_leaves=16)),
    ("grow_Depthwise", dict(grow_policy="Depthwise", min_data_in_leaf=20)),
]
if SEC == "a2":
    A = [("per_feature_ctr_with_priors", dict(per_feature_ctr=["4:Borders:Prior=0.5/1", "5:Counter:Prior=0/1"])),
         ("per_feature_ctr_borders_only", dict(per_feature_ctr=["5:Borders:Prior=0.5/1:Prior=1/1"])),
         ("simple_Borders_CtrBorderCount_50", dict(simple_ctr=["Borders:CtrBorderCount=50:Prior=0.5/1"])),
         ("simple_Borders_CtrBorderCount_255", dict(simple_ctr=["Borders:CtrBorderCount=255:Prior=0.5/1"])),
         ("no_cat_features_baseline", None)]
for loss in (["YetiRank", "Logloss"] if SEC in ("a", "a2") else []):
    for n, ex in A:
        if ex is None:
            run(n, loss, {}, cats=())
        else:
            run(n, loss, ex)
if SEC == "skip":
  run("final_ctr_Skip", "YetiRank", dict(final_ctr_computation_mode="Skip"), predict=False)
if SEC == "skip":
  print("now predicting with a Skip model (may abort the process)", flush=True)
  try:
    m = CatBoost(dict(BASE, loss_function="YetiRank", final_ctr_computation_mode="Skip")); m.fit(pool(tr, D["y"])); m.predict(pool(ev, D["y"]))
    res["final_ctr_Skip_predict"] = "PASS (unexpected)"
  except Exception as e:  # noqa
    res["final_ctr_Skip_predict"] = "FAIL " + str(e).strip().replace("\n", " | ")[:300]
  print("final_ctr Skip then predict:", res["final_ctr_Skip_predict"], flush=True)
# soft real-valued labels with categorical features: target borders
for loss in (["RMSE", "YetiRank", "QueryRMSE"] if SEC == "soft" else []):
    run("soft_default_1border", loss, {}, label=D["soft"], num=False, cats=("cat_id", "item_id"))
    run("soft_ctr_target_border_count_15", loss, dict(ctr_target_border_count=15), label=D["soft"], num=False, cats=("cat_id", "item_id"))
    run("soft_BinarizedTargetMeanValue", loss, dict(simple_ctr=["BinarizedTargetMeanValue:TargetBorderCount=15"], combinations_ctr=["BinarizedTargetMeanValue:TargetBorderCount=15"]),
        label=D["soft"], num=False, cats=("cat_id", "item_id"))
    run("soft_Buckets15", loss, dict(simple_ctr=["Buckets:TargetBorderCount=15"]), label=D["soft"], num=False, cats=("cat_id", "item_id"))

print("== (b) within-group prefix leak (pure-noise labels) ==", flush=True)
r = np.random.default_rng(11)
G, S = 800, 10
gid = np.repeat(np.arange(G), S)
y = np.concatenate([r.permutation(np.r_[np.ones(5), np.zeros(5)]) for _ in range(G)]).astype(np.float32)
uvec = r.normal(size=(G, 8)).astype(np.float32)
trm = gid < 600
leak = {}


def leak_run(name, loss, extra, kind="cat", order=None):
    idx = np.arange(len(gid)) if order is None else order
    df = pd.DataFrame({"user_id": gid[idx].astype(np.int32)})
    kw = dict(cat_features=["user_id"])
    if kind == "emb":
        df = pd.DataFrame({"emb": list(uvec[gid[idx]])}); kw = dict(embedding_features=["emb"])
    mt = trm[idx]
    try:
        ptr = Pool(df[mt].reset_index(drop=True), label=y[idx][mt], group_id=gid[idx][mt], **kw)
        pev = Pool(df[~mt].reset_index(drop=True), label=y[idx][~mt], group_id=gid[idx][~mt], **kw)
        em = "Logloss" if loss == "Logloss" else "NDCG:hints=skip_train~false"
        m = CatBoost(dict(iterations=200, learning_rate=0.2, depth=4, thread_count=6, task_type="CPU", random_seed=0, allow_writing_files=False, verbose=0,
                          loss_function=loss, eval_metric=em, **extra))
        m.fit(ptr, eval_set=pev)
        er = m.get_evals_result()
        k = "Logloss" if loss == "Logloss" else [q for q in er["learn"] if q.startswith("NDCG")][0]
        l0, l1 = er["learn"][k][0], er["learn"][k][-1]
        v0, v1 = er["validation"][k][0], er["validation"][k][-1]
        leak[f"{name}|{loss}"] = dict(metric=k, learn_first=round(l0, 4), learn_last=round(l1, 4), eval_first=round(v0, 4), eval_last=round(v1, 4), trees=m.tree_count_)
        print(f"LEAK-TEST {name:46s} {loss:9s} {k}: learn {l0:.4f} -> {l1:.4f} | eval {v0:.4f} -> {v1:.4f} (trees {m.tree_count_})", flush=True)
    except Exception as e:  # noqa
        leak[f"{name}|{loss}"] = "FAIL " + str(e).strip().replace("\n", " | ")[:300]
        print(f"LEAK-TEST {name:46s} {loss:9s} FAIL {str(e).strip()[:300]}", flush=True)
    res["leak"] = leak
    json.dump(res, open(OUT, "w"), indent=1)


print(f"   reference: Logloss of a constant 0.5 prediction = {np.log(2):.4f}; any learn Logloss clearly below it comes from same-group labels", flush=True)
for loss in (["Logloss", "YetiRank"] if SEC == "leak" else ["YetiRank"] if SEC == "leak2" else []):
    leak_run("user_id default (Sample)", loss, {})
    leak_run("user_id ctr_history_unit=Group", loss, dict(ctr_history_unit="Group"))
    leak_run("user_id one_hot", loss, dict(one_hot_max_size=1000) if loss == "Logloss" else dict(one_hot_max_size=1000))
    leak_run("user_id has_time (input order random)", loss, dict(has_time=True))
    leak_run("user_id Counter only", loss, dict(simple_ctr=["Counter"], combinations_ctr=["Counter"]))
    leak_run("user_id boosting Ordered", loss, dict(boosting_type="Ordered"))
    leak_run("emb group-constant default (LDA+KNN)", loss, {}, kind="emb")
    leak_run("emb group-constant KNN only", loss, dict(embedding_calcers=["KNN:k=5"]), kind="emb")
    leak_run("emb group-constant LDA only", loss, dict(embedding_calcers=["LDA:components=1"]), kind="emb")
print("== (c) has_time with label-sorted rows inside each group ==", flush=True)
order = np.lexsort((-y, gid))  # inside each group: positives first
for loss in (["Logloss", "YetiRank"] if SEC == "leak" else ["YetiRank"] if SEC == "leak2" else []):
    leak_run("user_id default, rows label-sorted", loss, {}, order=order)
    leak_run("user_id has_time, rows label-sorted", loss, dict(has_time=True), order=order)
if SEC == "tbc":
    for loss in ["RMSE", "MAE", "Huber:delta=1", "Quantile", "LogCosh", "Poisson", "Tweedie:variance_power=1.5", "CrossEntropy", "Logloss", "QueryRMSE", "QuerySoftMax", "YetiRank", "YetiRankPairwise",
                 "LambdaMart", "StochasticRank:metric=NDCG", "PairLogit", "PairLogitPairwise"]:
        lab = (1 / (1 + np.exp(-D["soft"]))).astype(np.float32) if loss in ("CrossEntropy", "Poisson", "Tweedie:variance_power=1.5", "QuerySoftMax") else D["y"] if loss == "Logloss" else D["soft"]
        run("ctr_target_border_count_5", loss, dict(ctr_target_border_count=5), label=lab, num=False, cats=("cat_id", "item_id"))
print("written", OUT)
