"""T2b: embedding calcers, firmer synthetic numbers (500 groups x 50 rows, 300 train / 200 eval groups) + option formats.
Which calcer carries information under which target type? hard binary labels vs soft real-valued (teacher-like) labels.
Writes logs/t2b_embedding_more.json
"""
import json, os, sys, time
assert os.environ.get("CUDA_VISIBLE_DEVICES", "x") == "", "CPU only"
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from synth import make, frame, split, ndcg
from catboost import CatBoost, Pool

OUT = os.path.join(os.path.dirname(__file__), "..", "logs", "t2b_embedding_more.json")
D = make(n_groups=500, gsize=50, n_items=2500, seed=5)
tr, ev = split(D, 300)
BASE = dict(iterations=80, learning_rate=0.15, depth=4, thread_count=4, task_type="CPU", random_seed=0, allow_writing_files=False, verbose=0)
res = {}
per_feed = {}


def pool(mask, label, embs, num=False):
    df = frame(D, num).loc[mask].reset_index(drop=True)
    for e in embs:
        df[e] = list(D[e][mask])
    return Pool(df, label=label[mask], group_id=D["gid"][mask], embedding_features=list(embs))


def run(name, loss, label, extra=None, embs=("emb_prod",), num=False):
    key = f"{name}|{loss}"
    try:
        m = CatBoost(dict(BASE, loss_function=loss, **(extra or {}))); m.fit(pool(tr, label, embs, num))
        nd = ndcg(np.asarray(m.predict(pool(ev, label, embs, num)), np.float64), D["y"][ev], D["gid"][ev])
        res[key] = dict(status="PASS", eval_ndcg=round(nd[0], 4), se=round(nd[1], 4), n=nd[2],
                        imp=dict(zip(m.feature_names_, [round(float(v), 1) for v in m.get_feature_importance(type="PredictionValuesChange")])))
        print(f"PASS {key:62s} eval NDCG {nd[0]:.4f} +- {nd[1]:.4f} (n={nd[2]}) {res[key]['imp'] if num else ''}", flush=True)
    except Exception as e:  # noqa
        res[key] = dict(status="FAIL", error=str(e).strip().replace("\n", " | ")[:500])
        print(f"FAIL {key:62s} {str(e).strip()[:400]}", flush=True)
    json.dump(res, open(OUT, "w"), indent=1)


rnd = ndcg(np.random.default_rng(1).normal(size=ev.sum()), D["y"][ev], D["gid"][ev]); dot = ndcg(D["emb_prod"][ev].sum(1).astype(float), D["y"][ev], D["gid"][ev])
print(f"references: random {rnd[0]:.4f} +- {rnd[1]:.4f}; plain dot <u,e> {dot[0]:.4f} +- {dot[1]:.4f}; n eval groups {rnd[2]}", flush=True)
res["_random"] = rnd; res["_dot"] = dot
soft01 = (1 / (1 + np.exp(-D["soft"]))).astype(np.float32)
for cal, ex in [("default", {}), ("KNN5", dict(embedding_calcers=["KNN:k=5"])), ("KNN50", dict(embedding_calcers=["KNN:k=50"])), ("LDA1", dict(embedding_calcers=["LDA:components=1"])),
                ("LDA4", dict(embedding_calcers=["LDA:components=4"])), ("LDA1_likelihood", dict(embedding_calcers=["LDA:components=1:likelihood=true"]))]:
    run(f"hard_{cal}", "Logloss", D["y"], ex)
    run(f"hard_{cal}", "YetiRank", D["y"], ex)
    run(f"soft_{cal}", "YetiRank", D["soft"], ex)
    run(f"soft_{cal}", "RMSE", D["soft"], ex)
    run(f"soft01_{cal}", "CrossEntropy", soft01, ex)
# option formats
run("fmt_embedding_processing_nested", "Logloss", D["y"], dict(embedding_processing={"embedding_processing": {"default": ["LDA:components=1"]}}))
run("fmt_embedding_processing_per_feature", "Logloss", D["y"], dict(embedding_processing={"embedding_processing": {"default": ["LDA:components=1"], "0": ["KNN:k=7"], "1": ["LDA:components=2", "KNN:k=3"]}}),
    embs=("emb_prod", "emb_item"))
run("fmt_calcers_two_same_type", "Logloss", D["y"], dict(embedding_calcers=["KNN:k=5", "KNN:k=50", "LDA:components=1"]))
# value on top of numeric features (the dot product itself is given as a numeric feature: does the calcer add anything beyond it?)
D["x"] = np.c_[D["x"], D["emb_prod"].sum(1)].astype(np.float32)
import synth
_f = synth.frame
def frame5(Dd, num=True, cats=(), texts=()):
    df = pd.DataFrame(Dd["x"], columns=[f"x{i}" for i in range(Dd["x"].shape[1])]) if num else pd.DataFrame(index=np.arange(Dd["n"]))
    return df
frame = frame5
def pool(mask, label, embs, num=False):  # noqa: F811
    df = frame5(D, num).loc[mask].reset_index(drop=True)
    for e in embs:
        df[e] = list(D[e][mask])
    return Pool(df, label=label[mask], group_id=D["gid"][mask], embedding_features=list(embs) or None)
for loss, lab in [("Logloss", D["y"]), ("YetiRank", D["y"])]:
    run("num+dot (no embedding)", loss, lab, embs=(), num=True)
    run("num+dot + emb_prod default", loss, lab, num=True)
    run("num+dot + emb_item + emb_user + emb_prod", loss, lab, embs=("emb_item", "emb_user", "emb_prod"), num=True)
print("written", OUT)
