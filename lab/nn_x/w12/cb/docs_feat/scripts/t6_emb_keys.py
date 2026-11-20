"""T6: which index does the per-feature key of embedding_processing use (flat feature index or index among embedding features)? + embedding_features_data layout.
CatBoost 1.2.10 CPU, synthetic. Pool columns: x0..x3 (flat 0..3), emb_item (flat 4, embedding #0), emb_prod (flat 5, embedding #1)."""
import os, sys, json
assert os.environ.get("CUDA_VISIBLE_DEVICES", "x") == "", "CPU only"
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from synth import make, frame, split
from catboost import CatBoost, Pool
D = make(n_groups=200, gsize=50, seed=2); tr, ev = split(D, 150)
def pool(mask, how):
    df = frame(D, True).loc[mask].reset_index(drop=True)
    if how == "df":
        df["emb_item"] = list(D["emb_item"][mask]); df["emb_prod"] = list(D["emb_prod"][mask])
        return Pool(df, label=D["y"][mask], group_id=D["gid"][mask], embedding_features=["emb_item", "emb_prod"])
    return Pool(df, label=D["y"][mask], group_id=D["gid"][mask], embedding_features=["emb_item", "emb_prod"],
                embedding_features_data={"emb_item": D["emb_item"][mask], "emb_prod": D["emb_prod"][mask]})
out = {}
def run(name, ep, how="df"):
    try:
        m = CatBoost(dict(iterations=60, depth=4, learning_rate=0.2, thread_count=4, task_type="CPU", random_seed=0, allow_writing_files=False, verbose=0, loss_function="Logloss", **ep))
        p = pool(tr, how); m.fit(p)
        sp = set()
        for t in range(m.tree_count_):
            for s in m._get_tree_splits(t, p):
                if "calcer_id" in s:
                    sp.add((s.split("src_feature_id=")[1].split(" ")[0], s.split("calcer_id=")[1].split(" ")[0][:6], s.split("local_id=")[1].split(",")[0].split(" ")[0]))
        per = {}
        for f, c, l in sp:
            per.setdefault(f, {}).setdefault(c, set()).add(l)
        out[name] = {f"embedding#{f}": {c: len(l) for c, l in v.items()} for f, v in sorted(per.items())}
        out[name]["feature_names"] = m.feature_names_
    except Exception as e:
        out[name] = "FAIL " + str(e).strip().replace("\n", " | ")[:300]
    print(f"{name:58s} {out[name]}", flush=True)
E = lambda d: dict(embedding_processing={"embedding_processing": d})
run("default (LDA 1 feature + KNN 2 features per column)", {})
run("key '0' -> KNN only", E({"default": ["LDA:components=3"], "0": ["KNN:k=5"]}))
run("key '1' -> KNN only", E({"default": ["LDA:components=3"], "1": ["KNN:k=5"]}))
run("key '4' -> KNN only", E({"default": ["LDA:components=3"], "4": ["KNN:k=5"]}))
run("key '5' -> KNN only", E({"default": ["LDA:components=3"], "5": ["KNN:k=5"]}))
run("embedding_features_data dict of 2-D arrays, default", {}, how="data")
run("embedding_features_data, key '0' -> KNN only", E({"default": ["LDA:components=3"], "0": ["KNN:k=5"]}), how="data")
json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "logs", "t6_emb_keys.json"), "w"), indent=1, default=str)
