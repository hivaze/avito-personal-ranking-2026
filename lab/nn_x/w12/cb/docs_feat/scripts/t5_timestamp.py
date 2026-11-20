"""T5: has_time + Pool(timestamp=...) with groups: which order reaches the ordered statistics? CatBoost 1.2.10 CPU, synthetic noise labels.
Rows inside each group are label-sorted (positives first). Feature = group-constant categorical id. Learn Logloss far below 0.6931 = the order leaks labels."""
import os, json, sys
assert os.environ.get("CUDA_VISIBLE_DEVICES", "x") == "", "CPU only"
import numpy as np, pandas as pd
from catboost import CatBoost, Pool
r = np.random.default_rng(11); G, S = 800, 10
gid = np.repeat(np.arange(G), S)
y = np.concatenate([np.r_[np.ones(5), np.zeros(5)] for _ in range(G)]).astype(np.float32)   # label-sorted inside each group
out = {}
def run(name, extra, ts=None, shuffle_within=False, cat=True):
    idx = np.arange(len(gid))
    if shuffle_within:
        idx = np.lexsort((r.random(len(gid)), gid))
    df = pd.DataFrame({"user_id": gid[idx].astype(np.int32)})
    try:
        p = Pool(df, label=y[idx], group_id=gid[idx], cat_features=["user_id"], timestamp=None if ts is None else ts[idx])
        m = CatBoost(dict(iterations=100, learning_rate=0.2, depth=4, thread_count=4, task_type="CPU", random_seed=0, allow_writing_files=False, verbose=0, loss_function="Logloss", **extra))
        m.fit(p); er = m.get_evals_result()["learn"]["Logloss"]
        out[name] = dict(learn_first=round(er[0], 4), learn_last=round(er[-1], 4))
    except Exception as e:
        out[name] = "FAIL " + str(e).strip().replace("\n", " | ")[:300]
    print(f"{name:75s} {out[name]}", flush=True)
ts_group = (gid * 1000).astype(np.int64)                      # one timestamp per group (like a feed timestamp)
ts_rand_within = (gid * 1000 + r.permutation(len(gid)) % 1000).astype(np.int64)   # random order inside a group
run("default permutations, label-sorted rows", {})
run("has_time=True, label-sorted rows, no timestamp", dict(has_time=True))
run("has_time=True, label-sorted rows, timestamp constant inside group", dict(has_time=True), ts=ts_group)
run("has_time=True, label-sorted rows, timestamp random inside group", dict(has_time=True), ts=ts_rand_within)
run("has_time=False, timestamp random inside group", {}, ts=ts_rand_within)
run("has_time=True, rows shuffled inside group beforehand", dict(has_time=True), shuffle_within=True)
ts_rev = ((G - gid) * 1000).astype(np.int64)
run("has_time=True, timestamps decreasing over groups (is the data re-sorted by timestamp?)", dict(has_time=True), ts=ts_rev)
json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "logs", "t5_timestamp.json"), "w"), indent=1)
