"""Staged val / holdout predictions of a saved famS model at fixed tree counts (for equal-round comparisons and level-vs-trees curves).
usage: CUDA_VISIBLE_DEVICES="" CBX_THREADS=4 python -u fs_staged.py <tag> <block none|raw|pca|cand> <n1,n2,...> [--dir preds]
writes famS/staged/<tag>@<n>_{val,hold}.npy and prints level per tree count (evaluation only)."""
import sys, json
import numpy as np
sys.path.insert(0, "<repo>/nn_x/w12/cb/famS")
import fs_common as fs
from fs_common import cbx, log
from catboost import CatBoost
tag, blk, ns = sys.argv[1], sys.argv[2], [int(v) for v in sys.argv[3].split(",")]
src = fs.HERE / (sys.argv[5] if len(sys.argv) > 5 and sys.argv[4] == "--dir" else "preds")
out = fs.HERE / "staged"; out.mkdir(exist_ok=True)
m = CatBoost(); m.load_model(str(src / f"{tag}.cbm"))
kw = dict(cat="native")
if blk != "none":
    kw["extra"] = fs.extra_cols(blk)
ev = cbx.Evaluator(); res = {}
ns = [n for n in ns if n <= m.tree_count_]
for sp in ("val", "hold"):
    pool = cbx.make_pool(cbx.load(sp), **kw)
    for n in ns:
        p = np.asarray(m.predict(pool, ntree_end=n, thread_count=cbx.THREADS), np.float32)
        np.save(out / f"{tag}@{n}_{sp}.npy", p)
        e = ev.ndcg(sp, p); res.setdefault(n, {})[sp] = float(e.mean())
    del pool
for n in ns:
    print(f"{tag}@{n}: val {res[n]['val']:.5f} holdout {res[n]['hold']:.5f}", flush=True)
(out / f"{tag}.staged.json").write_text(json.dumps(res))
