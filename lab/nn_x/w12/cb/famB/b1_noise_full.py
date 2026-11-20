"""B1 leak test at the REAL scale: the 120,591 xm fit feeds (17.7M rows), rows shuffled inside feeds, feeds in time order, has_time=True,
CTR255, max_ctr_complexity 4, features = the native keys ONLY (category_id, cl256, cl1024, cl4096), labels = NOISE.
With 17.7M rows and four keys an honest model cannot fit noise (in-sample capacity is negligible), so:
  Logloss, labels shuffled inside feeds ('in') vs across the pool ('glob'): learn loss after N trees must be equal (and equal to the base-rate entropy)
  YetiRank (the arm's loss), labels 'in': learn NDCG must stay at the random-score level
usage: python -u b1_noise_full.py [trees=30]"""
import json, sys, time
sys.path.insert(0, "<repo>/nn_x/w12/cb/famB")
from fb_common import *
from catboost import CatBoostRanker, CatBoostClassifier, Pool
import pandas as pd

TREES = int(sys.argv[1]) if len(sys.argv) > 1 else 30
tr = cbx.load("xm"); fit, _ = cbx.trainside_split(tr)
sh = ShufDS(tr, seed=0); rm = np.repeat(fit, tr.M)
gid = np.asarray(sh.gid)[rm]; y0 = sh.y[rm].astype(np.float32)
M = tr.M[fit]; off = np.r_[0, np.cumsum(M)[:-1]].astype(np.int64)
df = pd.DataFrame({"category_id": sh.take(lambda b: b.cat)[rm].astype(np.int64)})
for K in (256, 1024, 4096):
    df[f"cl{K}"] = clu(sh, K)[rm].astype(np.int64)
cats = list(df.columns)
r = np.random.default_rng(100)
y_glob = y0[r.permutation(len(y0))]
r = np.random.default_rng(100); y_in = y0[np.lexsort((r.random(len(y0)), gid))]
p = float(y0.mean()); ent = -(p * np.log(p) + (1 - p) * np.log(1 - p))
rnd = np.random.default_rng(5).random(len(y0)).astype(np.float32)
chance = float(cbx.ndcg_fast(rnd, y_in, off, M).mean())
log(f"pool: {len(M)} feeds, {len(y0)} rows, base rate {p:.5f}, base-rate entropy {ent:.6f}, random-score NDCG on the 'in' labels {chance:.5f}")
res = dict(entropy=ent, chance_ndcg=chance, trees=TREES)
for mode, y in (("in", y_in), ("glob", y_glob)):
    t0 = time.time()
    m = CatBoostClassifier(loss_function="Logloss", iterations=TREES, learning_rate=0.2, depth=8, l2_leaf_reg=5, border_count=254, random_seed=0, has_time=True,
                           max_ctr_complexity=4, simple_ctr=CTR255, combinations_ctr=CTR255, allow_writing_files=False, thread_count=cbx.THREADS)
    m.fit(Pool(df, label=y, cat_features=cats, thread_count=cbx.THREADS), verbose=10)
    c = m.evals_result_["learn"]["Logloss"]; res[f"logloss_{mode}"] = [float(c[0]), float(c[-1])]
    log(f"Logloss labels {mode}: learn first {c[0]:.6f} last {c[-1]:.6f} (entropy {ent:.6f}; last - entropy {c[-1] - ent:+.6f}) {time.time() - t0:.0f}s")
    json.dump(res, open(HERE / "out/b1_noise_full.json", "w"), indent=1)
log(f"LEAK SIZE (Logloss, glob - in, after {TREES} trees): {res['logloss_glob'][1] - res['logloss_in'][1]:+.6f}")
t0 = time.time()
kw = b0_params(seed=0, iterations=TREES, metric_period=5, has_time=True, max_ctr_complexity=4, simple_ctr=CTR255, combinations_ctr=CTR255)
kw.pop("od_type"); kw.pop("od_wait"); kw["eval_metric"] = "NDCG:type=Base;hints=skip_train~false"
m = CatBoostRanker(**kw).fit(Pool(df, label=y_in, group_id=gid, cat_features=cats, thread_count=cbx.THREADS), verbose=5)
c = m.evals_result_["learn"]; k = [k_ for k_ in c if k_.startswith("NDCG")][0]
res["yeti_in"] = [float(x) for x in c[k]]
log(f"YetiRank labels in: learn NDCG first {c[k][0]:.5f} last {c[k][-1]:.5f} (random-score level {chance:.5f}; last - chance {c[k][-1] - chance:+.5f}) {time.time() - t0:.0f}s")
json.dump(res, open(HERE / "out/b1_noise_full.json", "w"), indent=1)
