"""B1 leak test (docs_feat 5.3 design, adapted to candidate-level keys): on 2,000 xm feeds, labels are NOISE.
  labels 'in'   = the feed's own labels shuffled INSIDE the feed (number of positives per feed kept: the within-feed dependence that a
                  same-feed CTR prefix could exploit is present)
  labels 'glob' = labels shuffled across the WHOLE 2,000-feed pool (no within-feed dependence: only honest in-sample fit is possible)
  features: 'cat'  = the native categorical keys only (category_id, cl256, cl1024, cl4096)
            'full' = the exact arm inputs (177 features, category_id native, + 3 cluster keys native)
            'ctl'  = 'cat' + user_id as a native key (POSITIVE CONTROL: a feed-constant key must leak on CPU)
  model: the exact B1 configuration (YetiRank d8 lr 0.2 l2 5, has_time=True, rows shuffled inside feeds, feeds in time order,
         max_ctr_complexity 2 / 4), 100 trees; learn NDCG is forced on (hints=skip_train~false).
Criterion (fixed in PROGRESS.md before the run): gain(in) - gain(glob) <= 2 seed-SD for 'cat' and 'full'; 'ctl' must show a clear excess.
usage: python -u b1_noise.py"""
import json, sys, time
sys.path.insert(0, "<repo>/nn_x/w12/cb/famB")
from fb_common import *
from catboost import CatBoostRanker, CatBoostClassifier, Pool
import pandas as pd

NF, TREES = 2000, 100
tr = cbx.load("xm")
rg = np.random.default_rng(7)
keep = np.zeros(tr.n_feeds, bool); keep[rg.choice(tr.n_feeds, NF, replace=False)] = True
sh = ShufDS(tr, seed=0)
rm = np.repeat(keep, tr.M)
X = sh.X[rm]; gid = np.asarray(sh.gid)[rm]; y0 = sh.y[rm].astype(np.float32)
cl = {K: clu(sh, K)[rm] for K in (256, 1024, 4096)}
uid = sh.uid[rm]; uid = np.where(uid > 0, uid, -1 - gid)    # feed-constant key (unknown users: one key per feed)
M = tr.M[keep]; off = np.r_[0, np.cumsum(M)[:-1]]
log(f"noise pool: {NF} feeds, {len(y0)} rows, positives {int(y0.sum())}, base rate {y0.mean():.5f}")


def labels(mode, seed):
    r = np.random.default_rng(100 + seed)
    if mode == "glob":
        return y0[r.permutation(len(y0))]
    key = r.random(len(y0)); p = np.lexsort((key, gid))
    return y0[p]                                              # rows of a feed receive the feed's labels in a random order


def pool(feats, y):
    names = tr.names
    if feats == "full":
        df = cbx.to_df(X, names, "native"); cats = ["category_id"]
    else:
        df = pd.DataFrame({"category_id": np.nan_to_num(X[:, names.index("category_id")], nan=-1).astype(np.int32)}); cats = ["category_id"]
    for K in (256, 1024, 4096):
        df[f"cl{K}"] = cl[K].astype(np.int64); cats.append(f"cl{K}")
    if feats == "ctl":
        df["user_id"] = uid.astype(np.int64); cats.append("user_id")
    return Pool(df, label=y, group_id=gid, cat_features=cats, thread_count=cbx.THREADS)


res = []
t0 = time.time()
for feats, mccs in (("cat", (2, 4)), ("ctl", (2,)), ("full", (2, 4))):
    for mcc in mccs:
        for mode in ("in", "glob"):
            for seed in (0, 1, 2):
                y = labels(mode, seed)
                kw = b0_params(seed=0, iterations=TREES, has_time=True, max_ctr_complexity=mcc, metric_period=1, od_type=None, od_wait=None)
                kw["eval_metric"] = "NDCG:type=Base;hints=skip_train~false"
                kw = {k: v for k, v in kw.items() if v is not None}
                m = CatBoostRanker(**kw).fit(pool(feats, y), verbose=0)
                c = m.evals_result_["learn"]; k = [k_ for k_ in c if k_.startswith("NDCG")][0]
                first, last = float(c[k][0]), float(c[k][-1])
                r = dict(feats=feats, mcc=mcc, labels=mode, seed=seed, loss="YetiRank", learn_first=first, learn_last=last, gain=last - first)
                # the same CTR configuration under Logloss (rows iid: the scale of 'in' and 'glob' is identical); cat-only / control only
                if feats != "full":
                    mc = CatBoostClassifier(loss_function="Logloss", iterations=TREES, learning_rate=0.2, depth=8, l2_leaf_reg=5, border_count=254,
                                            random_seed=0, has_time=True, max_ctr_complexity=mcc, allow_writing_files=False,
                                            thread_count=cbx.THREADS).fit(pool(feats, y), verbose=0)
                    cl_ = mc.evals_result_["learn"]["Logloss"]
                    r.update(ll_first=float(cl_[0]), ll_last=float(cl_[-1]), ll_gain=float(cl_[0] - cl_[-1]))
                res.append(r)
                log(json.dumps(r))
json.dump(res, open(HERE / "out/b1_noise.json", "w"), indent=1)
log(f"runs done in {time.time() - t0:.0f}s")
print("\nSUMMARY (learn NDCG gain first -> last tree, mean +- SD over 3 label seeds)")
import itertools
ok = True
for feats, mcc in (("cat", 2), ("cat", 4), ("ctl", 2), ("full", 2), ("full", 4)):
    g = {mode: np.array([r["gain"] for r in res if r["feats"] == feats and r["mcc"] == mcc and r["labels"] == mode]) for mode in ("in", "glob")}
    d = g["in"].mean() - g["glob"].mean(); sd = np.sqrt((g["in"].var(ddof=1) + g["glob"].var(ddof=1)) / 2)
    line = f"{feats:5s} mcc {mcc}: in {g['in'].mean():+.5f} +- {g['in'].std(ddof=1):.5f} | glob {g['glob'].mean():+.5f} +- {g['glob'].std(ddof=1):.5f} | excess {d:+.5f} (seed-SD {sd:.5f})"
    if feats != "full":
        l = {mode: np.array([r["ll_gain"] for r in res if r["feats"] == feats and r["mcc"] == mcc and r["labels"] == mode]) for mode in ("in", "glob")}
        line += f" | Logloss learn drop in {l['in'].mean():.5f} glob {l['glob'].mean():.5f} excess {l['in'].mean() - l['glob'].mean():+.5f}"
    if feats != "ctl":
        passed = d <= 2 * sd; ok &= bool(passed); line += f" -> {'PASS' if passed else 'LEAK?'}"
    print(line)
print("B1 NOISE TEST:", "PASS" if ok else "CHECK")
