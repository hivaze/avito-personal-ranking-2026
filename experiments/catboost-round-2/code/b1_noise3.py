# Leak test for CatBoost's native target statistics, run before the real arm, in the form that
# settled it: 2,000 feeds with noise labels and the exact arm configuration. Native statistics on
# candidate-level keys are compared with hand-made twins that cannot leak (built from earlier
# feeds only); the user id is added as a positive control.
# How it was run: `python -u b1_noise3.py` from lab/nn_x/w12/cb/famB (CPU, 313 s, 2026-11-15
# 01:36 UTC). Its helper fb_common.py is not copied.
# What it produced: logs/b1_noise3.log (results/famB_native_ctr_noise_test.txt). On noise the user
# key reaches learn NDCG 0.558 against 0.266 for the twin; candidate-level keys next to the 177
# features show no excess (+0.002 +- 0.008). User-level native keys were never used.
# Kept as it ran: paths and names refer to the lab tree (see lab/ and docs/glossary.md).
# Original: lab/nn_x/w12/cb/famB/b1_noise3.py
"""B1 leak test, part 3: the EXACT arm configuration with 255 CTR borders (fb_common.CTR255), 2,000 xm feeds, noise labels.
  A  cat-only keys (category_id, cl256, cl1024, cl4096 native), max_ctr_complexity 2 / 4:
       YetiRank learn NDCG after 100 trees, labels shuffled INSIDE feeds ('in');
       Logloss learn loss after 100 trees, 'in' vs labels shuffled across the pool ('glob'): rows are iid under Logloss, so the two are scale-matched
  B  positive control: A (mcc 2) + user_id as a native key (feed-constant: must leak)
  C  exact arm inputs (177 features + the keys): native mcc 2 / 4  vs  the keys replaced by hand-made ordered statistics computed
       'group' = from earlier FEEDS only (cannot leak)   and   'prefix' = from all earlier rows including the same feed (what CPU does)
Twin columns per key: (clicks + 0.5) / (count + 1) and count. Label seeds 0-2 are the seeds of b1_noise.py / b1_noise2.py.
usage: python -u b1_noise3.py"""
import json, sys, time
sys.path.insert(0, "<repo>/nn_x/w12/cb/famB")
from fb_common import *
from catboost import CatBoostRanker, CatBoostClassifier, Pool
import pandas as pd
import numba as nb

NF, TREES, SEEDS = 2000, 100, (0, 1, 2)
tr = cbx.load("xm")
rg = np.random.default_rng(7)
keep = np.zeros(tr.n_feeds, bool); keep[rg.choice(tr.n_feeds, NF, replace=False)] = True
sh = ShufDS(tr, seed=0)
rm = np.repeat(keep, tr.M)
idx = sh.perm[rm]; srt = np.argsort(idx); X = np.empty((len(idx), len(tr.names)), np.float32); X[srt] = tr.X[idx[srt]]
gid = np.asarray(sh.gid)[rm]; y0 = sh.y[rm].astype(np.float32)
M = tr.M[keep]; off = np.r_[0, np.cumsum(M)[:-1]].astype(np.int64)
keys = {"category_id": np.nan_to_num(X[:, tr.names.index("category_id")], nan=-1).astype(np.int64)}
for K in (256, 1024, 4096):
    keys[f"cl{K}"] = clu(sh, K)[rm].astype(np.int64)
uid = sh.uid[rm]; uid = np.where(uid > 0, uid, -1 - gid).astype(np.int64)
kidx = {k: np.unique(v, return_inverse=True)[1].astype(np.int64) for k, v in keys.items()}


def labels(mode, seed):
    r = np.random.default_rng(100 + seed)
    if mode == "glob":
        return y0[r.permutation(len(y0))]
    key = r.random(len(y0))
    return y0[np.lexsort((key, gid))]


@nb.njit(cache=True)
def ordered(k, y, off, M, nk, group):
    n = np.zeros(nk); p = np.zeros(nk); on = np.empty(len(k), np.float32); op = np.empty(len(k), np.float32)
    for f in range(len(M)):
        a = off[f]; b = a + M[f]
        for i in range(a, b):
            on[i] = n[k[i]]; op[i] = p[k[i]]
            if not group:
                n[k[i]] += 1; p[k[i]] += y[i]
        if group:
            for i in range(a, b):
                n[k[i]] += 1; p[k[i]] += y[i]
    return on, op


def pool(kind, y, with_uid=False):
    """kind: 'cat' (keys only, native), 'full' (177 + keys native), 'tw_group' / 'tw_prefix' (177 with category native + twin columns)."""
    if kind == "cat":
        df = pd.DataFrame({k: v for k, v in keys.items()}); cats = list(keys)
    else:
        df = cbx.to_df(X.copy(), tr.names, "native"); cats = ["category_id"]
        if kind == "full":
            for k in ("cl256", "cl1024", "cl4096"):
                df[k] = keys[k]; cats.append(k)
        else:
            for k in keys:
                n, p = ordered(kidx[k], y.astype(np.float64), off, M, int(kidx[k].max()) + 1, kind == "tw_group")
                df[f"tw_{k}_ctr"] = (p + 0.5) / (n + 1.0); df[f"tw_{k}_n"] = n
    if with_uid:
        df["user_id"] = uid; cats.append("user_id")
    return Pool(df, label=y, group_id=gid, cat_features=cats, thread_count=cbx.THREADS)


def yeti(p, mcc):
    kw = b0_params(seed=0, iterations=TREES, metric_period=1, has_time=True, max_ctr_complexity=mcc, simple_ctr=CTR255, combinations_ctr=CTR255)
    kw.pop("od_type"); kw.pop("od_wait"); kw["eval_metric"] = "NDCG:type=Base;hints=skip_train~false"
    m = CatBoostRanker(**kw).fit(p, verbose=0)
    c = m.evals_result_["learn"]; k = [k_ for k_ in c if k_.startswith("NDCG")][0]
    return float(c[k][0]), float(c[k][-1])


def logl(p, mcc):
    m = CatBoostClassifier(loss_function="Logloss", iterations=TREES, learning_rate=0.2, depth=8, l2_leaf_reg=5, border_count=254, random_seed=0, has_time=True,
                           max_ctr_complexity=mcc, simple_ctr=CTR255, combinations_ctr=CTR255, allow_writing_files=False, thread_count=cbx.THREADS).fit(p, verbose=0)
    return float(m.evals_result_["learn"]["Logloss"][-1])


res = []; t0 = time.time()
for seed in SEEDS:
    yi, yg = labels("in", seed), labels("glob", seed)
    for mcc in (2, 4):
        f, l = yeti(pool("cat", yi), mcc)
        r = dict(test="A", mcc=mcc, seed=seed, yr_first=f, yr_last=l, ll_in=logl(pool("cat", yi), mcc), ll_glob=logl(pool("cat", yg), mcc)); res.append(r); log(json.dumps(r))
    f, l = yeti(pool("cat", yi, True), 2)
    r = dict(test="B", mcc=2, seed=seed, yr_first=f, yr_last=l, ll_in=logl(pool("cat", yi, True), 2), ll_glob=logl(pool("cat", yg, True), 2)); res.append(r); log(json.dumps(r))
    for kind, mcc in (("full", 2), ("full", 4), ("tw_group", 4), ("tw_prefix", 4)):
        f, l = yeti(pool(kind, yi), mcc)
        r = dict(test="C", kind=kind, mcc=mcc, seed=seed, yr_first=f, yr_last=l); res.append(r); log(json.dumps(r))
json.dump(res, open(HERE / "out/b1_noise3.json", "w"), indent=1)
log(f"runs done in {time.time() - t0:.0f}s")
ms = lambda a: f"{np.mean(a):.5f} +- {np.std(a, ddof=1) / np.sqrt(len(a)):.5f}"
tw = json.load(open(HERE / "out/b1_noise2.json"))
twv = lambda mode: np.array([r["learn_last"] for r in tw if not r["control"] and r["mode"] == mode and r["seed"] in SEEDS])
print("\nSUMMARY (2,000 feeds, noise labels, 100 trees; mean +- SE over 3 label seeds)")
print(f"twin, keys only (b1_noise2): YetiRank learn NDCG group {ms(twv('group'))} | prefix {ms(twv('prefix'))} | prefix - group {ms(twv('prefix') - twv('group'))}")
for t, mcc in (("A", 2), ("A", 4), ("B", 2)):
    rr = [r for r in res if r["test"] == t and r["mcc"] == mcc]
    yl = np.array([r["yr_last"] for r in rr]); d = np.array([r["ll_glob"] - r["ll_in"] for r in rr])
    print(f"{'keys only' if t == 'A' else 'CONTROL + user_id'} native mcc {mcc}: YetiRank learn NDCG {ms(yl)} (minus group twin {ms(yl - twv('group'))}) | "
          f"Logloss learn loss in {ms([r['ll_in'] for r in rr])} glob {ms([r['ll_glob'] for r in rr])} | excess drop under 'in' {ms(d)}")
g = lambda kind, mcc: np.array([r["yr_last"] for r in res if r["test"] == "C" and r["kind"] == kind and r["mcc"] == mcc])
print(f"exact arm inputs: native mcc 2 {ms(g('full', 2))} | native mcc 4 {ms(g('full', 4))} | group twin {ms(g('tw_group', 4))} | prefix twin {ms(g('tw_prefix', 4))}")
print(f"   native mcc 2 - group twin {ms(g('full', 2) - g('tw_group', 4))} | native mcc 4 - group twin {ms(g('full', 4) - g('tw_group', 4))} | prefix twin - group twin {ms(g('tw_prefix', 4) - g('tw_group', 4))}")
