"""B1 leak test, part 2: the scale-matched control for b1_noise.py. Same 2,000 feeds, same noise labels shuffled INSIDE feeds.
The ordered target statistic of each key is computed by hand in two ways and given to the B0 recipe as numeric columns:
  'prefix' = sample-level history (what CatBoost CPU does with has_time: all earlier rows INCLUDING the same feed's earlier rows)
  'group'  = group-level history (earlier FEEDS only: what ctr_history_unit=Group would do; cannot leak by construction)
Both twins have the same columns, the same learner and the same labels, so gain(prefix) - gain(group) isolates the within-feed prefix.
Keys: category_id, cl256, cl1024, cl4096 (the B1 keys); positive control: + user_id (feed-constant).
Also printed: mean within-feed correlation between each statistic and the noise label (0 = no information).
usage: python -u b1_noise2.py"""
import json, sys, time
sys.path.insert(0, "<repo>/nn_x/w12/cb/famB")
from fb_common import *
from catboost import CatBoostRanker, Pool
import numba as nb

NF, TREES, SEEDS = 2000, 100, (0, 1, 2, 3, 4)
tr = cbx.load("xm")
rg = np.random.default_rng(7)
keep = np.zeros(tr.n_feeds, bool); keep[rg.choice(tr.n_feeds, NF, replace=False)] = True
sh = ShufDS(tr, seed=0)
rm = np.repeat(keep, tr.M)
gid = np.asarray(sh.gid)[rm]; y0 = sh.y[rm].astype(np.float32)
M = tr.M[keep]; off = np.r_[0, np.cumsum(M)[:-1]].astype(np.int64)
keys = {"category_id": sh.take(lambda b: b.cat)[rm].astype(np.int64)}
for K in (256, 1024, 4096):
    keys[f"cl{K}"] = clu(sh, K)[rm].astype(np.int64)
uid = sh.uid[rm]; keys["user_id"] = np.where(uid > 0, uid, -1 - gid).astype(np.int64)
keys = {k: np.unique(v, return_inverse=True)[1].astype(np.int64) for k, v in keys.items()}


def labels(seed):
    r = np.random.default_rng(100 + seed); key = r.random(len(y0))
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


def feed_corr(x, y):
    return float(cbx._feed_corr(off, M, cbx.feed_z(off, M, np.ascontiguousarray(x, np.float32)), cbx.feed_z(off, M, np.ascontiguousarray(y, np.float32))).mean())


res = []
for ctl in (False, True):
    ks = [k for k in keys if ctl or k != "user_id"]
    for seed in SEEDS:
        y = labels(seed)
        for mode in ("prefix", "group"):
            cols, cors = [], {}
            for k in ks:
                n, p = ordered(keys[k], y.astype(np.float64), off, M, int(keys[k].max()) + 1, mode == "group")
                ctr = (p + 0.5) / (n + 1.0)
                cols += [ctr, n]; cors[k] = feed_corr(ctr, y)
            X = np.stack(cols, 1).astype(np.float32)
            kw = b0_params(seed=0, iterations=TREES, metric_period=1); kw.pop("od_type"); kw.pop("od_wait")
            kw["eval_metric"] = "NDCG:type=Base;hints=skip_train~false"
            m = CatBoostRanker(**kw).fit(Pool(X, label=y, group_id=gid, thread_count=cbx.THREADS), verbose=0)
            c = m.evals_result_["learn"]; kk = [k_ for k_ in c if k_.startswith("NDCG")][0]
            r = dict(control=ctl, seed=seed, mode=mode, learn_first=float(c[kk][0]), learn_last=float(c[kk][-1]), corr=cors)
            res.append(r); log(json.dumps(r))
json.dump(res, open(HERE / "out/b1_noise2.json", "w"), indent=1)
print("\nSUMMARY: learn NDCG after 100 trees on noise labels, prefix twin minus group twin, paired by label seed")
for ctl in (False, True):
    a = np.array([r["learn_last"] for r in res if r["control"] == ctl and r["mode"] == "prefix"])
    b = np.array([r["learn_last"] for r in res if r["control"] == ctl and r["mode"] == "group"])
    d = a - b; se = d.std(ddof=1) / np.sqrt(len(d))
    cp = {k: np.mean([r["corr"][k] for r in res if r["control"] == ctl and r["mode"] == "prefix"]) for k in res[-1 if ctl else 0]["corr"]}
    cg = {k: np.mean([r["corr"][k] for r in res if r["control"] == ctl and r["mode"] == "group"]) for k in res[-1 if ctl else 0]["corr"]}
    print(f"{'CONTROL (+ user_id)' if ctl else 'B1 keys'}: prefix {a.mean():.5f} group {b.mean():.5f} diff {d.mean():+.5f} +- {se:.5f} (n={len(d)} label seeds, z {d.mean() / se:+.1f})")
    print("   within-feed corr(statistic, noise label): prefix", {k: round(v, 4) for k, v in cp.items()}, "| group", {k: round(v, 4) for k, v in cg.items()})
