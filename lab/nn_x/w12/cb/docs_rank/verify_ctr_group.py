"""docs_rank: group machinery x ordered target statistics. Synthetic. Does a categorical that is CONSTANT inside a feed (here: a
feed-unique id and a 'user' id shared by a few feeds) see the labels of the SAME feed while its CTR is computed in training?
Reads resolved ctr_history_unit and compares a model with / without the feed-level categorical. CPU, 4 threads."""
import os, json, sys
assert os.environ.get("CUDA_VISIBLE_DEVICES", "x") == ""
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from catboost import CatBoostRanker, Pool
TH = 4
def make(nf, seed):
    r = np.random.default_rng(seed); M = r.integers(20, 120, nf); n = int(M.sum()); off = np.r_[0, np.cumsum(M)[:-1]]; gid = np.repeat(np.arange(nf), M)
    X = r.normal(size=(n, 8)).astype(np.float32); w = np.random.default_rng(1).normal(size=8); u = X @ w; u = u / u.std()
    y = np.zeros(n, np.float32)
    for g in range(nf):
        s = slice(off[g], off[g] + M[g]); p = np.exp(2.5 * (u[s] - u[s].max())); p /= p.sum(); k = 1 + r.poisson(0.7)
        y[off[g] + r.choice(M[g], size=min(k, M[g] - 1), replace=False, p=p)] = 1
    user = np.repeat(r.integers(0, nf // 3, nf), M)          # a 'user' with about 3 feeds
    return X, y, gid, M, off, user
def ndcg(s, y, off, M):
    out = []
    for o, m in zip(off, M):
        t = y[o:o + m]; od = np.lexsort((t, -s[o:o + m])); d = 1 / np.log2(np.arange(m) + 2.0); out.append((t[od] * d).sum() / (np.sort(t)[::-1] * d).sum())
    return float(np.mean(out))
Xt, yt, gt, Mt, ot, ut = make(1500, 0); Xe, ye, ge, Me, oe, ue = make(500, 1)
def frame(X, gid, user, which, shift=0):
    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
    if "feed" in which: df["feed_cat"] = (gid + shift).astype(str)
    if "user" in which: df["user_cat"] = (user + shift).astype(str)
    return df
res = []
for which in [(), ("feed",), ("user",)]:
    for kw in [{}, dict(ctr_history_unit="Sample"), dict(ctr_history_unit="Group"), dict(boosting_type="Ordered"), dict(one_hot_max_size=2)]:
        cats = [c + "_cat" for c in which]
        try:
            m = CatBoostRanker(loss_function="YetiRank", iterations=150, learning_rate=0.1, depth=6, thread_count=TH, random_seed=0, verbose=False,
                               allow_writing_files=False, use_best_model=False, **kw)
            m.fit(Pool(frame(Xt, gt, ut, which), yt, group_id=gt, cat_features=cats or None, thread_count=TH))
            # eval feeds are NEW feeds / new users (ids shifted so they are unseen categories)
            p = m.predict(Pool(frame(Xe, ge, ue, which, shift=10 ** 6), group_id=ge, cat_features=cats or None, thread_count=TH), thread_count=TH)
            ptr = m.predict(Pool(frame(Xt, gt, ut, which), group_id=gt, cat_features=cats or None, thread_count=TH), thread_count=TH)
            fi = dict(zip(m.feature_names_, np.round(m.get_feature_importance(type="PredictionValuesChange"), 2)))
            ap = m.get_all_params()
            r = dict(cats=cats, kw=kw, eval_ndcg=round(ndcg(p, ye, oe, Me), 4), train_ndcg=round(ndcg(ptr, yt, ot, Mt), 4),
                     imp={k: float(v) for k, v in fi.items() if k.endswith("_cat")}, ctr_history_unit=ap.get("ctr_history_unit", "<absent>"),
                     simple_ctr=ap.get("simple_ctr", "<absent>"), counter_calc_method=ap.get("counter_calc_method", "<absent>"))
        except Exception as e:  # noqa: BLE001
            r = dict(cats=cats, kw=kw, error=" ".join(str(e).split())[:300])
        res.append(r); print(json.dumps(r), flush=True)
        if not which: break
json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "verify_ctr_group.json"), "w"), indent=1)
