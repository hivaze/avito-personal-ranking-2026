"""docs_rank: CPU cost per boosting iteration of each CatBoost ranking objective / tree policy / sampling option at the SHAPE of
this problem (synthetic numbers; no project data): feeds with M ~ the exploration train selection (percentiles 54/80/128/214/283,
mean about 148), 177 float features, about 1.7 positives per feed, 12 threads, depth 8, 254 borders (the cb26 settings).

Per-iteration seconds are read from a python callback (CPU only), median over iterations 4..N.
usage: CUDA_VISIBLE_DEVICES="" python bench_cpu.py <n_feeds> <n_iter> [tag-substring ...]   -> out/bench_<n_feeds>.json
"""
import json, os, sys, time
assert os.environ.get("CUDA_VISIBLE_DEVICES", "x") == "", "CPU only"
import numpy as np
import catboost
from catboost import CatBoost, Pool

TH = 12
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "out"); os.makedirs(OUT, exist_ok=True)
n_feeds = int(sys.argv[1]); n_iter = int(sys.argv[2]); only = sys.argv[3:]
t00 = time.time(); log = lambda *a: print(f"[{time.time() - t00:6.0f}s]", *a, flush=True)


def make(nf, seed, d=177):
    r = np.random.default_rng(seed)
    qs = np.array([50, 54, 80, 128, 214, 283, 300]); ps = np.array([0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0])
    M = np.interp(r.uniform(size=nf), ps, qs).round().astype(np.int64)
    n = int(M.sum()); off = np.r_[0, np.cumsum(M)[:-1]]; gid = np.repeat(np.arange(nf), M)
    X = r.standard_normal(size=(n, d), dtype=np.float32)
    w = np.random.default_rng(5).normal(size=d) * (np.arange(d) < 30)
    u = X @ w.astype(np.float32); u = (u / u.std() + 0.5 * X[:, 0] * X[:, 1]).astype(np.float32)
    g = r.gumbel(size=n).astype(np.float32) + 2.0 * u                    # Gumbel-top-k = sampling without replacement
    y = np.zeros(n, np.float32); k = 1 + r.poisson(0.7, nf)
    for f in range(nf):
        s = slice(off[f], off[f] + M[f]); y[off[f] + np.argpartition(-g[s], k[f] - 1)[:k[f]]] = 1.0
    t = (u + 0.3 * r.standard_normal(n).astype(np.float32)).astype(np.float32)
    return X, y, t, gid, M


class Timer:
    def __init__(self): self.t = [time.time()]
    def after_iteration(self, info):
        self.t.append(time.time()); return True


X, y, t, gid, M = make(n_feeds, 0)
log(f"train {n_feeds} feeds, {len(y)} rows x {X.shape[1]} features, mean M {M.mean():.1f}, positives per feed {y.sum() / n_feeds:.2f}")
t0 = time.time(); ptr = Pool(X, label=y, group_id=gid, thread_count=TH); t_pool = time.time() - t0
t_q = float("nan")   # Pool.quantize() has no thread_count argument, so quantisation is left to fit() (12 threads); see 'setup_s'
log(f"Pool build {t_pool:.1f}s")
ptr_soft = Pool(X, label=t, group_id=gid, thread_count=TH)
ptr32 = None

B = dict(depth=8, learning_rate=0.08, l2_leaf_reg=5, thread_count=TH, random_seed=0, verbose=False, allow_writing_files=False)
CFG = [
    ("YetiRank (Classic) = cb26 settings on CPU", "YetiRank", {}, "hard"),
    ("YetiRank:mode=NDCG", "YetiRank:mode=NDCG", {}, "hard"),
    ("YetiRank:mode=NDCG;num_neighbors=3", "YetiRank:mode=NDCG;num_neighbors=3", {}, "hard"),
    ("YetiRank:mode=MRR", "YetiRank:mode=MRR", {}, "hard"),
    ("YetiRank:permutations=30", "YetiRank:permutations=30", {}, "hard"),
    ("LambdaMart", "LambdaMart", {}, "hard"),
    ("LambdaMart:norm=false", "LambdaMart:norm=false", {}, "hard"),
    ("StochasticRank:metric=NDCG", "StochasticRank:metric=NDCG", {}, "hard"),
    ("QuerySoftMax", "QuerySoftMax", {}, "hard"),
    ("QueryRMSE (teacher logit label)", "QueryRMSE", {}, "soft"),
    ("RMSE (teacher logit label)", "RMSE", {}, "soft"),
    ("PairLogit (all pos x neg pairs)", "PairLogit", {}, "hard"),
    ("Logloss", "Logloss", {}, "hard"),
    ("YetiRank depth 6", "YetiRank", dict(depth=6), "hard"),
    ("YetiRank depth 10", "YetiRank", dict(depth=10), "hard"),
    ("YetiRank Depthwise depth 8 min_data_in_leaf 100", "YetiRank", dict(grow_policy="Depthwise", min_data_in_leaf=100), "hard"),
    ("YetiRank Lossguide max_leaves 255 depth 16 min_data_in_leaf 100", "YetiRank", dict(grow_policy="Lossguide", max_leaves=255, depth=16, min_data_in_leaf=100), "hard"),
    ("YetiRank Lossguide max_leaves 63", "YetiRank", dict(grow_policy="Lossguide", max_leaves=63, depth=16, min_data_in_leaf=100), "hard"),
    ("QueryRMSE Lossguide max_leaves 255 (teacher logit)", "QueryRMSE", dict(grow_policy="Lossguide", max_leaves=255, depth=16, min_data_in_leaf=100), "soft"),
    ("YetiRank bootstrap No", "YetiRank", dict(bootstrap_type="No"), "hard"),
    ("YetiRank Bernoulli subsample 0.5 (object)", "YetiRank", dict(bootstrap_type="Bernoulli", subsample=0.5), "hard"),
    ("YetiRank sampling_unit=Group Bernoulli 0.5", "YetiRank", dict(bootstrap_type="Bernoulli", subsample=0.5, sampling_unit="Group"), "hard"),
    ("YetiRank MVS subsample 0.3", "YetiRank", dict(bootstrap_type="MVS", subsample=0.3), "hard"),
    ("YetiRank rsm 0.3", "YetiRank", dict(rsm=0.3), "hard"),
    ("YetiRank boosting_type=Ordered", "YetiRank", dict(boosting_type="Ordered"), "hard"),
    ("YetiRank + langevin", "YetiRank", dict(langevin=True, diffusion_temperature=10000), "hard"),
    ("X1 YetiRank:permutations=3", "YetiRank:permutations=3", {}, "hard"),
    ("X2 QuerySoftMax + Newton leaves", "QuerySoftMax", dict(leaf_estimation_method="Newton"), "hard"),
    ("X3 YetiRank (Classic) on a real-valued label", "YetiRank", {}, "soft"),
    ("X4 LambdaMart on a real-valued label", "LambdaMart:norm=false", {}, "soft"),
]
res = dict(n_feeds=n_feeds, rows=int(len(y)), n_feat=int(X.shape[1]), threads=TH, pool_build_s=t_pool, quantize_s=t_q,
           load=os.getloadavg(), catboost=catboost.__version__, rows_cfg=[])
for name, loss, kw, lab in CFG:
    if only and not any(o in name for o in only):
        continue
    p = dict(B); p.update(kw)
    try:
        m = CatBoost(dict(p, loss_function=loss, iterations=n_iter))
        t0 = time.time(); tm = Timer(); m.fit(ptr if lab == "hard" else ptr_soft, callbacks=[tm]); tot = time.time() - t0
        d = np.diff(tm.t)
        r = dict(name=name, loss=loss, kw=kw, total_s=tot, setup_s=float(tm.t[0] - t0 + d[0]), med_iter_s=float(np.median(d[3:])),
                 mean_iter_s=float(np.mean(d[3:])), n_iter=len(d))
    except Exception as e:  # noqa: BLE001
        r = dict(name=name, loss=loss, kw=kw, error=" ".join(str(e).split())[:300])
    res["rows_cfg"].append(r); log(json.dumps(r))
    json.dump(res, open(os.path.join(OUT, f"bench_{n_feeds}{'_extra' if only else ''}.json"), "w"), indent=1)

# cost of the built-in NDCG eval metric on a val-sized eval set (about 14.4K feeds x 96 = 1.39M rows)
if not only:
    Xe, ye, te, ge, Me = make(9500, 1)
    pe = Pool(Xe, label=ye, group_id=ge, thread_count=TH)
    for mp in (1, 50):
        tm = Timer(); m = CatBoost(dict(B, loss_function="YetiRank", iterations=n_iter, eval_metric="NDCG:type=Base", metric_period=mp))
        t0 = time.time(); m.fit(ptr, eval_set=pe, callbacks=[tm]); d = np.diff(tm.t)
        r = dict(name=f"YetiRank + eval_set {len(ye)} rows, NDCG:type=Base, metric_period={mp}", med_iter_s=float(np.median(d[3:])),
                 mean_iter_s=float(np.mean(d[3:])), total_s=time.time() - t0)
        res["rows_cfg"].append(r); log(json.dumps(r))
    t0 = time.time(); pr = m.predict(pe, thread_count=TH); res["predict_eval_s"] = time.time() - t0
    log(f"predict {len(ye)} rows with {m.tree_count_} trees: {res['predict_eval_s']:.2f}s")
    json.dump(res, open(os.path.join(OUT, f"bench_{n_feeds}.json"), "w"), indent=1)
log("DONE")
