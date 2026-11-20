"""docs_rank: DIRECTION check of the ranking-training levers on a SYNTHETIC analogue of the problem (no project data).

Feeds of 50..300 candidates (mean about 148), about 1.7 clicks per feed, full-list NDCG with pessimistic ties. The click logit
depends on visible features X (the student sees them) and on hidden features Z (only the 'teacher' sees them): the student is
representation-limited by construction, as the tree model of record is. Teacher score = the true click logit + small noise.

Every arm: 3 seeds, the same train / eval feeds, 200 trees, depth 6, lr 0.1, 12 threads, CPU. Read-out: mean eval NDCG over
seeds, and the paired per-feed difference to the reference arm (seed-mean per feed) with its SE over the eval feeds.
This is a mechanism check, NOT evidence about the real data (on the real data the objective choice measured +-0.001).

usage: CUDA_VISIBLE_DEVICES="" python synth_levers.py                 -> out/synth_levers.json (arms H*, S*, T*)
       CUDA_VISIBLE_DEVICES="" python synth_levers.py T              -> out/synth_levers_filter_T.json (name prefixes; the first
                                                                       selected arm is the reference of the deltas)
"""
import json, os, sys, time
assert os.environ.get("CUDA_VISIBLE_DEVICES", "x") == "", "CPU only"
import numpy as np
from catboost import CatBoost, Pool

TH = 12
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "out"); os.makedirs(OUT, exist_ok=True)
t00 = time.time(); log = lambda *a: print(f"[{time.time() - t00:6.0f}s]", *a, flush=True)
D_VIS, D_HID = 20, 6


def make(nf, seed):
    r = np.random.default_rng(seed)
    qs = np.array([50, 54, 80, 128, 214, 283, 300]); ps = np.array([0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0])
    M = np.interp(r.uniform(size=nf), ps, qs).round().astype(np.int64)
    n = int(M.sum()); off = np.r_[0, np.cumsum(M)[:-1]]; gid = np.repeat(np.arange(nf), M)
    X = r.standard_normal(size=(n, D_VIS), dtype=np.float32); Z = r.standard_normal(size=(n, D_HID), dtype=np.float32)
    rt = np.random.default_rng(777)
    wv = rt.normal(size=D_VIS) * (np.arange(D_VIS) < 10); wh = rt.normal(size=D_HID)
    vis = X @ wv.astype(np.float32) * 0.3 + 0.5 * X[:, 0] * X[:, 1] + 0.5 * np.abs(X[:, 2]) - 0.4 * (X[:, 3] > 1) * X[:, 4]
    hid = Z @ wh.astype(np.float32) * 0.3 + 0.4 * Z[:, 0] * X[:, 5]
    logit = 1.15 * (vis / vis.std()) + 0.8 * (hid / hid.std())             # true click logit (within-feed softmax scale)
    g = r.gumbel(size=n).astype(np.float32) + logit
    y = np.zeros(n, np.float32); k = 1 + r.poisson(0.7, nf)
    for f in range(nf):
        s = slice(off[f], off[f] + M[f]); y[off[f] + np.argpartition(-g[s], k[f] - 1)[:k[f]]] = 1.0
    teacher = (logit + 0.15 * r.standard_normal(n)).astype(np.float32)
    return dict(X=X, y=y, t=teacher, gid=gid, M=M, off=off, n=n, logit=logit.astype(np.float32))


def ndcg_feed(score, y, off, M):
    out = np.zeros(len(M))
    for g in range(len(M)):
        s = score[off[g]:off[g] + M[g]]; t = y[off[g]:off[g] + M[g]]
        o = np.lexsort((t, -s)); disc = 1.0 / np.log2(np.arange(M[g]) + 2.0)
        out[g] = (t[o] * disc).sum() / (np.sort(t)[::-1] * disc).sum()
    return out


def soft(t, off, M, T=1.0):
    q = np.zeros(len(t), np.float64)
    for o, m in zip(off, M):
        e = np.exp((t[o:o + m] - t[o:o + m].max()) / T); q[o:o + m] = e / e.sum()
    return q.astype(np.float32)


TR = make(3000, 1); EV = make(3000, 2)
npos = np.repeat(np.add.reduceat(TR["y"], TR["off"]), TR["M"])
LAB = dict(y=TR["y"], logit=TR["t"], q1=soft(TR["t"], TR["off"], TR["M"], 1.0))
LAB["mix"] = (0.5 * TR["y"] / npos + 0.5 * LAB["q1"]).astype(np.float32)
LAB["logit_plus_y"] = (TR["t"] + 1.0 * TR["y"]).astype(np.float32)
POOLS = {k: Pool(TR["X"], label=v, group_id=TR["gid"], thread_count=TH) for k, v in LAB.items()}
PEV = Pool(EV["X"], group_id=EV["gid"], thread_count=TH)
log(f"train {len(TR['M'])} feeds / {TR['n']} rows, eval {len(EV['M'])} feeds / {EV['n']} rows; positives per feed {TR['y'].mean() * TR['M'].mean():.2f}")
ref_levels = dict(random=float(ndcg_feed(np.random.default_rng(0).normal(size=EV["n"]), EV["y"], EV["off"], EV["M"]).mean()),
                  teacher=float(ndcg_feed(EV["t"].astype(np.float64), EV["y"], EV["off"], EV["M"]).mean()),
                  true_logit=float(ndcg_feed(EV["logit"].astype(np.float64), EV["y"], EV["off"], EV["M"]).mean()))
log("levels", ref_levels)

B = dict(iterations=200, learning_rate=0.1, depth=6, thread_count=TH, verbose=False, allow_writing_files=False)
NEWTON = dict(leaf_estimation_method="Newton")
ARMS = [  # name, loss, label key, extra params
    ("H0 YetiRank (Classic, decay 0.85) [reference]", "YetiRank", "y", {}),
    ("H1 YetiRank:decay=0.95", "YetiRank:decay=0.95", "y", {}),
    ("H2 YetiRank:decay=0.99", "YetiRank:decay=0.99", "y", {}),
    ("H3 YetiRank:mode=NDCG", "YetiRank:mode=NDCG", "y", {}),
    ("H4 YetiRank:mode=NDCG;num_neighbors=3", "YetiRank:mode=NDCG;num_neighbors=3", "y", {}),
    ("H5 YetiRank:mode=MRR", "YetiRank:mode=MRR", "y", {}),
    ("H6 YetiRank:permutations=30", "YetiRank:permutations=30", "y", {}),
    ("H7 LambdaMart", "LambdaMart", "y", {}),
    ("H8 LambdaMart:norm=false", "LambdaMart:norm=false", "y", {}),
    ("H9 QuerySoftMax (Gradient leaves = default)", "QuerySoftMax", "y", {}),
    ("H10 QuerySoftMax + Newton leaves", "QuerySoftMax", "y", NEWTON),
    ("H11 Logloss (pointwise)", "Logloss", "y", {}),
    ("H12 YetiRank Lossguide max_leaves 64", "YetiRank", "y", dict(grow_policy="Lossguide", max_leaves=64, depth=16, min_data_in_leaf=50)),
    ("H13 YetiRank l2_leaf_reg=30", "YetiRank", "y", dict(l2_leaf_reg=30)),
    ("H14 YetiRank rsm=0.5", "YetiRank", "y", dict(rsm=0.5)),
    ("H15 YetiRank sampling_unit=Group Bernoulli 0.5", "YetiRank", "y", dict(sampling_unit="Group", bootstrap_type="Bernoulli", subsample=0.5)),
    ("S1 QueryRMSE <- teacher logit", "QueryRMSE", "logit", {}),
    ("S2 RMSE <- teacher logit", "RMSE", "logit", {}),
    ("S3 QuerySoftMax + Newton <- softmax(teacher)", "QuerySoftMax", "q1", NEWTON),
    ("S4 QuerySoftMax + Newton <- 0.5 y/npos + 0.5 softmax(teacher)", "QuerySoftMax", "mix", NEWTON),
    ("S5 YetiRank <- teacher logit", "YetiRank", "logit", {}),
    ("S6 YetiRank:mode=NDCG <- softmax(teacher)", "YetiRank:mode=NDCG", "q1", {}),
    ("S7 LambdaMart:norm=false <- softmax(teacher)", "LambdaMart:norm=false", "q1", {}),
    ("S8 QueryRMSE <- teacher logit + 1.0 y", "QueryRMSE", "logit_plus_y", {}),
    ("S9 QueryRMSE <- teacher (150 trees) then init_model -> YetiRank on clicks (50 trees, lr 0.03)", "2stage", "logit", {}),
    # second round (run with the prefix filter: "T"); the first selected arm is the reference of a filtered run
    ("T0 QueryRMSE <- teacher logit, symmetric depth 6 [reference of round 2]", "QueryRMSE", "logit", {}),
    ("T1 QueryRMSE <- teacher logit, symmetric depth 8", "QueryRMSE", "logit", dict(depth=8)),
    ("T2 QueryRMSE <- teacher logit, symmetric depth 10", "QueryRMSE", "logit", dict(depth=10)),
    ("T3 QueryRMSE <- teacher logit, Lossguide max_leaves 64", "QueryRMSE", "logit", dict(grow_policy="Lossguide", max_leaves=64, depth=16, min_data_in_leaf=50)),
    ("T4 QueryRMSE <- teacher logit, Lossguide max_leaves 255", "QueryRMSE", "logit", dict(grow_policy="Lossguide", max_leaves=255, depth=16, min_data_in_leaf=50)),
    ("T5 QueryRMSE <- teacher logit, Depthwise depth 8 min_data_in_leaf 50", "QueryRMSE", "logit", dict(grow_policy="Depthwise", depth=8, min_data_in_leaf=50)),
    ("T6 QueryRMSE <- teacher logit, rsm 0.5", "QueryRMSE", "logit", dict(rsm=0.5)),
    ("T7 QueryRMSE <- teacher logit, l2_leaf_reg 30", "QueryRMSE", "logit", dict(l2_leaf_reg=30)),
    ("T8 QuerySoftMax + Newton <- 0.5 y/npos + 0.5 softmax(teacher), beta=2", "QuerySoftMax:beta=2", "mix", NEWTON),
    ("T9 YetiRank l2_leaf_reg=3 (hard clicks)", "YetiRank", "y", dict(l2_leaf_reg=3)),
]
only = sys.argv[1:]
res = dict(levels=ref_levels, n_eval_feeds=int(len(EV["M"])), n_train_feeds=int(len(TR["M"])), arms=[])
PF = {}
for name, loss, lab, kw in ARMS:
    if only and not any(name.startswith(o) for o in only):
        continue
    per_seed = []; t0 = time.time(); err = None
    for seed in (0, 1, 2):
        try:
            if loss == "2stage":
                m1 = CatBoost(dict(B, loss_function="QueryRMSE", iterations=150, random_seed=seed)); m1.fit(POOLS["logit"])
                m = CatBoost(dict(B, loss_function="YetiRank", iterations=50, learning_rate=0.03, random_seed=seed)); m.fit(POOLS["y"], init_model=m1)
            else:
                p = dict(B); p.update(kw); m = CatBoost(dict(p, loss_function=loss, random_seed=seed)); m.fit(POOLS[lab])
            pr = np.asarray(m.predict(PEV, thread_count=TH), np.float64)
            per_seed.append(ndcg_feed(pr, EV["y"], EV["off"], EV["M"]))
        except Exception as e:  # noqa: BLE001
            err = " ".join(str(e).split())[:300]; break
    if err:
        res["arms"].append(dict(name=name, error=err)); log(name, "ERROR", err); continue
    A = np.stack(per_seed); PF[name] = A.mean(0)
    r = dict(name=name, loss=loss, label=lab, kw=kw, ndcg_seed=[float(x) for x in A.mean(1)], ndcg=float(A.mean()), seed_sd=float(A.mean(1).std(ddof=1)),
             sec=time.time() - t0)
    refn = ARMS[0][0] if not only else next(iter(PF))
    r["ref"] = refn
    if refn in PF:
        d = PF[name] - PF[refn]; r["delta_vs_ref"] = float(d.mean()); r["se"] = float(d.std(ddof=1) / np.sqrt(len(d)))
        r["z"] = float(r["delta_vs_ref"] / r["se"]) if r["se"] > 0 else 0.0
    res["arms"].append(r)
    log(f"{name}: NDCG {r['ndcg']:.4f} (seeds {np.round(r['ndcg_seed'], 4).tolist()}) delta {r.get('delta_vs_ref', 0):+.4f} +- {r.get('se', 0):.4f}  {r['sec']:.0f}s")
    json.dump(res, open(os.path.join(OUT, f"synth_levers_filter_{only[0]}.json" if only else "synth_levers.json"), "w"), indent=1)
log("DONE")
