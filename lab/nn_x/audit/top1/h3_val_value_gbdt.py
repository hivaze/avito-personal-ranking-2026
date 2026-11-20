"""H3 datum: how much are the 14,423 in-distribution (val) labelled feeds worth, at GBDT level, compared with
150K production-period train feeds, and does JOINT training with val up-weighted beat the fine-tune-style use?

Harness = the TRUNC replication proxy (nn_x/insights/review/r2_trunc_replicate.py): 53 leak-free GBDT features,
LightGBM lambdarank (63 leaves, lr 0.05, l2 10, bagging/feature 0.8), 200 rounds, 12 threads. Read-only on caches.

Arms (all read on val, 14,423 feeds; val folds = 5-fold grouped by user, user 0 feeds are singletons):
  A  train-only: U2 150K train feeds (143,586 production + 6,414 exploration-like) -> val.
  B  val-only OOF: 4/5 of val feeds -> the held-out 1/5.
  C  joint OOF: U2 + 4/5 of val with row weight w (w in 1, 10) -> held-out 1/5.
  D  'fine-tune' OOF: boosting FROM arm A's val prediction (init_score) on 4/5 of val, 100 rounds, lr 0.02 -> 1/5.
Readouts: val NDCG per arm, paired per-feed delta vs A with SE, slices M<50 / M>=50 / Tue / Thu.
Outputs: out/h3_results.json (incremental), out/h3_perfeed_<arm>.npy, log to stdout.
"""
import json, sys, time
from pathlib import Path
import numpy as np, polars as pl, lightgbm as lgb

ROOT = Path("<repo>"); sys.path.insert(0, str(ROOT / "src"))
from metrics import ndcg_per_feed  # noqa: E402
G = ROOT / "nn_x/insights/data_usage/out/gb"; RV = ROOT / "nn_x/insights/review/out/gb"
OUT = ROOT / "nn_x/audit/top1/out"; OUT.mkdir(parents=True, exist_ok=True)
FEATS = json.load(open(G / "feats.json")); TH = 12
PARAMS = dict(objective="lambdarank", lambdarank_truncation_level=1000, label_gain=[0, 1], learning_rate=0.05,
              num_leaves=63, min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
              lambda_l2=10.0, num_threads=TH, verbose=-1, seed=11, max_bin=255)
ROUNDS = 200
arms = sys.argv[1:] or ["A", "B", "D", "C10", "C1"]

Xv = np.load(G / "val_X.npy"); mv = dict(np.load(G / "val_meta.npz"))
Mv = mv["M"].astype(np.int64); yv = mv["y"].astype(np.float32); uv = mv["user_id"].astype(np.int64)
offv = np.concatenate([[0], np.cumsum(Mv)[:-1]]); nv = len(Mv); fidv = np.repeat(np.arange(nv), Mv)
feeds = pl.read_parquet(ROOT / "data/flat/val/feeds.parquet")
ts = feeds["timestamp"].to_numpy().astype(np.int64)
day = ((ts + 3 * 3600) // 86400).astype(np.int64); day = day - day.min()   # 0 = Tue Sep 22 MSK, 2 = Thu Sep 24
# user-grouped folds, user 0 = singleton groups
grp = np.where(uv > 0, uv, 10_000_000 + np.arange(nv))
fold = ((grp.astype(np.uint64) * np.uint64(2654435761)) % np.uint64(2**32) % np.uint64(5)).astype(np.int64)
print("fold sizes", np.bincount(fold), flush=True)

res_path = OUT / "h3_results.json"
res = json.load(open(res_path)) if res_path.exists() else {}

def slices(nd):
    return dict(all=float(nd.mean()), M50lo=float(nd[Mv < 50].mean()), M50hi=float(nd[Mv >= 50].mean()),
                tue=float(nd[day == 0].mean()), thu=float(nd[day == 2].mean()))

def paired(a, b):
    d = a - b
    o = {}
    for name, msk in [("all", np.ones(nv, bool)), ("M50lo", Mv < 50), ("M50hi", Mv >= 50), ("tue", day == 0), ("thu", day == 2)]:
        x = d[msk]; o[name] = [float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x)))]
    return o

def save(arm, nd, extra=None):
    np.save(OUT / f"h3_perfeed_{arm}.npy", nd.astype(np.float32))
    res[arm] = dict(ndcg=slices(nd), **(extra or {}))
    if "A" in res and arm != "A":
        res[arm]["delta_vs_A"] = paired(nd, np.load(OUT / "h3_perfeed_A.npy"))
    json.dump(res, open(res_path, "w"), indent=1)
    print(arm, json.dumps(res[arm]), flush=True)

def rows_of(fsel):
    return np.concatenate([np.arange(offv[f], offv[f] + Mv[f]) for f in fsel])

t0 = time.time()
if "A" in arms:
    X = np.load(RV / "U2_X.npy"); m = dict(np.load(RV / "U2_meta.npz"))
    b = lgb.train(PARAMS, lgb.Dataset(X, m["y"], group=m["M"], feature_name=FEATS), ROUNDS)
    pA = b.predict(Xv, num_threads=TH); np.save(OUT / "h3_pred_A.npy", pA.astype(np.float32))
    save("A", ndcg_per_feed(pA, yv, offv, Mv), dict(secs=time.time() - t0, n_train_feeds=int(len(m["M"]))))
    del X, m, b

if "B" in arms:
    t0 = time.time(); pB = np.zeros(len(yv))
    for k in range(5):
        tr = np.where(fold != k)[0]; te = np.where(fold == k)[0]
        rtr = rows_of(tr); rte = rows_of(te)
        b = lgb.train(PARAMS, lgb.Dataset(Xv[rtr], yv[rtr], group=Mv[tr], feature_name=FEATS), ROUNDS)
        pB[rte] = b.predict(Xv[rte], num_threads=TH)
        print(f"  B fold {k} done {time.time()-t0:.0f}s", flush=True)
    save("B", ndcg_per_feed(pB, yv, offv, Mv), dict(secs=time.time() - t0, n_train_feeds="~11.5K val feeds per fold"))

if "D" in arms:
    t0 = time.time(); pD = np.zeros(len(yv)); pA = np.load(OUT / "h3_pred_A.npy").astype(np.float64)
    # within-feed standardise A's scores so the init_score scale is comparable to a lambdarank margin (scale 4 as in audits)
    mu = np.bincount(fidv, weights=pA, minlength=nv) / Mv
    sd = np.sqrt(np.bincount(fidv, weights=(pA - mu[fidv]) ** 2, minlength=nv) / Mv); sd[sd < 1e-9] = 1
    zA = 4.0 * (pA - mu[fidv]) / sd[fidv]
    P = dict(PARAMS, learning_rate=0.02, num_leaves=31, min_data_in_leaf=200)
    for k in range(5):
        tr = np.where(fold != k)[0]; te = np.where(fold == k)[0]
        rtr = rows_of(tr); rte = rows_of(te)
        b = lgb.train(P, lgb.Dataset(Xv[rtr], yv[rtr], group=Mv[tr], init_score=zA[rtr], feature_name=FEATS), 100)
        pD[rte] = zA[rte] + b.predict(Xv[rte], num_threads=TH)
        print(f"  D fold {k} done {time.time()-t0:.0f}s", flush=True)
    save("D", ndcg_per_feed(pD, yv, offv, Mv), dict(secs=time.time() - t0, note="boost from A (init_score 4*z), 100 rounds lr 0.02"))

for arm in [a for a in arms if a.startswith("C")]:
    w = float(arm[1:]); t0 = time.time(); pC = np.zeros(len(yv))
    X = np.load(RV / "U2_X.npy"); m = dict(np.load(RV / "U2_meta.npz"))
    for k in range(5):
        tr = np.where(fold != k)[0]; te = np.where(fold == k)[0]
        rtr = rows_of(tr); rte = rows_of(te)
        Xj = np.concatenate([X, Xv[rtr]]); yj = np.concatenate([m["y"], yv[rtr]])
        gj = np.concatenate([m["M"], Mv[tr]]); wj = np.concatenate([np.ones(len(m["y"]), np.float32), np.full(len(rtr), w, np.float32)])
        b = lgb.train(PARAMS, lgb.Dataset(Xj, yj, group=gj, weight=wj, feature_name=FEATS), ROUNDS)
        pC[rte] = b.predict(Xv[rte], num_threads=TH)
        del Xj, yj, b
        print(f"  {arm} fold {k} done {time.time()-t0:.0f}s", flush=True)
    save(arm, ndcg_per_feed(pC, yv, offv, Mv), dict(secs=time.time() - t0, val_row_weight=w))
    del X, m
print("done", flush=True)
