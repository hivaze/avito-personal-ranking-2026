"""critique2 N1: is the h3 C1 datum (+0.00206 +- 0.00101, joint GBDT with val rows at weight 1 vs train-only A) robust to
model seed? The per-feed paired SE treats both models as fixed; a GBDT with bagging 0.8 / feature_fraction 0.8 has seed noise
the h3 script never measured (every arm used seed 11). Re-runs arm A with seeds 12, 13 and arm C1 with seed 12, same folds,
same harness (copied from nn_x/audit/top1/h3_val_value_gbdt.py). Also reads the stored C1 vs C10 paired difference.
CPU only, 12 threads. Writes out/c2_01_h3_seed.json and prints paired deltas."""
import os
for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "POLARS_MAX_THREADS"]:
    os.environ[k] = "12"
import json, sys, time
from pathlib import Path
import numpy as np, polars as pl, lightgbm as lgb
ROOT = Path("<repo>"); sys.path.insert(0, str(ROOT / "src"))
from metrics import ndcg_per_feed
G = ROOT / "nn_x/insights/data_usage/out/gb"; RV = ROOT / "nn_x/insights/review/out/gb"
H3 = ROOT / "nn_x/audit/top1/out"; OUT = ROOT / "nn_x/audit/critique2/out"
FEATS = json.load(open(G / "feats.json")); TH = 12
PARAMS = dict(objective="lambdarank", lambdarank_truncation_level=1000, label_gain=[0, 1], learning_rate=0.05,
              num_leaves=63, min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
              lambda_l2=10.0, num_threads=TH, verbose=-1, seed=11, max_bin=255)
ROUNDS = 200
Xv = np.load(G / "val_X.npy"); mv = dict(np.load(G / "val_meta.npz"))
Mv = mv["M"].astype(np.int64); yv = mv["y"].astype(np.float32); uv = mv["user_id"].astype(np.int64)
offv = np.concatenate([[0], np.cumsum(Mv)[:-1]]); nv = len(Mv)
feeds = pl.read_parquet(ROOT / "data/flat/val/feeds.parquet")
ts = feeds["timestamp"].to_numpy().astype(np.int64)
day = ((ts + 3 * 3600) // 86400).astype(np.int64); day = day - day.min()
grp = np.where(uv > 0, uv, 10_000_000 + np.arange(nv))
fold = ((grp.astype(np.uint64) * np.uint64(2654435761)) % np.uint64(2**32) % np.uint64(5)).astype(np.int64)
SL = [("all", np.ones(nv, bool)), ("M<50", Mv < 50), ("M>=50", Mv >= 50), ("Tue", day == 0), ("Thu", day == 2)]
def paired(a, b):
    d = a - b
    return {n: [float(d[m].mean()), float(d[m].std(ddof=1) / np.sqrt(m.sum()))] for n, m in SL}
def fmt(p):
    return "  ".join(f"{n} {v[0]:+.5f}±{v[1]:.5f}" for n, v in p.items())
def rows_of(fsel):
    return np.concatenate([np.arange(offv[f], offv[f] + Mv[f]) for f in fsel])
res = {}
A11 = np.load(H3 / "h3_perfeed_A.npy").astype(np.float64)
C1_11 = np.load(H3 / "h3_perfeed_C1.npy").astype(np.float64)
C10_11 = np.load(H3 / "h3_perfeed_C10.npy").astype(np.float64)
B11 = np.load(H3 / "h3_perfeed_B.npy").astype(np.float64)
print("stored: C1(s11) - C10(s11):", fmt(paired(C1_11, C10_11)), flush=True)
print("stored: C1(s11) - A(s11):  ", fmt(paired(C1_11, A11)), flush=True)
print("stored: C10(s11) - A(s11): ", fmt(paired(C10_11, A11)), flush=True)
res["stored_C1_minus_C10"] = paired(C1_11, C10_11)
X = np.load(RV / "U2_X.npy"); m = dict(np.load(RV / "U2_meta.npz"))
per = {"A11": A11, "C1_11": C1_11}
t0 = time.time()
for s in [12, 13]:
    b = lgb.train(dict(PARAMS, seed=s), lgb.Dataset(X, m["y"], group=m["M"], feature_name=FEATS), ROUNDS)
    pA = b.predict(Xv, num_threads=TH); nd = ndcg_per_feed(pA, yv, offv, Mv).astype(np.float64)
    per[f"A{s}"] = nd; np.save(OUT / f"c2_01_perfeed_A{s}.npy", nd.astype(np.float32))
    print(f"A seed {s}: val {nd.mean():.5f}; A{s} - A11:", fmt(paired(nd, A11)), f"[{time.time()-t0:.0f}s]", flush=True)
    res[f"A{s}_minus_A11"] = paired(nd, A11); res[f"A{s}_val"] = float(nd.mean())
    del b
print("A12 - A13:", fmt(paired(per["A12"], per["A13"])), flush=True)
res["A12_minus_A13"] = paired(per["A12"], per["A13"])
# C1 with seed 12
for s in [12]:
    pC = np.zeros(len(yv))
    for k in range(5):
        tr = np.where(fold != k)[0]; te = np.where(fold == k)[0]
        rtr = rows_of(tr); rte = rows_of(te)
        Xj = np.concatenate([X, Xv[rtr]]); yj = np.concatenate([m["y"], yv[rtr]])
        gj = np.concatenate([m["M"], Mv[tr]]); wj = np.ones(len(yj), np.float32)
        b = lgb.train(dict(PARAMS, seed=s), lgb.Dataset(Xj, yj, group=gj, weight=wj, feature_name=FEATS), ROUNDS)
        pC[rte] = b.predict(Xv[rte], num_threads=TH)
        del Xj, yj, b
        print(f"  C1 seed {s} fold {k} done {time.time()-t0:.0f}s", flush=True)
    nd = ndcg_per_feed(pC, yv, offv, Mv).astype(np.float64)
    per[f"C1_{s}"] = nd; np.save(OUT / f"c2_01_perfeed_C1_{s}.npy", nd.astype(np.float32))
    res[f"C1_{s}_val"] = float(nd.mean())
    for ref in ["A11", "A12", "A13", "C1_11"]:
        p = paired(nd, per[ref]); res[f"C1_{s}_minus_{ref}"] = p
        print(f"C1 seed {s} ({nd.mean():.5f}) - {ref} ({per[ref].mean():.5f}):", fmt(p), flush=True)
    # seed-averaged contrast: mean over seeds of C1 minus mean over seeds of A (2 seeds each)
    Cbar = 0.5 * (nd + per["C1_11"]); Abar = (per["A11"] + per["A12"] + per["A13"]) / 3
    p = paired(Cbar, Abar); res["C1_2seedmean_minus_A_3seedmean"] = p
    print("mean(C1 s11,s12) - mean(A s11,s12,s13):", fmt(p), flush=True)
json.dump(res, open(OUT / "c2_01_h3_seed.json", "w"), indent=1)
print("done", flush=True)
