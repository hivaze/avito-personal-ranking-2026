"""Review R2: independent replication of the data_usage GBDT truncation-offset proxy on a FRESH 150K-feed train sample
(disjoint from data_usage's U, seed 2026), new LightGBM seeds. Reads data_usage val/HO caches (read-only).
Arms: base, tprod1 (offset a*L[group, rkend 0-2] on production rows only, L estimated on the sample itself).
Also reports the delta at 100..400 rounds (no post-hoc round choice) and split by M<50 / M>=50 and Tue/Thu."""
import sys, json, time
import numpy as np, polars as pl, lightgbm as lgb
from pathlib import Path
ROOT = Path("<repo>"); sys.path.insert(0, str(ROOT / "src"))
from metrics import ndcg_per_feed
DU = ROOT / "nn_x/insights/data_usage/out"; G = DU / "gb"; OUT = ROOT / "nn_x/insights/review/out/gb"
FEATS = json.load(open(G / "feats.json"))
mode = sys.argv[1]
if mode == "build":
    t = pl.read_parquet(DU / "train_feeds.parquet")
    M = t["M"].to_numpy().astype(np.int64); cls = t["cls"].to_numpy()
    off = np.concatenate([[0], np.cumsum(M)[:-1]])
    U_old = np.load(G / "U_meta.npz")["fidx"]
    rng = np.random.default_rng(2026)
    cand = np.setdiff1d(np.where(cls != 2)[0], U_old)
    U = np.sort(rng.choice(cand, 150000, replace=False))
    rows = np.concatenate([np.arange(off[f], off[f] + M[f]) for f in U])
    g = M[U]; fid = np.repeat(np.arange(len(U)), g); goff = np.concatenate([[0], np.cumsum(g)[:-1]])
    pos = np.load(ROOT / "data/flat/train/cand_pos.npy", mmap_mode="r")[rows].astype(np.int64)
    o = np.lexsort((pos, fid)); rank = np.empty(len(rows), np.int64); rank[o] = np.arange(len(rows)) - goff[fid[o]]
    rkend = g[fid] - 1 - rank
    Xm = np.empty((len(rows), len(FEATS)), np.float32)
    for j, fn in enumerate(FEATS):
        Xm[:, j] = np.load(ROOT / f"data/feats/train/{fn}.npy", mmap_mode="r")[rows]
    np.save(OUT / "U2_X.npy", Xm)
    np.savez(OUT / "U2_meta.npz", fidx=U, M=g, y=np.load(ROOT / "data/flat/train/cand_target.npy", mmap_mode="r")[rows].astype(np.float32),
             rank=rank.astype(np.int16), rkend=rkend.astype(np.int16), cls=cls[U])
    print("built", Xm.shape, "overlap with U:", len(np.intersect1d(U, U_old)), "share cls1", np.mean(cls[U] == 1), flush=True)
    sys.exit(0)

arm, seed, rounds = sys.argv[2], int(sys.argv[3]), 400
X = np.load(OUT / "U2_X.npy"); m = dict(np.load(OUT / "U2_meta.npz"))
nf = len(m["M"]); fid = np.repeat(np.arange(nf), m["M"])
init = np.zeros(len(fid))
if arm.startswith("tprod") or arm.startswith("trunc"):
    a = float(arm[5:])
    Mr = m["M"][fid]; c = m["cls"][fid]; y = m["y"]; re = m["rkend"].astype(np.int64)
    grp = np.where(c >= 1, 2, np.where(Mr < 50, 0, 1))
    L = np.zeros((3, 3))
    for gg in range(3):
        base = y[(grp == gg) & (re >= 3)].mean()
        for k in range(3):
            L[gg, k] = np.log(max(y[(grp == gg) & (re == k)].mean(), 1e-4) / base)
    print("lifts", np.round(L, 3).tolist(), flush=True)
    sel = (grp <= 1) if arm.startswith("tprod") else np.ones(len(grp), bool)
    init += np.where((re <= 2) & sel, a * L[grp, np.minimum(re, 2)], 0)
Xv = np.load(G / "val_X.npy"); mv = dict(np.load(G / "val_meta.npz"))
Xh = np.load(G / "HO_X.npy"); mh = dict(np.load(G / "HO_meta.npz"))
Mv = mv["M"].astype(np.int64); offv = np.concatenate([[0], np.cumsum(Mv)[:-1]])
Mh = mh["M"].astype(np.int64); offh = np.concatenate([[0], np.cumsum(Mh)[:-1]])
params = dict(objective="lambdarank", lambdarank_truncation_level=1000, label_gain=[0, 1], learning_rate=0.05,
              num_leaves=63, min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
              lambda_l2=10.0, num_threads=12, verbose=-1, seed=seed, max_bin=255)
t0 = time.time()
ds = lgb.Dataset(X, m["y"], group=m["M"], init_score=init if np.any(init != 0) else None, feature_name=FEATS)
b = lgb.train(params, ds, rounds)
out = {}
for k in (100, 200, 300, 400):
    nv = ndcg_per_feed(b.predict(Xv, num_iteration=k, num_threads=12), mv["y"], offv, Mv)
    nh = ndcg_per_feed(b.predict(Xh, num_iteration=k, num_threads=12), mh["y"], offh, Mh)
    out[f"val{k}"] = nv.astype(np.float32); out[f"ho{k}"] = nh.astype(np.float32)
np.savez(OUT / f"res_{arm}_s{seed}.npz", **out)
print(arm, seed, {k: round(float(v.mean()), 5) for k, v in out.items()}, f"{time.time()-t0:.0f}s", flush=True)
