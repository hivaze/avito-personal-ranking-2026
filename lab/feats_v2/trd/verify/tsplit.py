import sys, json, numpy as np
sys.path.insert(0, "<repo>/src")
from common import *
import eval_family as ef
import lightgbm as lgb
fam = sys.argv[1].split(","); seeds = int(sys.argv[2]); thr = int(sys.argv[3])
f, M, off = feed_meta("val")
y = np.load(FLAT / "val/cand_target.npy").astype(np.float32)
fidx = np.load(FLAT / "val/cand_fidx.npy")
ts = f["timestamp"].to_numpy().astype(np.int64)
THU0 = 1790197200  # 2026-09-24 00:00 MSK
day = (ts >= THU0).astype(int)   # 0 = Tue (+1 spillover feed), 1 = Thu (+2 spillover feeds)
print("feeds per day", np.bincount(day), flush=True)
base = ef.base_features()
Xb = load_feats("val", base)
Xf = load_feats("val", fam)
rng = np.random.default_rng(123); Xr = rng.random((len(y), 4)).astype(np.float32)
cfgs = {"base": (Xb, base), "fam": (np.hstack([Xb, Xf]), base + fam), "rand4": (np.hstack([Xb, Xr]), base + [f"zr{i}" for i in range(4)])}
rowday = day[fidx]
res = {}
for d_tr in (0, 1):
    trf = np.flatnonzero(day == d_tr); vaf = np.flatnonzero(day != d_tr)
    tr = rowday == d_tr
    o = vaf
    for seed in range(seeds):
        preds = {}
        for k, (X, names) in cfgs.items():
            p = dict(ef.PARAMS, num_threads=thr, seed=seed)
            cat = ["category_id"] if "category_id" in names else []
            b = lgb.train(p, lgb.Dataset(X[tr], y[tr], group=M[trf], feature_name=names, categorical_feature=cat), ef.ROUNDS)
            s = np.full(len(y), -1e9); s[~tr] = b.predict(X[~tr])
            preds[k] = ndcg_per_feed(s, y, off, M)[vaf]
        for k in ("fam", "rand4"):
            res.setdefault((d_tr, k), []).append(preds[k] - preds["base"])
        print("train_day", d_tr, "seed", seed, "base", round(preds["base"].mean(), 5), {k: round((preds[k] - preds["base"]).mean(), 5) for k in ("fam", "rand4")}, flush=True)
out = {}
for (d_tr, k), L in res.items():
    D = np.array(L); d = D.mean(0)
    out[f"train{'TueThu'[3*d_tr:3*d_tr+3]}_{k}"] = dict(delta=float(d.mean()), se=float(d.std() / np.sqrt(len(d))), per_seed=[float(x.mean()) for x in D])
print(json.dumps(out, indent=1))
