"""(c) reverse direction: residual GBDT trained on the 30K in-period HOLDOUT (init = holdout train-only G3 pool z; rounds
chosen on an inner 20%-of-users holdout split), applied to ALL val (init = val final G3 z, and separately the val
train-only pool z). Disjoint populations -> no CV needed on val. Control = level 0 (g only), same seeds.
Usage: python gbdt_h2v.py <level> <seed> [lr] [leaves] [minleaf]
"""
import sys, time
import numpy as np
import lightgbm as lgb
sys.path.insert(0, "/tmp/scratchpad/dpa/nonlinear")
from nlc import load, ndcg, D, wmean_se, pmix_w  # noqa
from featmat import build
from metrics import ndcg_per_feed

t0 = time.time()
level, seed = int(sys.argv[1]), int(sys.argv[2])
lr = float(sys.argv[3]) if len(sys.argv) > 3 else 0.03
leaves = int(sys.argv[4]) if len(sys.argv) > 4 else 15
minleaf = int(sys.argv[5]) if len(sys.argv) > 5 else 500
tag = f"H2V_L{level}_s{seed}_lr{lr}_nl{leaves}_ml{minleaf}"
dv = load("val"); dh = load("hold")
Xv, names, cats = build(dv, level); Xh, _, _ = build(dh, level)
ROUNDS = list(range(10, 801, 10))


def make_obj(M, off):
    fid = np.repeat(np.arange(len(M)), M)

    def obj(preds, ds):
        y = ds.get_label()
        mx = np.maximum.reduceat(preds, off)
        e = np.exp(preds - mx[fid]); Z = np.bincount(fid, e, len(M)); p = e / Z[fid]
        npos = np.bincount(fid, y, len(M)); w = (npos > 0).astype(np.float64)
        return (p - y / np.maximum(npos, 1)[fid]) * w[fid], np.maximum(p * (1 - p), 1e-6) * w[fid]
    return obj


def sub(d, feeds):
    M = d["M"][feeds]; off0 = d["off"][feeds]
    rows = np.concatenate([np.arange(o, o + m) for o, m in zip(off0, M)])
    return rows, M, np.concatenate([[0], np.cumsum(M)[:-1]]).astype(np.int64)


uid = dh["uid"]; rng = np.random.default_rng(2000 + seed)
uu = np.unique(uid[uid != 0]); inner_u = set(rng.choice(uu, len(uu) // 5, replace=False).tolist())
inn = np.array([(u in inner_u) if u != 0 else (rng.random() < 0.2) for u in uid.tolist()])
fa, fb = np.where(~inn)[0], np.where(inn)[0]
ra, Ma, oa = sub(dh, fa); rb, Mb, ob = sub(dh, fb)
params = dict(objective=make_obj(Ma, oa), learning_rate=lr, num_leaves=leaves, min_data_in_leaf=minleaf,
              min_sum_hessian_in_leaf=1e-3, feature_fraction=0.7, bagging_fraction=0.7, bagging_freq=1, lambda_l2=10.0,
              max_bin=63, seed=seed, num_threads=12, verbose=-1, max_cat_to_onehot=64, cat_smooth=50, cat_l2=10)
bst = lgb.train(params, lgb.Dataset(Xh[ra], label=dh["y"][ra], init_score=dh["g"][ra], categorical_feature=cats or "auto"),
                num_boost_round=ROUNDS[-1])
gb = dh["g"][rb]; base_in = ndcg_per_feed(gb.astype(np.float32), dh["y"][rb], ob, Mb).mean()
best, bestv = 0, -1
for r in ROUNDS:
    v = ndcg_per_feed((gb + bst.predict(Xh[rb], num_iteration=r, raw_score=True)).astype(np.float32), dh["y"][rb], ob, Mb).mean()
    if v > bestv:
        best, bestv = r, v
print(f"[{tag}] inner holdout gain {bestv - base_in:+.5f} at {best} rounds ({time.time()-t0:.0f}s)", flush=True)
res = bst.predict(Xv, num_iteration=best, raw_score=True)
W = pmix_w()
for gname in ["g", "gto"]:
    b = ndcg(dv, dv[gname]); n = ndcg(dv, dv[gname] + res); dl = n - b
    pm, ps = wmean_se(dl, W["private"])
    M = dv["M"]
    print(f"[{tag}] VAL({gname}) {n.mean():.5f} vs {b.mean():.5f}: delta {dl.mean():+.5f} +- {dl.std()/np.sqrt(len(dl)):.5f} | "
          f"pmix {pm:+.5f}+-{ps:.5f} | M<50 {dl[M<50].mean():+.5f} M>=50 {dl[M>=50].mean():+.5f}")
np.save(f"{D}/gbh2v_res_{tag}.npy", res.astype(np.float32))
imp = bst.feature_importance("gain"); o = np.argsort(-imp)[:20]
print("top gain:", ", ".join(f"{names[i]} {imp[i]/imp.sum():.3f}" for i in o))
print(f"done {time.time()-t0:.0f}s")
