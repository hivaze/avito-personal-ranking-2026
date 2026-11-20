"""LGB-R (AMENDMENT 3; INFORMATION ONLY, post hoc): a tree residual model that cannot re-shape the pool score.
  features WITHOUT b; control [M, cold]; arm A = control + [n_w, r_w, z_w, cs_w] for windows a, a24 (arm B / ALL also printed);
  init_score = beta_fold * b; num_leaves 7, lr 0.03, 100 rounds, min_data_in_leaf 1000, lambda_l2 10, truncation 100.
  Real read, then the P1 profile-swap placebo for arm A (seeds 100..109), same folds / seeds as s3.
  cd <repo> && CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 .venv/bin/python nn_x/w12/fresh/pregate/s3d_lgbr.py
"""
import json
import sys
import time

import lightgbm as lgb
import numpy as np

sys.path.insert(0, "<repo>/nn_x/w12/fresh/pregate")
from lib import OUT, WIN, forms
from nested import ARMS, FOLD_SEEDS, NFOLD, Fitter, Val, gain, wstat

sys.path.insert(0, "<repo>/src")
from metrics import ndcg_per_feed
import pmix

PARAMS = dict(objective="lambdarank", num_leaves=7, learning_rate=0.03, min_data_in_leaf=1000, lambda_l2=10.0,
              lambdarank_truncation_level=100, feature_fraction=1.0, deterministic=True, force_row_wise=True, num_threads=12,
              seed=0, verbose=-1)
ROUNDS = 100
t0 = time.time()
V = Val(); W = pmix.weights()
cm = np.load(f"{OUT}/cnt_main.npy")[:V.N]
Xctl = np.stack([V.M[V.fid].astype(np.float64), V.cold.astype(np.float64)], 1)


def arm_feats(c, arm):
    cols = [Xctl]
    for w in ARMS[arm]:
        n = c[:, WIN.index(w)].astype(np.float64)
        r, z = forms(V.off, V.M, n)
        cols.append(np.stack([n, r, z, V.cold * (n >= 1)], 1))
    return np.hstack(cols)


BETA = {}


def run(mats):
    out = {k: [] for k in mats}
    for sd in FOLD_SEEDS:
        fold = V.folds(sd)
        sc = {k: np.zeros(V.N) for k in mats}
        for kf in range(NFOLD):
            te_f = fold == kf; tr_rows = ~te_f[V.fid]; te_rows = te_f[V.fid]
            if (sd, kf) not in BETA:
                BETA[(sd, kf)] = Fitter(V, ~te_f).fit(None)[0][0]
            beta = BETA[(sd, kf)]
            for k, X in mats.items():
                ds = lgb.Dataset(X[tr_rows], label=V.y[tr_rows], group=V.M[~te_f], init_score=beta * V.b[tr_rows], free_raw_data=False)
                bst = lgb.train(PARAMS, ds, num_boost_round=ROUNDS)
                sc[k][te_rows] = beta * V.b[te_rows] + bst.predict(X[te_rows], raw_score=True)
        for k in mats:
            out[k].append(ndcg_per_feed(sc[k], V.y, V.off, V.M))
    return out


out = run({"control": Xctl, "A": arm_feats(cm, "A"), "B": arm_feats(cm, "B"), "ALL": arm_feats(cm, "ALL"), "L": arm_feats(cm, "L")})
ctl = out["control"]
dc, gc, sec, zc = gain(ctl, [V.base_ndcg] * len(FOLD_SEEDS))
print(f"LGB-R control {np.mean([c.mean() for c in ctl]):.5f}; control - raw pool score: {gc:+.5f} +- {sec:.5f} (z {zc:+.2f}) | {time.time() - t0:.0f}s", flush=True)
res = {"control_minus_pool": [gc, sec, zc]}
for k in ("A", "B", "ALL", "L"):
    d, g, se, z = gain(out[k], ctl)
    dp, gp, sep, zp = gain(out[k], [V.base_ndcg] * len(FOLD_SEEDS))
    pub, priv = wstat(d, W["public"]), wstat(d, W["private"])
    res[k] = dict(gain=g, se=se, z=z, vs_pool=[gp, sep, zp], pmix_public=pub, pmix_private=priv,
                  per_seed=[float((a - c).mean()) for a, c in zip(out[k], ctl)])
    print(f"LGB-R arm {k:4s}: GAIN vs control {g:+.5f} +- {se:.5f} (z {z:+.2f}) | vs raw pool {gp:+.5f} +- {sep:.5f} (z {zp:+.2f}) | per seed "
          + " ".join(f"{x:+.5f}" for x in res[k]["per_seed"]) + f" | pmix public {pub[0]:+.5f}+-{pub[1]:.5f} private {priv[0]:+.5f}+-{priv[1]:.5f}", flush=True)
day_r = V.day[V.fid]; blk = (((V.ts + 10800) % 86400) // 10800)[V.fid]
strata = (day_r - day_r.min()) * 16 + blk * 2 + V.cold.astype(np.int64)
order = np.argsort(strata, kind="stable"); ss = strata[order]
bounds = np.flatnonzero(np.r_[True, ss[1:] != ss[:-1], True])
res["placebo_A"] = []; res["placebo_B"] = []
for seed in range(100, 110):
    rng = np.random.default_rng(seed)
    src = np.arange(V.N)
    for a, b in zip(bounds[:-1], bounds[1:]):
        idx = order[a:b]
        src[idx] = idx[rng.permutation(b - a)]
    o = run({"A": arm_feats(cm[src], "A"), "B": arm_feats(cm[src], "B")})
    gA = gain(o["A"], ctl); gB = gain(o["B"], ctl)
    res["placebo_A"].append([seed, gA[1], gA[3]]); res["placebo_B"].append([seed, gB[1], gB[3]])
    print(f"[LGB-R P1] seed {seed}: A {gA[1]:+.5f} (z {gA[3]:+.2f}) | B {gB[1]:+.5f} (z {gB[3]:+.2f}) | {time.time() - t0:.0f}s", flush=True)
for k in ("A", "B"):
    pl_ = np.array([x[1] for x in res[f"placebo_{k}"]])
    print(f"LGB-R arm {k}: real {res[k]['gain']:+.5f} | placebo mean {pl_.mean():+.5f} sd {pl_.std(ddof=1):.5f} min {pl_.min():+.5f} max {pl_.max():+.5f} | "
          f"real above {int((res[k]['gain'] > pl_).sum())} of {len(pl_)} draws")
json.dump(res, open(f"{OUT}/s3d_lgbr.json", "w"), indent=1)
print(f"DONE {time.time() - t0:.0f}s")
