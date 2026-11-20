"""R7: probe of the builder's one disagreement (LightGBM secondary). My folds, init_score = b, read at several round counts,
within-feed placebo and a feed-level-only arm. Information only.
"""
import json
import time

import lightgbm as lgb
import numpy as np

from rlib import OUT, Val, ndcg_feeds

t0 = time.time()
V = Val()
assert abs(V.base.mean() - 0.40271) < 2e-5
C = np.load(f"{OUT}/my_counts.npy")[:V.N].astype(np.float64)
PARAMS = dict(objective="lambdarank", num_leaves=15, learning_rate=0.05, min_data_in_leaf=100, lambda_l2=10.0, lambdarank_truncation_level=100,
              feature_fraction=1.0, deterministic=True, force_row_wise=True, num_threads=12, seed=0, verbose=-1)
ROUNDS = (0, 5, 10, 20, 50, 100, 200)
FOLD_SEEDS = (11, 12, 13)
cold = V.cold.astype(np.float64)
Xc = np.stack([V.b, V.M[V.fid].astype(np.float64), cold], 1)


def arm_A(Cx):
    cols = [Xc]
    for k in (0, 1):
        n = Cx[:, k]; r, z = V.forms(n)
        cols.append(np.stack([n, r, z, cold * (n >= 1)], 1))
    return np.hstack(cols)


def arm_F(Cx):
    cols = [Xc]
    for k in (0, 1):
        n = Cx[:, k]
        cols.append(np.stack([(np.bincount(V.fid, n >= 1, V.nf) / V.M)[V.fid], (np.bincount(V.fid, np.log1p(n), V.nf) / V.M)[V.fid]], 1))
    cols.append((np.bincount(V.fid, cold * (Cx[:, 0] >= 1), V.nf) / V.M)[V.fid][:, None])
    return np.hstack(cols)


folds = [V.folds(s) for s in FOLD_SEEDS]


def run(mats):
    """{name: X} -> {name: array [len(ROUNDS), nf] of per-feed NDCG averaged over the fold seeds}."""
    out = {k: np.zeros((len(ROUNDS), V.nf)) for k in mats}
    for fold in folds:
        sc = {k: np.zeros((len(ROUNDS), V.N)) for k in mats}
        for kf in range(5):
            te_f = fold == kf; tr = ~te_f[V.fid]; te = te_f[V.fid]
            for k, X in mats.items():
                ds = lgb.Dataset(X[tr], label=V.y[tr], group=V.M[~te_f], init_score=V.b[tr], free_raw_data=False)
                bst = lgb.train(PARAMS, ds, num_boost_round=max(ROUNDS))
                for ri, r in enumerate(ROUNDS):
                    sc[k][ri, te] = V.b[te] + (bst.predict(X[te], raw_score=True, num_iteration=r) if r > 0 else 0.0)
        for k in mats:
            for ri in range(len(ROUNDS)):
                out[k][ri] += ndcg_feeds(sc[k][ri], V.y, V.off, V.M) / len(folds)
    return out


def st(d):
    g = d.mean(); se = d.std() / np.sqrt(len(d))
    return [float(g), float(se), float(g / se) if se > 0 else 0.0]


real = run({"control": Xc, "A": arm_A(C), "F": arm_F(C)})
res = {"rounds": ROUNDS, "real": {}}
print(f"real fits done {time.time() - t0:.0f}s\n=== REAL (my folds, seeds {FOLD_SEEDS}); per round: control - raw pool | A - control | A - raw pool | F - control", flush=True)
for ri, r in enumerate(ROUNDS):
    a = st(real["control"][ri] - V.base); b_ = st(real["A"][ri] - real["control"][ri]); c = st(real["A"][ri] - V.base); f = st(real["F"][ri] - real["control"][ri])
    res["real"][r] = dict(ctl_minus_pool=a, A_minus_ctl=b_, A_minus_pool=c, F_minus_ctl=f)
    print(f"  rounds {r:3d}: ctl-pool {a[0]:+.5f}+-{a[1]:.5f} | A-ctl {b_[0]:+.5f}+-{b_[1]:.5f} (z {b_[2]:+.2f}) | A-pool {c[0]:+.5f}+-{c[1]:.5f} (z {c[2]:+.2f}) | F-ctl {f[0]:+.5f}+-{f[1]:.5f} (z {f[2]:+.2f})", flush=True)
json.dump(res, open(f"{OUT}/r7_lgb.json", "w"), indent=1)


def permute_within(grp, rng):
    u = rng.random(V.N)
    so = np.argsort(grp, kind="stable"); sr = np.lexsort((u, grp))
    src = np.empty(V.N, np.int64); src[so] = sr
    return src


grp_feed = V.fid * 2 + V.cold.astype(np.int64)
blk = (((V.ts + 10800) % 86400) // 10800)[V.fid]
grp_strat = (V.day[V.fid] - V.day.min()) * 16 + blk * 2 + V.cold.astype(np.int64)
res["placebo"] = []
for kind, grp, seeds in (("WF", grp_feed, range(500, 510)), ("STRAT", grp_strat, range(600, 605))):
    for seed in seeds:
        Cp = C[permute_within(grp, np.random.default_rng(seed))]
        o = run({"A": arm_A(Cp)})
        rec = {"kind": kind, "seed": seed, "A_minus_ctl": {r: st(o["A"][ri] - real["control"][ri]) for ri, r in enumerate(ROUNDS)},
               "A_minus_pool": {r: st(o["A"][ri] - V.base) for ri, r in enumerate(ROUNDS)}}
        res["placebo"].append(rec)
        print(f"[{kind}] seed {seed}: A-ctl by round " + " ".join(f"{r}:{rec['A_minus_ctl'][r][0]:+.5f}" for r in ROUNDS[1:]) + f" | {time.time() - t0:.0f}s", flush=True)
        json.dump(res, open(f"{OUT}/r7_lgb.json", "w"), indent=1)
print(f"DONE {time.time() - t0:.0f}s")
