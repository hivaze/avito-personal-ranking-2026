"""Step 3, SECONDARY family (information only; DECLARATION.txt C4): LightGBM lambdarank on top of the pool score.

  init_score = beta_fold * b (beta_fold = conditional logit on [b] alone, training folds); final score = init + trees.
  num_leaves 15, lr 0.05, 200 rounds (fixed), min_data_in_leaf 100, lambda_l2 10, truncation 100, deterministic.
  control features [b, M, cold]; arm = control + [n_w, r_w, z_w, cs_w] per window; arms A and ALL; same folds / seeds as s3.
Usage: s3b_lgb.py real            -> out/s3b_lgb.json
       s3b_lgb.py P1 100 109      -> placebo draws (profile swap) for arm A, appended to out/s3b_lgb_P1.jsonl
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

PARAMS = dict(objective="lambdarank", num_leaves=15, learning_rate=0.05, min_data_in_leaf=100, lambda_l2=10.0,
              lambdarank_truncation_level=100, feature_fraction=1.0, deterministic=True, force_row_wise=True, num_threads=12,
              seed=0, verbose=-1)
ROUNDS = 200
mode = sys.argv[1]
t0 = time.time()
V = Val()
assert abs(V.base_ndcg.mean() - 0.40271) < 2e-5
cm = np.load(f"{OUT}/cnt_main.npy")[:V.N]
Xctl = np.stack([V.b, V.M[V.fid].astype(np.float64), V.cold.astype(np.float64)], 1)


def arm_feats(c, arm):
    cols = [Xctl]
    for w in ARMS[arm]:
        n = c[:, WIN.index(w)].astype(np.float64)
        r, z = forms(V.off, V.M, n)
        cols.append(np.stack([n, r, z, V.cold * (n >= 1)], 1))
    return np.hstack(cols)


def run(mats):
    """mats {name: feature matrix}; returns {name: [per-seed per-feed NDCG]}."""
    out = {k: [] for k in mats}
    for sd in FOLD_SEEDS:
        fold = V.folds(sd)
        sc = {k: np.zeros(V.N) for k in mats}
        for kf in range(NFOLD):
            te_f = fold == kf; tr_rows = ~te_f[V.fid]; te_rows = te_f[V.fid]
            beta = Fitter(V, ~te_f).fit(None)[0][0]
            for k, X in mats.items():
                ds = lgb.Dataset(X[tr_rows], label=V.y[tr_rows], group=V.M[~te_f], init_score=beta * V.b[tr_rows], free_raw_data=False)
                bst = lgb.train(PARAMS, ds, num_boost_round=ROUNDS)
                sc[k][te_rows] = beta * V.b[te_rows] + bst.predict(X[te_rows], raw_score=True)
        for k in mats:
            out[k].append(ndcg_per_feed(sc[k], V.y, V.off, V.M))
        print(f"  fold seed {sd}: " + " | ".join(f"{k} {out[k][-1].mean():.5f}" for k in mats) + f" | {time.time() - t0:.0f}s", flush=True)
    return out


if mode == "real":
    W = pmix.weights()
    mats = {"control": Xctl, "A": arm_feats(cm, "A"), "ALL": arm_feats(cm, "ALL"), "B": arm_feats(cm, "B")}
    out = run(mats)
    res = {"control_ndcg_by_seed": [float(x.mean()) for x in out["control"]], "base_ndcg": float(V.base_ndcg.mean())}
    dc, gc, sec, zc = gain(out["control"], [V.base_ndcg] * len(FOLD_SEEDS))
    print(f"LGB control - raw pool score b: {gc:+.5f} +- {sec:.5f} (z {zc:+.2f})")
    res["control_minus_pool"] = [gc, sec, zc]
    save = {f"control_seed{s}": out["control"][i] for i, s in enumerate(FOLD_SEEDS)}
    for k in ("A", "B", "ALL"):
        d, g, se, z = gain(out[k], out["control"])
        pub, priv = wstat(d, W["public"]), wstat(d, W["private"])
        res[k] = dict(gain=g, se=se, z=z, per_seed=[float((a - c).mean()) for a, c in zip(out[k], out["control"])], pmix_public=pub, pmix_private=priv)
        save["d_" + k] = d
        print(f"LGB arm {k:4s}: GAIN {g:+.5f} +- {se:.5f} (z {z:+.2f}) | per seed {res[k]['per_seed']} | pmix public {pub[0]:+.5f}+-{pub[1]:.5f} private {priv[0]:+.5f}+-{priv[1]:.5f}")
    np.savez(f"{OUT}/s3b_lgb_perfeed.npz", **save)
    json.dump(res, open(f"{OUT}/s3b_lgb.json", "w"), indent=1)
else:
    s0, s1 = int(sys.argv[2]), int(sys.argv[3])
    ref = np.load(f"{OUT}/s3b_lgb_perfeed.npz")
    ctl = [ref[f"control_seed{s}"] for s in FOLD_SEEDS]
    day_r = V.day[V.fid]; blk = (((V.ts + 10800) % 86400) // 10800)[V.fid]
    strata = (day_r - day_r.min()) * 16 + blk * 2 + V.cold.astype(np.int64)
    order = np.argsort(strata, kind="stable"); ss = strata[order]
    bounds = np.flatnonzero(np.r_[True, ss[1:] != ss[:-1], True])
    for seed in range(s0, s1 + 1):
        rng = np.random.default_rng(seed)
        src = np.arange(V.N)
        for a, b in zip(bounds[:-1], bounds[1:]):
            idx = order[a:b]
            src[idx] = idx[rng.permutation(b - a)]
        out = run({"A": arm_feats(cm[src], "A")})
        d, g, se, z = gain(out["A"], ctl)
        with open(f"{OUT}/s3b_lgb_P1.jsonl", "a") as f:
            f.write(json.dumps({"seed": seed, "A": [g, se, z]}) + "\n")
        print(f"[LGB P1] seed {seed}: A {g:+.5f} (z {z:+.2f})", flush=True)
print(f"DONE {time.time() - t0:.0f}s")
