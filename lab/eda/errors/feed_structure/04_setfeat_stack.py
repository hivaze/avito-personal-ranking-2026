"""Learned test: do model-relative set-level features (similarity to the base model's top-k, category softmax mass,
best-in-category, near-duplicate group stats, feed confidence) add information on top of a base score?
LightGBM lambdarank with init_score = z(base), GroupKFold(5) by user on non-leak val feeds.
Control = [z, M] only (same model, same folds); paired per-feed NDCG difference set - control."""
import argparse, time
from fs_common import *
import numba as nb, lightgbm as lgb
from sklearn.model_selection import GroupKFold

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="nn_final2_trainonly_val")
ap.add_argument("--rounds", default="100,300")
ap.add_argument("--obj", default="lambdarank")
args = ap.parse_args()


@nb.njit(parallel=True, cache=True)
def set_feats(X, miss, off, M, z, cat):
    n = len(z); K = 13
    F = np.full((n, K), np.nan, np.float32)
    for f in nb.prange(len(M)):
        o = off[f]; m = M[f]
        zf = z[o:o + m]; cf = cat[o:o + m]
        Xf = X[o:o + m]; C = Xf @ Xf.T
        order = np.argsort(-zf)
        t1 = order[0]
        zm = zf.max(); w = np.exp(zf - zm); w /= w.sum()
        ent = -(w * np.log(w + 1e-12)).sum() / np.log(max(m, 2))
        # category mass and best-in-category
        for i in range(m):
            mass = 0.0; better = 0; ncat = 0
            for j in range(m):
                if cf[j] == cf[i]:
                    mass += w[j]; ncat += 1
                    if zf[j] > zf[i]: better += 1
            F[o + i, 0] = np.log(mass + 1e-12)
            F[o + i, 1] = better
            F[o + i, 2] = ncat / m
            F[o + i, 3] = 1.0 if (cf[i] == cf[t1] and i != t1) else 0.0
            F[o + i, 4] = zf[t1] - zf[i]
            F[o + i, 5] = ent
        for i in range(m):
            if miss[o + i]: continue
            # cos to top1 (to 2nd for the top1 itself), max cos to top3 / top10 excluding self, softmax-weighted cos, near-dup stats
            c1 = C[i, order[1]] if (i == t1 and m > 1) else C[i, t1]
            F[o + i, 6] = c1
            mx3 = -2.0; mx10 = -2.0
            for r in range(min(10, m)):
                j = order[r]
                if j == i or miss[o + j]: continue
                if r < 3 and C[i, j] > mx3: mx3 = C[i, j]
                if C[i, j] > mx10: mx10 = C[i, j]
            F[o + i, 7] = mx3 if mx3 > -2 else np.nan
            F[o + i, 8] = mx10 if mx10 > -2 else np.nan
            s = 0.0; ws = 0.0; nd = 0; dmax = zf[i]; csum = 0.0
            for j in range(m):
                if j == i or miss[o + j]: continue
                s += w[j] * C[i, j]; ws += w[j]; csum += C[i, j]
                if C[i, j] > 0.95:
                    nd += 1
                    if zf[j] > dmax: dmax = zf[j]
            F[o + i, 9] = s / ws if ws > 0 else np.nan
            F[o + i, 10] = nd
            F[o + i, 11] = dmax - zf[i]
            F[o + i, 12] = csum / max(m - 1, 1)
    return F

FN = ["cat_logmass", "n_better_in_cat", "cat_share", "same_cat_top1", "gap_top1", "feed_entropy", "cos_top1", "maxcos_top3",
      "maxcos_top10", "softcos", "n_dup95", "dup95_gap", "mean_cos_feed"]

t0 = time.time()
d = load_val()
X, miss = load_emb_norm(d["emb_idx"])
M, off, t, fid = d["M"], d["off"], d["target"], d["fid"]
use = ~d["leak"]
s = np.load(PREDS / f"{args.base}.npy").astype(np.float64)
z = zfeed(s, off, M)
F = set_feats(X, miss, off, M, z, d["category_id"].astype(np.int64))
# centre within-feed for features whose feed-level part is irrelevant? keep raw; GBDT with init_score handles it.
base_pf = ndcg_per_feed(z, t, off, M)
print(f"features {time.time()-t0:.1f}s; base {args.base} non-leak {base_pf[use].mean():.5f}")

feeds = np.where(use)[0]
users = d["feeds"]["user_id"].to_numpy()[feeds]
rowsel = use[fid]
Xall = np.column_stack([z, M[fid].astype(np.float32), F]).astype(np.float32)
names = ["z", "M"] + FN
rounds = [int(r) for r in args.rounds.split(",")]
preds = {(c, r): np.zeros(len(z)) for c in ("control", "set") for r in rounds}
gkf = GroupKFold(n_splits=5)
params = dict(objective=args.obj, lambdarank_truncation_level=300, learning_rate=0.03, num_leaves=15, min_data_in_leaf=500,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, num_threads=16, verbose=-1, seed=0)
imp = np.zeros(len(names))
for k, (tr, te) in enumerate(gkf.split(feeds, groups=users)):
    ftr, fte = feeds[tr], feeds[te]
    rtr = np.concatenate([np.arange(off[f], off[f] + M[f]) for f in ftr])
    rte = np.concatenate([np.arange(off[f], off[f] + M[f]) for f in fte])
    if args.obj == "binary":
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression(C=100.0).fit(z[rtr, None], t[rtr])
        init = lr.decision_function(z[:, None])
    else:
        init = z
    for c, cols in (("control", [0, 1]), ("set", list(range(len(names))))):
        ds = lgb.Dataset(Xall[rtr][:, cols], t[rtr], group=M[ftr], init_score=init[rtr], feature_name=[names[i] for i in cols])
        bst = lgb.train(params, ds, num_boost_round=max(rounds))
        for r in rounds:
            preds[(c, r)][rte] = init[rte] + bst.predict(Xall[rte][:, cols], num_iteration=r, raw_score=True)
        if c == "set":
            imp += bst.feature_importance("gain")
    print(f"fold {k} done {time.time()-t0:.1f}s", flush=True)

res = {}
for r in rounds:
    pc = ndcg_per_feed(preds[("control", r)], t, off, M); ps = ndcg_per_feed(preds[("set", r)], t, off, M)
    m1 = paired(pc, base_pf, use); m2 = paired(ps, base_pf, use); m3 = paired(ps, pc, use)
    res[r] = dict(control_vs_base=m1, set_vs_base=m2, set_vs_control=m3)
    print(f"rounds {r}: control-base {m1[0]:+.5f} ± {m1[1]:.5f} | set-base {m2[0]:+.5f} ± {m2[1]:.5f} | set-control {m3[0]:+.5f} ± {m3[1]:.5f}")
order = np.argsort(-imp)
print("gain importance (set model):", ", ".join(f"{names[i]} {imp[i]/imp.sum():.3f}" for i in order))
json.dump(dict(base=args.base, res={str(k): v for k, v in res.items()}, imp={names[i]: float(imp[i] / imp.sum()) for i in order}),
          open(OUT / f"out_04_{args.base}_{args.obj}.json", "w"), indent=1)
