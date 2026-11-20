"""Is there category information beyond the NN's implied category marginal, even when the combiner is fitted on val?

1) Coverage: are positives' categories present in the user's history at all?
2) Category-level stacker fitted on val (5-fold GroupKFold by user, leak feeds excluded from train and eval):
   features = NN group features (LSE, rank, max, mean) [+ handcrafted group features] [+ train category-model logit].
   OOF category NDCG vs NN-LSE (paired).
3) Item level: swap the NN category marginal for the OOF stacker's (nested choice of T, a).
4) Segment breakdown.
"""
import json, sys, time
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from catlib import *

t0 = time.time()
base_name = sys.argv[1] if len(sys.argv) > 1 else "nn_trainonly"
v = load_view("val", with_hist=False)
v = build_groups(v)
va = dict(np.load(OUT / "grp_val.npz", allow_pickle=True))
assert np.array_equal(va["g_cat"], v["g_cat"])
names = list(va["names"]); Xh = va["X"]
leak = leak_mask_val(); keep = ~leak
mask = keep & (v["G"] >= 2)
off, M, cg, gf = v["off"], v["M"], v["cand_g"], v["g_feed"]
if base_name == "v5blend":
    stk = feed_z(off, M, np.load(PREDS / "stk2_res4_clean_val.npy").astype(np.float64))
    s = feed_z(off, M, 0.5 * feed_z(off, M, np.load(PREDS / "nn_final2_val_oof.npy").astype(np.float64)) + 0.5 * stk)
else:
    fn = {"nn_trainonly": "nn_final2_trainonly_val", "nn_oof": "nn_final2_val_oof"}[base_name]
    s = feed_z(off, M, np.load(PREDS / f"{fn}.npy").astype(np.float64))
out = {"base": base_name}

# ---------------------------------------------------------------- 1) coverage
hc = Xh[:, names.index("h_cnt")]; h20 = Xh[:, names.index("h_cnt20")]
yg = v["g_y"] > 0
cov = dict(pos_cat_in_hist=float((hc[yg] > 0).mean()), all_cat_in_hist=float((hc > 0).mean()),
           pos_cat_in_last20=float((h20[yg] > 0).mean()), all_cat_in_last20=float((h20 > 0).mean()))
# feeds where no positive category is in the history at all
fpos_inhist = np.bincount(gf, weights=(yg & (hc > 0)), minlength=len(M)) > 0
cov["feeds_no_pos_cat_in_hist"] = float((~fpos_inhist)[mask].mean())
lse1 = glse(v, s)
nd_nn = cat_ndcg(v, lse1)
nd_item = item_ndcg(v, s)
cov["catndcg_nn_pos_in_hist"] = mse(nd_nn, mask & fpos_inhist)
cov["catndcg_nn_pos_not_in_hist"] = mse(nd_nn, mask & ~fpos_inhist)
cov["itemndcg_nn_pos_in_hist"] = mse(nd_item, keep & fpos_inhist)
cov["itemndcg_nn_pos_not_in_hist"] = mse(nd_item, keep & ~fpos_inhist)
# rank of the best positive category under NN-LSE
print(json.dumps(cov, indent=1), flush=True)
out["coverage"] = cov

# ---------------------------------------------------------------- 2) val-fitted category stacker
def grk(x):
    return feed_pctrank(v["goff"], v["G"], np.asarray(x, np.float64))
lse_feed = np.empty(len(M)); _ = None
mx = np.full(len(M), -np.inf); np.maximum.at(mx, gf, lse1)
logp_nn = lse1 - (mx[gf] + np.log(np.bincount(gf, weights=np.exp(lse1 - mx[gf]), minlength=len(M)))[gf])
nnf = np.stack([lse1, logp_nn, grk(lse1), gmax(v, s), gmean(v, s), 0.5 * glse(v, s / 0.5), 2 * glse(v, s / 2)], 1)
nn_names = ["nn_lse1", "nn_logp", "nn_lse_rk", "nn_max", "nn_mean", "nn_lse05", "nn_lse2"]
catA = np.load(OUT / "cat_A_val_grp.npy").astype(np.float64)[:, None]
users = v["user_id"].copy(); users[users == 0] = -np.arange(1, (users == 0).sum() + 1)
idx = np.where(keep)[0]
folds = np.full(len(M), -1)
for k, (_, te_) in enumerate(GroupKFold(5, shuffle=True, random_state=0).split(idx, groups=users[idx])):
    folds[idx[te_]] = k
gfold = folds[gf]
P = dict(objective="binary", learning_rate=0.03, num_leaves=15, min_data_in_leaf=200, feature_fraction=0.8,
         bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, num_threads=16, verbose=-1, seed=0)
sets = {"nn_only": (nnf, nn_names),
        "nn+hand": (np.hstack([nnf, Xh]), nn_names + names),
        "nn+catA": (np.hstack([nnf, catA]), nn_names + ["catA"]),
        "hand_only": (Xh, names)}
oofs = {}
for sname, (X, fnames) in sets.items():
    t1 = time.time()
    oof = np.zeros(len(gf))
    for k in range(5):
        trg = (gfold >= 0) & (gfold != k); teg = gfold == k
        d = lgb.Dataset(X[trg], v["g_y"][trg], feature_name=fnames, categorical_feature=["cat"] if "cat" in fnames else [])
        # fixed rounds (no early stopping on the evaluated fold); 3 inner choices max
        b = lgb.train(P, d, 600)
        oof[teg] = b.predict(X[teg], raw_score=True)
    oofs[sname] = oof
    nd = cat_ndcg(v, oof)
    r = dict(catndcg=mse(nd, mask), d_vs_nnLSE=paired(nd, nd_nn, mask))
    out[f"stack_{sname}"] = r
    print(f"[{base_name}] cat-stack {sname:10s}: catNDCG {r['catndcg'][0]:.4f} | vs NN-LSE {r['d_vs_nnLSE'][0]:+.4f} ± {r['d_vs_nnLSE'][1]:.4f} | {time.time()-t1:.0f}s", flush=True)

# ---------------------------------------------------------------- 3) item-level swap with the OOF stacker
def nested(grid):
    res = np.full(len(M), np.nan); ch = []
    for k in range(5):
        trm = keep & (folds >= 0) & (folds != k); tem = folds == k
        best = max(grid, key=lambda p_: grid[p_][trm].mean()); ch.append(str(best)); res[tem] = grid[best][tem]
    return res, ch
for sname in ["nn_only", "nn+hand", "nn+catA"]:
    lq = (-np.log1p(np.exp(-oofs[sname])))[cg]
    grid = {}
    for T in (0.5, 1.0, 2.0):
        lnn = (T * glse(v, s / T))[cg] / T
        for a in (0.05, 0.1, 0.2, 0.3, 0.5, 1.0):
            grid[(T, a)] = item_ndcg(v, s / T + a * (lq - lnn))
    oo, ch = nested(grid)
    d = paired(oo, nd_item, keep)
    out[f"item_swap_{sname}"] = dict(d=d, chosen=ch)
    print(f"[{base_name}] item swap with {sname:8s}: {d[0]:+.4f} ± {d[1]:.4f}  chosen {ch}", flush=True)

# ---------------------------------------------------------------- 4) segments (NN-LSE catNDCG and the nn+hand stacker delta)
nd_h = cat_ndcg(v, oofs["nn+hand"])
segs = {}
Hh = v["H"]; uz = v["user_id"] == 0
for nm, sm in [("M<30", M < 30), ("30<=M<100", (M >= 30) & (M < 100)), ("M>=100", M >= 100),
               ("H<20", Hh < 20), ("20<=H<100", (Hh >= 20) & (Hh < 100)), ("H>=100", Hh >= 100), ("user0", uz),
               ("pos cat in hist", fpos_inhist), ("pos cat NOT in hist", ~fpos_inhist)]:
    mm = mask & sm
    segs[nm] = dict(n=int(mm.sum()), nn_lse=float(nd_nn[mm].mean()), stack_delta=paired(nd_h, nd_nn, mm),
                    item_nn=float(nd_item[keep & sm].mean()))
    print(f"  seg {nm:20s} n={mm.sum():5d} NN catNDCG {nd_nn[mm].mean():.4f} | nn+hand delta {segs[nm]['stack_delta'][0]:+.4f} ± {segs[nm]['stack_delta'][1]:.4f} | item NN {nd_item[keep & sm].mean():.4f}")
out["segments"] = segs
json.dump(out, open(OUT / f"06_val_cv_catstack_{base_name}.json", "w"), indent=1)
print("total", time.time() - t0)
