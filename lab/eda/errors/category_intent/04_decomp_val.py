"""Item-level test of the category-intent model on val (leak feeds excluded).

score variants (s = per-feed z-scored base score, lq = category-model log P(category) of the candidate's category):
  swap(T, a):   s/T + a * (lq - logP_NN(c; T))     (a=1: replace the base's implied category marginal by the model's)
  add(b):       s + b * lq
  literal:      lq + z_within_category(s)
Parameters are chosen by nested 5-fold CV over val feeds (GroupKFold by user): pick on 4 folds, score the 5th.
"""
import json, sys, time
import numpy as np
from sklearn.model_selection import GroupKFold
from catlib import *

t0 = time.time()
cat_tags = sys.argv[1].split(",") if len(sys.argv) > 1 else ["A"]
v = load_view("val", with_hist=False)
v = build_groups(v)
leak = leak_mask_val()
keep = ~leak
off, M = v["off"], v["M"]
cg = v["cand_g"]
users = v["user_id"].copy()
users[users == 0] = -np.arange(1, (users == 0).sum() + 1)  # user 0: each feed its own group
folds = np.full(len(M), -1)
idx = np.where(keep)[0]
for k, (_, te_) in enumerate(GroupKFold(5, shuffle=True, random_state=0).split(idx, groups=users[idx])):
    folds[idx[te_]] = k

bases = {"nn_trainonly": np.load(PREDS / "nn_final2_trainonly_val.npy").astype(np.float64),
         "nn_oof": np.load(PREDS / "nn_final2_val_oof.npy").astype(np.float64)}
stk = feed_z(off, M, np.load(PREDS / "stk2_res4_clean_val.npy").astype(np.float64))
bases["v5blend"] = feed_z(off, M, 0.5 * feed_z(off, M, bases["nn_oof"]) + 0.5 * stk)
bases = {k: feed_z(off, M, x) for k, x in bases.items()}


def within_cat_z(s):
    mu = np.bincount(cg, weights=s, minlength=len(v["g_n"])) / v["g_n"]
    sd = np.sqrt(np.maximum(np.bincount(cg, weights=s * s, minlength=len(v["g_n"])) / v["g_n"] - mu ** 2, 0)) + 1e-6
    z = (s - mu[cg]) / sd[cg]
    z[v["g_n"][cg] == 1] = 0.0
    return z


def nested(perfeed_by_param, base_pf):
    """perfeed_by_param: dict param -> per-feed ndcg. Returns OOF per-feed ndcg and chosen params per fold."""
    oof = np.full(len(M), np.nan); chosen = []
    for k in range(5):
        trm = keep & (folds != k) & (folds >= 0); tem = folds == k
        best = max(perfeed_by_param, key=lambda p_: perfeed_by_param[p_][trm].mean())
        chosen.append(str(best)); oof[tem] = perfeed_by_param[best][tem]
    return oof, chosen


out = {}
for tag in cat_tags:
    gp = np.load(OUT / f"cat_{tag}_val_grp.npy").astype(np.float64)
    if tag[0] in "BS":
        lq_g = gp  # lambdarank utility, scale absorbed by a / b
    else:
        lq_g = -np.log1p(np.exp(-gp))  # log sigmoid
    lq = lq_g[cg]
    for bname, s in bases.items():
        t1 = time.time()
        nd0 = item_ndcg(v, s)
        grid_swap = {}
        for T in (0.5, 1.0, 2.0):
            lnn = (T * glse(v, s / T))[cg] / T  # log-sum-exp of s/T within category (feed constant irrelevant)
            for a in (0.1, 0.2, 0.3, 0.5, 0.7, 1.0):
                grid_swap[("swap", T, a)] = item_ndcg(v, s / T + a * (lq - lnn))
        grid_add = {("add", b): item_ndcg(v, s + b * lq) for b in (0.05, 0.1, 0.2, 0.3, 0.5, 1.0)}
        lit = item_ndcg(v, lq + within_cat_z(s))
        oof_s, ch_s = nested(grid_swap, nd0)
        oof_a, ch_a = nested(grid_add, nd0)
        allg = dict(grid_swap); allg.update(grid_add)
        oof_all, ch_all = nested(allg, nd0)
        r = dict(base=mse(nd0, keep),
                 literal=dict(m=mse(lit, keep), d=paired(lit, nd0, keep)),
                 swap_nested=dict(m=mse(oof_s, keep), d=paired(oof_s, nd0, keep), chosen=ch_s),
                 add_nested=dict(m=mse(oof_a, keep), d=paired(oof_a, nd0, keep), chosen=ch_a),
                 any_nested=dict(m=mse(oof_all, keep), d=paired(oof_all, nd0, keep), chosen=ch_all),
                 grid={str(k_): paired(x, nd0, keep) for k_, x in allg.items()})
        out[f"{tag}|{bname}"] = r
        print(f"[cat {tag} | base {bname}] base {r['base'][0]:.4f} | literal {r['literal']['d'][0]:+.4f}±{r['literal']['d'][1]:.4f}"
              f" | swap(nested) {r['swap_nested']['d'][0]:+.4f}±{r['swap_nested']['d'][1]:.4f} {ch_s}"
              f" | add(nested) {r['add_nested']['d'][0]:+.4f}±{r['add_nested']['d'][1]:.4f} {ch_a} | {time.time()-t1:.0f}s", flush=True)
        bestg = sorted(r["grid"].items(), key=lambda t: -t[1][0])[:5]
        print("    best grid points (in-sample, optimistic):", [(k_, round(d[0], 5), round(d[1], 5)) for k_, d in bestg], flush=True)
json.dump(out, open(OUT / f"04_decomp_{'_'.join(cat_tags)}.json", "w"), indent=1)
print("total", time.time() - t0)
