"""Train the category-intent model on train (feed, category) groups; evaluate category NDCG on val.

Variants (coarse, <= 3): A binary on R+S, B lambdarank on R+S, C binary on R only.
Early stopping on a 10% user-disjoint train holdout (never on val).
"""
import json, sys, time
import numpy as np
import lightgbm as lgb
from catlib import *

t0 = time.time()
tr = dict(np.load(OUT / "grp_train.npz", allow_pickle=True))
va = dict(np.load(OUT / "grp_val.npz", allow_pickle=True))
te = dict(np.load(OUT / "grp_test.npz", allow_pickle=True))
names = list(tr["names"])
drop = set(sys.argv[2].split(",")) if len(sys.argv) > 2 and sys.argv[2] else set()
use = [i for i, n in enumerate(names) if n not in drop]
fn = [names[i] for i in use]
variants = sys.argv[1].split(",") if len(sys.argv) > 1 else ["A", "B", "C"]
tag = sys.argv[3] if len(sys.argv) > 3 else ""

nf = len(tr["G"])
users = tr["user_id"]
rng = np.random.default_rng(1)
uu = np.unique(users)
hold_u = set(rng.choice(uu, int(0.1 * len(uu)), replace=False).tolist())
hold_f = np.array([u in hold_u for u in users]) & (users != 0) | ((users == 0) & (rng.random(nf) < 0.1))
print(f"train feeds {nf}, holdout feeds {hold_f.sum()} | features {len(fn)}", flush=True)
gmask_hold = hold_f[tr["g_feed"]]

leak = leak_mask_val()


def val_view():
    return dict(g_y=va["g_y"], goff=va["goff"], G=va["G"])


vview = val_view()
mask = (~leak) & (va["G"] >= 2)
nn_lse_ref = None
ref = np.load(OUT / "01_val_catndcg_perfeed.npz")
nn_ref = ref["nn_trainonly_LSE_T1.0"]
f3_ref = ref["f3_+_tbn"]

P = dict(learning_rate=0.05, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8,
         bagging_freq=1, lambda_l2=10.0, num_threads=16, verbose=-1, seed=0, cat_l2=10.0, cat_smooth=20.0)
res = {}
for var in variants:
    t1 = time.time()
    fsel = np.ones(nf, bool) if var in ("A", "B") else tr["is_R"]
    trf = fsel & ~hold_f
    hof = fsel & hold_f
    gtr = trf[tr["g_feed"]]; gho = hof[tr["g_feed"]]
    X = tr["X"][:, use]
    p = dict(P)
    if var == "B":
        p.update(objective="lambdarank", metric="ndcg", eval_at=[100], lambdarank_truncation_level=100)
        dtr = lgb.Dataset(X[gtr], tr["g_y"][gtr], group=tr["G"][trf], feature_name=fn, categorical_feature=["cat"])
        dho = lgb.Dataset(X[gho], tr["g_y"][gho], group=tr["G"][hof], reference=dtr)
    else:
        p.update(objective="binary", metric="binary_logloss")
        dtr = lgb.Dataset(X[gtr], tr["g_y"][gtr], feature_name=fn, categorical_feature=["cat"])
        dho = lgb.Dataset(X[gho], tr["g_y"][gho], reference=dtr)
    b = lgb.train(p, dtr, 4000, valid_sets=[dho], callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(200)])
    pv = b.predict(va["X"][:, use], raw_score=True)
    pt = b.predict(te["X"][:, use], raw_score=True)
    ph = b.predict(X[gho], raw_score=True)
    nd = cat_ndcg(vview, pv)
    # holdout category ndcg (train)
    hv = dict(g_y=tr["g_y"][gho], goff=np.concatenate([[0], np.cumsum(tr["G"][hof])[:-1]]), G=tr["G"][hof])
    ndh = cat_ndcg(hv, ph)
    m, se = mse(nd, mask)
    d_nn = paired(nd, nn_ref, mask); d_f3 = paired(nd, f3_ref, mask)
    imp = sorted(zip(fn, b.feature_importance("gain")), key=lambda t: -t[1])
    tot = sum(i for _, i in imp)
    res[var] = dict(best_iter=b.best_iteration, val_catndcg=m, se=se, d_vs_nnLSE=d_nn, d_vs_f3=d_f3,
                    holdout_catndcg=float(ndh.mean()),
                    top_gain=[(n, round(i / tot, 4)) for n, i in imp[:25]])
    print(f"[{var}{tag}] iters {b.best_iteration} val catNDCG {m:.4f} ± {se:.4f} | vs NN-LSE {d_nn[0]:+.4f} ± {d_nn[1]:.4f}"
          f" | vs f3 {d_f3[0]:+.4f} ± {d_f3[1]:.4f} | holdout(train) {ndh.mean():.4f} | {time.time()-t1:.0f}s", flush=True)
    print("   top gain:", res[var]["top_gain"][:15], flush=True)
    np.save(OUT / f"cat_{var}{tag}_val_grp.npy", pv.astype(np.float32))
    np.save(OUT / f"cat_{var}{tag}_test_grp.npy", pt.astype(np.float32))
    np.save(OUT / f"cat_{var}{tag}_val_perfeed.npy", nd)
    b.save_model(str(OUT / f"cat_{var}{tag}.txt"))
json.dump(res, open(OUT / f"03_catmodel{tag}_{'_'.join(variants)}.json", "w"), indent=1)
print("total", time.time() - t0)
