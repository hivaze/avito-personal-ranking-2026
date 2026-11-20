"""(c) Does ANY nonlinear residual structure exist on top of G3? LightGBM boosted FROM z(G3) (init_score = g) with a
listwise softmax-CE objective (per-feed normalised, = the NN's loss), user-grouped 5-fold OOF on val; round count chosen
on an inner 20%-of-users split of the training folds (never on the outer fold). Control = the same with features [g] only
(level 0), same folds and seeds. Report OOF NDCG deltas vs raw G3 and the paired difference level - control.
Holdout: model trained on ALL val (inner split for rounds) applied to holdout (init = holdout train-only pool z).
Usage: python gbdt.py <level 0|1|2> <seed> [lr] [leaves] [min_leaf]
Writes oof_<level>_<seed>.npy (val OOF scores) and hold_<level>_<seed>.npy.
"""
import sys, time
import numpy as np
import lightgbm as lgb
sys.path.insert(0, "/tmp/scratchpad/dpa/nonlinear")
from nlc import load, folds5, ndcg, D  # noqa
from featmat import build

t0 = time.time()
level, seed = int(sys.argv[1]), int(sys.argv[2])
lr = float(sys.argv[3]) if len(sys.argv) > 3 else 0.03
leaves = int(sys.argv[4]) if len(sys.argv) > 4 else 15
minleaf = int(sys.argv[5]) if len(sys.argv) > 5 else 500
tag = f"L{level}_s{seed}_lr{lr}_nl{leaves}_ml{minleaf}"
dv = load("val"); dh = load("hold")
Xv, names, cats = build(dv, level); Xh, names_h, _ = build(dh, level)
assert names == names_h
print(f"[{tag}] features {len(names)} ({time.time()-t0:.0f}s)", flush=True)
fold = folds5(dv, seed=0)  # fixed outer folds (same for every arm/seed)
ROUNDS = list(range(10, 401, 10))


def make_obj(M, off):
    fid = np.repeat(np.arange(len(M)), M)

    def obj(preds, ds):
        y = ds.get_label()
        mx = np.maximum.reduceat(preds, off)
        e = np.exp(preds - mx[fid]); Z = np.bincount(fid, e, len(M)); p = e / Z[fid]
        npos = np.bincount(fid, y, len(M))
        w = (npos > 0).astype(np.float64)
        gr = (p - y / np.maximum(npos, 1)[fid]) * w[fid]
        he = np.maximum(p * (1 - p), 1e-6) * w[fid]
        return gr, he
    return obj


def sub(d, feeds):
    """rows/offsets of a subset of feeds (kept in order)."""
    M = d["M"][feeds]; off0 = d["off"][feeds]
    rows = np.concatenate([np.arange(o, o + m) for o, m in zip(off0, M)])
    off = np.concatenate([[0], np.cumsum(M)[:-1]]).astype(np.int64)
    return rows, M, off


def fit_predict(d, X, tr_feeds, rng_seed, targets):
    """inner split of tr_feeds by user (20%), fit, choose rounds on inner, predict list of (X_t, g_t, M_t, off_t)."""
    uid = d["uid"][tr_feeds]
    rng = np.random.default_rng(1000 + rng_seed)
    uu = np.unique(uid[uid != 0]); inner_u = set(rng.choice(uu, len(uu) // 5, replace=False).tolist())
    inn = np.array([(u in inner_u) if u != 0 else (rng.random() < 0.2) for u in uid.tolist()])
    fa, fb = tr_feeds[~inn], tr_feeds[inn]
    ra, Ma, oa = sub(d, fa); rb, Mb, ob = sub(d, fb)
    params = dict(objective=make_obj(Ma, oa), learning_rate=lr, num_leaves=leaves, min_data_in_leaf=minleaf,
                  min_sum_hessian_in_leaf=1e-3, feature_fraction=0.7, bagging_fraction=0.7, bagging_freq=1,
                  lambda_l2=10.0, max_bin=63, seed=rng_seed, num_threads=12, verbose=-1,
                  max_cat_to_onehot=64, cat_smooth=50, cat_l2=10)
    dtr = lgb.Dataset(X[ra], label=d["y"][ra], init_score=d["g"][ra], categorical_feature=cats or "auto", free_raw_data=False)
    bst = lgb.train(params, dtr, num_boost_round=ROUNDS[-1])
    best, bestv = 0, None
    gb = d["g"][rb]
    from metrics import ndcg_per_feed
    base_in = ndcg_per_feed(gb.astype(np.float32), d["y"][rb], ob, Mb).mean()
    for r in ROUNDS:
        v = ndcg_per_feed((gb + bst.predict(X[rb], num_iteration=r, raw_score=True)).astype(np.float32), d["y"][rb], ob, Mb).mean()
        if bestv is None or v > bestv:
            best, bestv = r, v
    outs = [gt + bst.predict(Xt, num_iteration=best, raw_score=True) for Xt, gt in targets]
    return outs, best, bestv - base_in, bst


oof = np.empty(len(dv["y"]))
info = []
for k in range(5):
    te = np.where(fold == k)[0]; tr = np.where(fold != k)[0]
    rt, _, _ = sub(dv, te)
    (s,), best, gin, _ = fit_predict(dv, Xv, tr, seed * 10 + k, [(Xv[rt], dv["g"][rt])])
    oof[rt] = s
    info.append((best, gin))
    print(f"  fold {k}: rounds {best} inner gain {gin:+.5f} ({time.time()-t0:.0f}s)", flush=True)
base = ndcg(dv, dv["g"]); new = ndcg(dv, oof); dl = new - base
print(f"[{tag}] VAL OOF {new.mean():.5f} vs G3 {base.mean():.5f}: delta {dl.mean():+.5f} +- {dl.std()/np.sqrt(len(dl)):.5f}")
np.save(f"{D}/gb_oof_{tag}.npy", oof.astype(np.float32))
# holdout: train on all val
(sh,), best, gin, bst = fit_predict(dv, Xv, np.arange(dv["nf"]), seed * 10 + 9, [(Xh, dh["g"])])
bh = ndcg(dh, dh["g"]); nh = ndcg(dh, sh); dh_ = nh - bh
print(f"[{tag}] HOLD (val-trained, rounds {best}) {nh.mean():.5f} vs {bh.mean():.5f}: delta {dh_.mean():+.5f} +- {dh_.std()/np.sqrt(len(dh_)):.5f}")
np.save(f"{D}/gb_hold_{tag}.npy", sh.astype(np.float32))
imp = bst.feature_importance("gain")
o = np.argsort(-imp)[:25]
print("top gain features:", ", ".join(f"{names[i]} {imp[i]/imp.sum():.3f}" for i in o))
print(f"done {time.time()-t0:.0f}s")
