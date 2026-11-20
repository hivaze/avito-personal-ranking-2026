"""audit 3a: representation test, stage 1. LightGBM (the student's params) on the 177 cb26 tabular features -> the W16 train-only pool
score (per-feed z). No label is used: the target is the NN score on test / val / holdout feeds, where every W16 member is out of sample.
3-way cross-fit over the three feed sets: model for H trains on T+V, for V on T+H, for T on V+H. Saves out/s1_<set>.npy (raw prediction).
CPU only, 12 threads."""
import time
from au_common import *
sys.path.insert(0, str(ROOT / "nn_x/cb26"))
import lightgbm as lgb
from cbdata import load_tab
from cb_member import member_features
t0 = time.time(); log = lambda *a: print(f"[{time.time()-t0:6.0f}s]", *a, flush=True)
ROUNDS = int(sys.argv[1]) if len(sys.argv) > 1 else 800
m = Meta(); names = member_features(); log(len(names), "features")
SP = dict(V="val", H="hold", T="test"); TAB = dict(V="val", H="trH", T="test")
X = {k: load_tab(TAB[k], names) for k in SP}; Z = {k: w16_pool(m, SP[k]) for k in SP}
log({k: X[k].shape for k in X})
P = dict(objective="regression", learning_rate=0.1, num_leaves=255, min_data_in_leaf=100, feature_fraction=0.5, bagging_fraction=0.5,
         bagging_freq=1, lambda_l2=10, max_bin=255, num_threads=12, verbose=-1, seed=0)


def feedcorr(a, b, off, M):
    """mean over feeds of the within-feed Pearson correlation."""
    fid = np.repeat(np.arange(len(M)), M); n = M.astype(np.float64)
    ma = np.bincount(fid, a, len(M)) / n; mb = np.bincount(fid, b, len(M)) / n
    da = a - ma[fid]; db = b - mb[fid]
    num = np.bincount(fid, da * db, len(M)); den = np.sqrt(np.bincount(fid, da * da, len(M)) * np.bincount(fid, db * db, len(M)))
    return float(np.mean(num[den > 0] / den[den > 0]))


res = {}
for tgt in ("H", "V", "T"):
    tr = [k for k in SP if k != tgt]
    d = lgb.Dataset(np.concatenate([X[k] for k in tr]), np.concatenate([Z[k] for k in tr]), feature_name=names, categorical_feature=["category_id"], free_raw_data=True)
    sp = SP[tgt]; off, M = m.off[sp], m.M[sp]; res[tgt] = {}
    def cb(env, tgt=tgt, sp=sp, off=off, M=M):
        it = env.iteration + 1
        if it in (200, 400, 600) or it == ROUNDS:
            p = env.model.predict(X[tgt], num_iteration=it, num_threads=12).astype(np.float32)
            zp = m.z(sp, p); r = dict(rho=corr(zp, Z[tgt]), feedcorr=feedcorr(p.astype(np.float64), Z[tgt].astype(np.float64), off, M),
                                      r2=float(1 - np.mean((Z[tgt] - p) ** 2) / np.mean(Z[tgt] ** 2)))
            if sp in m.y: r["ndcg"] = float(ndcg_fast(p, m.y[sp], off, M).mean()); r["ndcg_w16"] = float(ndcg_fast(Z[tgt], m.y[sp], off, M).mean())
            res[tgt][it] = r; log(f"target set {tgt} (train {'+'.join(tr)}) round {it}: {json.dumps({k: round(v, 5) for k, v in r.items()})}")
            if it == ROUNDS: np.save(OUT / f"s1_{tgt}.npy", p)
    b = lgb.train(P, d, ROUNDS, callbacks=[cb])
    if tgt == "V":
        imp = sorted(zip(b.feature_importance("gain"), names), reverse=True)[:25]; tot = b.feature_importance("gain").sum()
        res["imp_V"] = [(n, float(g / tot)) for g, n in imp]; log("top gain share:", [(n, round(float(g / tot), 3)) for g, n in imp[:12]])
    json.dump(res, open(OUT / "s1.json", "w"), indent=1)
log("DONE")
