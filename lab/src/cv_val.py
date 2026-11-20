"""5-fold GroupKFold (by user) LightGBM lambdarank on val only. Reports OOF full NDCG."""
import sys, time, json
sys.path.insert(0, "<repo>/src")
from common import *
import lightgbm as lgb
from sklearn.model_selection import GroupKFold


def run(names, params=None, rounds=3000, nfold=5, seed=0, verbose=False):
    f, M, off = feed_meta("val")
    X = load_feats("val", names)
    y = np.load(FLAT / "val/cand_target.npy").astype(np.float32)
    fidx = np.load(FLAT / "val/cand_fidx.npy")
    users = f["user_id"].to_numpy()
    p = dict(objective="lambdarank", metric="ndcg", eval_at=[1000], learning_rate=0.03, num_leaves=63,
             min_data_in_leaf=100, feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10,
             lambdarank_truncation_level=1000, num_threads=48, verbose=-1, seed=seed)
    if params: p.update(params)
    oof = np.zeros(len(y))
    gkf = GroupKFold(n_splits=nfold)
    best_its = []
    for k, (tr_f, va_f) in enumerate(gkf.split(np.arange(len(M)), groups=users)):
        tr_f.sort(); va_f.sort()
        tr_rows = np.isin(fidx, tr_f); va_rows = ~tr_rows
        dtr = lgb.Dataset(X[tr_rows], y[tr_rows], group=M[tr_f], feature_name=names, categorical_feature=[n for n in names if n == "category_id"])
        dva = lgb.Dataset(X[va_rows], y[va_rows], group=M[va_f], reference=dtr)
        b = lgb.train(p, dtr, rounds, valid_sets=[dva], callbacks=[lgb.early_stopping(200, verbose=False)])
        best_its.append(b.best_iteration)
        oof[va_rows] = b.predict(X[va_rows], num_iteration=b.best_iteration)
    score = ndcg(oof, y, off, M)
    return score, best_its, oof, b


if __name__ == "__main__":
    names = [n for n in all_feature_names() if not n.startswith("u_het") or True]
    t = time.time()
    s, its, oof, b = run(names)
    print(f"ALL feats ({len(names)}): OOF NDCG={s:.4f} its={its} {time.time()-t:.0f}s")
    imp = sorted(zip(b.feature_importance("gain"), names), reverse=True)
    for g, n in imp[:40]: print(f"{g:12.0f} {n}")
