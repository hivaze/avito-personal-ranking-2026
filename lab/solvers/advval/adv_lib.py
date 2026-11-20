"""Adversarial classifier with user-grouped CV (LightGBM, CPU, <= 16 threads)."""
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from common_av import psi

P_FEED = dict(objective="binary", learning_rate=0.03, num_leaves=15, min_data_in_leaf=100, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, num_threads=16, verbose=-1, seed=7)
P_CAND = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=500, feature_fraction=0.8,
              bagging_fraction=0.7, bagging_freq=1, lambda_l2=10.0, num_threads=16, verbose=-1, seed=7,
              max_bin=127)


def groups_of(user_id, feed_key):
    """GroupKFold groups: user id; user 0 (pooled unknown user) -> one group per feed"""
    user_id = np.asarray(user_id, np.int64)
    g = user_id.copy()
    z = user_id == 0
    g[z] = -(np.asarray(feed_key, np.int64)[z] + 1)
    return g


def run_adv(X, y, groups, names, params, rounds, cat=None, shap_rows=40000, n_folds=5, seed=0):
    """X float32 [n, F], y {0,1}. Returns dict with OOF prob, fold AUCs, pooled AUC, gain and mean|SHAP| per feature."""
    gkf = GroupKFold(n_splits=n_folds)
    oof = np.zeros(len(y))
    aucs, gain, shap = [], np.zeros(len(names)), np.zeros(len(names))
    rng = np.random.default_rng(seed)
    best_iter = []
    for k, (tr, te) in enumerate(gkf.split(X, y, groups)):
        dtr = lgb.Dataset(X[tr], y[tr], feature_name=list(names), categorical_feature=cat or "auto",
                          free_raw_data=True)
        bst = lgb.train(params, dtr, num_boost_round=rounds)
        p = bst.predict(X[te])
        oof[te] = p
        aucs.append(roc_auc_score(y[te], p))
        gain += bst.feature_importance("gain") / max(bst.feature_importance("gain").sum(), 1e-12)
        sr = te if len(te) <= shap_rows else rng.choice(te, shap_rows, replace=False)
        c = bst.predict(X[sr], pred_contrib=True)[:, :-1]
        shap += np.abs(c).mean(0)
    return dict(oof=oof, fold_auc=np.array(aucs), auc=roc_auc_score(y, oof), gain=gain / n_folds,
                shap=shap / n_folds, names=list(names))


def univariate(X, y, names):
    """per-feature PSI (B relative to A) and univariate AUC (|AUC - 0.5|, NaN -> median)"""
    out = {}
    for j, n in enumerate(names):
        a, b = X[y == 0, j], X[y == 1, j]
        x = X[:, j].astype(np.float64)
        xm = np.where(np.isnan(x), np.nanmedian(x) if np.isfinite(np.nanmedian(x)) else 0.0, x)
        try:
            au = roc_auc_score(y, xm)
        except ValueError:
            au = 0.5
        out[n] = dict(psi=psi(a, b), auc1=float(au), mean_a=float(np.nanmean(a)) if np.isfinite(a).any() else np.nan,
                      mean_b=float(np.nanmean(b)) if np.isfinite(b).any() else np.nan)
    return out


def top_table(res, uni, k=12, by="shap"):
    order = np.argsort(res[by])[::-1][:k]
    lines = ["| feature | mean abs SHAP | gain share | PSI | univariate AUC | mean A | mean B |", "|---|---|---|---|---|---|---|"]
    for j in order:
        n = res["names"][j]; u = uni[n]
        lines.append(f"| {n} | {res['shap'][j]:.4f} | {res['gain'][j]:.3f} | {u['psi']:.4f} | {u['auc1']:.3f} | "
                     f"{u['mean_a']:.4g} | {u['mean_b']:.4g} |")
    return "\n".join(lines)
