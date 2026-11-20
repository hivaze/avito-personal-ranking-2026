"""audit common: W16 train-only pool z on val / holdout / test, tree-model registry, paired stats. CPU only, read-only on every
prediction file. z = per-feed score z (stack_val.feed_rank_z[:,1], the project's member score), rho = corr of the z arrays
(= the project's 'wcorr'); a Spearman variant (per-feed rank pct) is reported next to it."""
import os, sys, json
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "POLARS_MAX_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[k] = "12"
sys.path.insert(0, "<repo>/src"); sys.path.insert(0, "<repo>/nn_x/w12/recheck/pool_composition")
import numpy as np
from pathlib import Path
from pc_common import Meta, ndcg_fast, feed_rank_z   # pc_common sets the same env caps
ROOT = Path("<repo>"); W = ROOT / "nn_x/w5/preds"; HERE = ROOT / "nn_x/w12/cb/audit"; CACHE = HERE / "cache"; OUT = HERE / "out"
W16 = [f"w5_e10_kd05w192_s{s}" for s in range(951, 959)] + [f"w5_e10_kd05t25w192_s{s}" for s in range(951, 959)]
LB = ROOT / "nn_x/w12/recheck/long_bets/out"


def w16_paths(v):
    return dict(val=W / f"nn_{v}_trainonly_val.npy", hold=W / f"holdout/nn_{v}_holdout.npy", test=W / f"nn_{v}_trainonly_test.npy")


def w16_pool(meta, sp, members=False):
    """member-equal mean of per-feed z of the 16 train-only students (cached)."""
    f = CACHE / f"w16_{sp}_z.npy"; fm = CACHE / f"w16_{sp}_members.npy"
    if f.exists() and (not members or fm.exists()):
        return (np.load(f), np.load(fm, mmap_mode="r")) if members else np.load(f)
    Z = np.stack([meta.z(sp, np.load(w16_paths(v)[sp])) for v in W16], 0)
    np.save(f, Z.mean(0).astype(np.float32)); np.save(fm, Z.astype(np.float32))
    return (np.load(f), np.load(fm, mmap_mode="r")) if members else np.load(f)


# tree models: name -> dict(val=, hold=, test=) paths (missing key = not on disk); note = training population
TREES = {
    "cb26_e3_xm_yeti (CatBoost YetiRank, 134K xm, hard)": dict(val=ROOT / "nn_x/cb26/data/e3_xm_yeti_val.npy", hold=ROOT / "nn_x/cb26/data/e3_xm_yeti_trH.npy", test=ROOT / "nn_x/cb26/data/e3_xm_yeti_test.npy"),
    "w25_A2c (CatBoost YetiRank, 134K xm, 239 feats, hard)": dict(val=ROOT / "nn_x/w25/out/A2c_cb26_val.npy", hold=ROOT / "nn_x/w25/out/A2c_cb26_trH.npy", test=ROOT / "nn_x/w25/out/A2c_cb26_test.npy"),
    "w25_A1_base_xm (CatBoost, 485K feeds, hard)": dict(val=ROOT / "nn_x/w25/out/A1_base_xm_val.npy", hold=ROOT / "nn_x/w25/out/A1_base_xm_trH.npy", test=ROOT / "nn_x/w25/out/A1_base_xm_test.npy"),
    "w25_A1_base_xm_s1 (seed 1)": dict(val=ROOT / "nn_x/w25/out/A1_base_xm_s1_val.npy", hold=ROOT / "nn_x/w25/out/A1_base_xm_s1_trH.npy", test=ROOT / "nn_x/w25/out/A1_base_xm_s1_test.npy"),
    "w25_A2_all_xm (CatBoost, 485K feeds, 239 feats, hard)": dict(val=ROOT / "nn_x/w25/out/A2_all_xm_val.npy", hold=ROOT / "nn_x/w25/out/A2_all_xm_trH.npy", test=ROOT / "nn_x/w25/out/A2_all_xm_test.npy"),
    "lgb_H_xm_134118 (LightGBM lambdarank r600, hard)": dict(val=LB / "gbdt_H_xm_134118_val.npy", hold=LB / "gbdt_H_xm_134118_trH.npy"),
    "lgb_H_xm_60000 (LightGBM lambdarank r600, hard)": dict(val=LB / "gbdt_H_xm_60000_val.npy", hold=LB / "gbdt_H_xm_60000_trH.npy"),
    "lgb_K_xm_60000 (LightGBM student, teacher_mean11, r600)": dict(val=LB / "gbdt_K_xm_60000_val.npy", hold=LB / "gbdt_K_xm_60000_trH.npy"),
    "lgb_K_xm_134118 (LightGBM student, teacher_mean11, r600)": dict(val=LB / "gbdt_K_xm_134118_val.npy", hold=LB / "gbdt_K_xm_134118_trH.npy"),
    "lgb_X_xm_60000 (student, cross-fitted teacher)": dict(val=LB / "gbdt_X_xm_60000_val.npy", hold=LB / "gbdt_X_xm_60000_trH.npy"),
    "lgb_Y_xm_60000 (student, teacher + 1.0 y)": dict(val=LB / "gbdt_Y_xm_60000_val.npy", hold=LB / "gbdt_Y_xm_60000_trH.npy"),
    "lgb_CY_xm_60000 (student, teacher + y + co-click)": dict(val=LB / "gbdt_CY_xm_60000_val.npy", hold=LB / "gbdt_CY_xm_60000_trH.npy"),
    "archive xgb_tr1m_ndcg (XGBoost rank:ndcg, 1M train feeds)": dict(val=ROOT / "archive/preds/xgb_tr1m_ndcg_val.npy", test=ROOT / "archive/preds/xgb_tr1m_ndcg_test.npy"),
    "archive cb_tr300k_luo (CatBoost, 300K train feeds)": dict(val=ROOT / "archive/preds/cb_tr300k_luo_val.npy", test=ROOT / "archive/preds/cb_tr300k_luo_test.npy"),
    "archive cb_tr500k_pw (CatBoost YetiRankPairwise, 500K)": dict(val=ROOT / "archive/preds/cb_tr500k_pw_val.npy", test=ROOT / "archive/preds/cb_tr500k_pw_test.npy"),
    "archive xgb_x300_luo (XGBoost, 300K exploration)": dict(val=ROOT / "archive/preds/xgb_x300_luo_val.npy", test=ROOT / "archive/preds/xgb_x300_luo_test.npy"),
    "archive lgb_tr300k_feat (LightGBM, 300K train feeds)": dict(val=ROOT / "archive/preds/lgb_tr300k_feat_val.npy", test=ROOT / "archive/preds/lgb_tr300k_feat_test.npy"),
}

se = lambda d: float(np.std(d) / np.sqrt(len(d)))
fmt = lambda d: f"{np.mean(d):+.5f} +- {se(d):.5f}"


def rank_pct(meta, sp, x):
    return feed_rank_z(meta.off[sp], meta.M[sp], np.asarray(x, np.float32))[:, 0]


def corr(a, b):
    return float(np.corrcoef(np.asarray(a, np.float64), np.asarray(b, np.float64))[0, 1])


def light(n=2):
    """cap numba + BLAS threads for light analysis scripts that run next to two heavy jobs."""
    import numba, threadpoolctl
    numba.set_num_threads(n); threadpoolctl.threadpool_limits(n)


def clogit(X, y, off, M, iters=25):
    """listwise softmax (conditional logit, target y / n_pos per feed). Returns beta, sandwich SE (feeds as clusters), mean loss per feed."""
    X = np.asarray(X, np.float64); n, p = X.shape; fid = np.repeat(np.arange(len(M)), M)
    npos = np.bincount(fid, y, len(M)); keep = npos[fid] > 0; t = np.where(keep, y / np.maximum(npos[fid], 1), 0.0)
    nf = int((npos > 0).sum()); b = np.zeros(p)
    for _ in range(iters):
        s = X @ b; mx = np.maximum.reduceat(s, off); e = np.exp(s - mx[fid]); den = np.add.reduceat(e, off); pr = np.where(keep, e / den[fid], 0.0)
        G = np.stack([np.bincount(fid, (pr - t) * X[:, j], len(M)) for j in range(p)], 1)           # per-feed gradient
        mu = np.stack([np.bincount(fid, pr * X[:, j], len(M)) for j in range(p)], 1)
        Hs = np.zeros((p, p))
        for i in range(p):
            for j in range(p):
                Hs[i, j] = (pr * X[:, i] * X[:, j]).sum() - (mu[:, i] * mu[:, j]).sum()
        step = np.linalg.solve(Hs, G.sum(0)); b = b - step
        if np.abs(step).max() < 1e-9: break
    Hi = np.linalg.inv(Hs); V = Hi @ (G.T @ G) @ Hi
    s = X @ b; mx = np.maximum.reduceat(s, off); lse = mx + np.log(np.add.reduceat(np.exp(s - mx[fid]), off))
    loss = -(np.bincount(fid, t * s, len(M)) - np.where(npos > 0, lse, 0.0))
    return b, np.sqrt(np.diag(V)), loss  # loss per feed (0 for feeds without positives)
