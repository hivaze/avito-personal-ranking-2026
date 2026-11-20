"""Residual LightGBM over z(v11) with the exact nn/finetune.py folds (GroupKFold(5) by user_id, user 0 = one group).
Objective binary with a per-fold logistic init on z(v11) (stk2 recipe), fixed 300 rounds (no early stopping on val).
Usage: python s04_resid.py <config> [seeds]"""
import os, sys, time
from common_e import meta, ndcg_per_feed, ROOT, SCR, fz
import numpy as np
import polars as pl
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression

cfg = sys.argv[1]
OBJ = os.environ.get("OBJ", "xendcg")
INITC = float(os.environ.get("INITC", "1.0"))
seeds = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "0,1").split(",")]
f, M, off, fid = meta("val")
n = len(M)
y = np.load(f"{ROOT}/data/flat/val/cand_target.npy").astype(np.float64)
s = np.load(f"{SCR}/out/v11_val.npy")
pf = pl.read_parquet(f"{SCR}/out/perfeed.parquet")
Z = np.load(f"{SCR}/out/attrs.npz")


def ld(nm):
    for d in (f"{ROOT}/data/feats/val", f"{ROOT}/data/feats_new/val", f"{ROOT}/eda/errors/synthesis/derived/val"):
        p = f"{d}/{nm}.npy"
        if os.path.exists(p):
            return np.asarray(np.load(p, mmap_mode="r"), dtype=np.float32)
    raise FileNotFoundError(nm)


CAND = ["f1", "f2", "f3", "f4", "f5", "maxsim32", "seq_max_all", "cos_u", "cos_h10", "cat_cnt_all", "cat_share_all", "cat_min_hpos",
        "it_hist_all_luo", "seq_cv_pop", "seq_cv_invd20", "it_tr_ctr", "it_tr_coec", "ctx_nd_max", "ctx_nd_c90", "ctx_nd_c95",
        "feed_cat_share", "centrality", "pre_fe", "pre_new", "pre_hmax32", "cold_item", "ag_age", "seq_argmax_hpos", "cat_cnt_20",
        "ctx_hn_rrw", "ctx_nd_knn_h", "seq_dcos_hl2", "cos_hall"] + [f"cat_cnt_et{k}" for k in range(1, 11)]
FEED = ["M", "H", "hist_n_uniq_cat", "user0", "new_user"] + [f"hist_et{k}_share" for k in range(1, 7)] + [f"hist_es{k}_share" for k in range(0, 5)]
cols, X = [], []
zs = fz(s, fid, M)
X.append(zs.astype(np.float32)); cols.append("z_v11")
rk = np.load(f"{SCR}/out/v11_rank.npy")
if os.environ.get("PCT", "0") == "1":
    X.append((rk / np.maximum(M[fid] - 1, 1)).astype(np.float32)); cols.append("v11_pct")
if cfg in ("cand", "cand_feed"):
    for c in CAND:
        X.append(Z[c] if c in Z.files else ld(c)); cols.append(c)
    X.append(Z["et_am"].astype(np.float32)); cols.append("et_am")
    X.append(Z["es_am"].astype(np.float32)); cols.append("es_am")
if cfg == "cand_feed":
    for c in FEED:
        X.append(Z[c]); cols.append(c)
    X.append(pf["coldshare"].to_numpy()[fid].astype(np.float32)); cols.append("coldshare")
if cfg.startswith("solv"):
    names = {"solv4": ["nn_final4_val_oof", "nn_final4_trainonly_val"],
             "solvall": ["nn_final4_val_oof", "nn_final4_trainonly_val", "pre_final_val", "xgb_x300_luo_val", "nn_final3_val_oof",
                         "nn_final3_trainonly_val", "nn_final5cd_val_oof", "nn_final5cd_trainonly_val", "stk2_res4_clean_val"]}[cfg]
    for nm in names:
        X.append(fz(np.load(f"{ROOT}/preds/{nm}.npy"), fid, M).astype(np.float32)); cols.append("z_" + nm)
X = np.stack(X, 1)
print(cfg, X.shape, flush=True)
uid = f["user_id"].to_numpy().astype(np.int64)
folds = list(GroupKFold(n_splits=5).split(np.arange(n), groups=uid))
infold = np.zeros(n, np.int64)
for k, (_, ho) in enumerate(folds):
    infold[ho] = k
rowfold = infold[fid]
base = ndcg_per_feed(s, y, off, M)
out = {}
for seed in seeds:
    oof = {R: np.zeros(len(y)) for R in (100, 200, 300)}
    t0 = time.time()
    for k in range(5):
        tr = rowfold != k
        te = ~tr
        if OBJ == "binary":
            lr = LogisticRegression(C=1e3).fit(zs[tr, None], y[tr])
            init = lr.decision_function(zs[:, None])
        else:
            init = INITC * zs
        objd = {"binary": dict(objective="binary"), "xendcg": dict(objective="rank_xendcg"),
                "lambdarank": dict(objective="lambdarank", lambdarank_truncation_level=60, lambdarank_norm=True)}[OBJ]
        mono = [1 if c == "z_v11" else 0 for c in cols] if os.environ.get("MONO", "1") == "1" else [0] * len(cols)
        p = dict(**objd, monotone_constraints=mono, monotone_constraints_method="advanced", learning_rate=0.03, num_leaves=15, min_data_in_leaf=500, feature_fraction=0.7,
                 bagging_fraction=0.7, bagging_freq=1, lambda_l2=10.0, num_threads=16, verbose=-1, seed=seed,
                 deterministic=True, force_row_wise=True)
        ds = lgb.Dataset(X[tr], y[tr], init_score=init[tr], free_raw_data=False)
        if OBJ != "binary":
            ds.set_group(M[np.unique(fid[tr])])
        b = lgb.train(p, ds, num_boost_round=300)
        for R in (100, 200, 300):
            oof[R][te] = init[te] + b.predict(X[te], num_iteration=R, raw_score=True)
        if k == 0 and seed == seeds[0]:
            imp = sorted(zip(b.feature_importance("gain"), cols), reverse=True)[:15]
            print("  fold0 top gain:", [(c, round(g / 1e3, 1)) for g, c in imp], flush=True)
    for R in (100, 200, 300):
        out[(seed, R)] = ndcg_per_feed(oof[R], y, off, M)
    np.save(f"{SCR}/out/resid_{cfg}_{OBJ}_s{seed}.npy", oof[300].astype(np.float32))
    print(f"  seed {seed} done {time.time() - t0:.0f}s", flush=True)
ts = f["timestamp"].to_numpy()
day = pf["day"].to_numpy()
sl = {"all": np.ones(n, bool), "M<50": M < 50, "M>=50": M >= 50, "Tue": day == 0, "Thu": day == 1,
      "all_cold": pf["poscold"].to_numpy() == "all_cold"}
for R in (100, 200, 300):
    d = np.mean([out[(sd, R)] - base for sd in seeds], 0)
    per = [round(float((out[(sd, R)] - base).mean()), 5) for sd in seeds]
    print(f" {cfg} {OBJ} c{INITC} R={R}: " + "  ".join(f"{k} {d[m].mean():+.5f}+-{d[m].std() / np.sqrt(m.sum()):.5f}" for k, m in sl.items()) + f"  per-seed {per}")
