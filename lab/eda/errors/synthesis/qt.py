"""Residual-stacker quick test for the error-analysis synthesis (src/stack2.py protocol, leak-free reporting).

Base margin = affine map (2 params, LR on val) of z(nn_final3_trainonly), which never saw val labels.
Scores: nn_final3_trainonly + xgb_x300_luo (z and rank). Base features: stk2_res4_clean's compact list (files that still exist).
XGBoost binary:logistic, CPU 16 threads, 5-fold GroupKFold by user (user 0 split by feed), seeds 0..S-1 (same folds for every
config). Each config = base + one group. Saves per-feed NDCG (mean over seeds) of the stacker and of the v5-style blend
0.5 z(nn_final3_val_oof) + 0.5 z(stacker) to out/qt_<group>.npz; compare.py makes paired deltas on the 14,306 leak-free feeds.
"""
import os, sys, json, time, argparse
os.environ.setdefault("OMP_NUM_THREADS", "16")
sys.path.insert(0, "<repo>/src")
from common import *
from stack_val import feed_rank_z
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

SNAP = __import__("pathlib").Path("<repo>/eda/errors/synthesis/snap")
def pp(fn):
    import pathlib
    return SNAP / fn if (SNAP / fn).exists() else pathlib.Path("<repo>/preds") / fn


D = Path("<repo>/eda/errors/synthesis")
PREDS = Path("<repo>/preds")
BASE_PRED = "nn_final3_trainonly"
OOF_PRED = "nn_final3_val_oof"
SCORES = ["nn_final3_trainonly", "xgb_x300_luo"]
RES4 = ("f1,f2,f3,f4,f5,M,H,category_id,maxsim32,cos_u,cos_h10,cat_cnt_20,emb_missing,als_exp5,als_exp10,als_last10,"
        "als_samecat_exp10,als_has,it_hist_all_luo,it_hist_et1_luo,it_hist_all_luo_rk,seq_cv_invd20,seq_cv_invd20_rk,seq_cv_norm,"
        "seq_cv_pop,ctx_nd_hcent,ctx_nd_knn_f1,ctx_nd_knn_h,ctx_nd_knn_h_dev,ctx_nd_knn_u,ctx_nd_knn_u_dev,ctx_nd_max,ctx_nd_top5h,"
        "ctx_nd_top5u,ctx_nd_ucent").split(",")
GROUPS = {
    "base": [],
    # existing signals recommended by lenses (not new information, but untested on the nn_final3 base)
    "E_pre3A": ["pre_v3A_fe", "pre_v3A_new", "pre_v3A_fe_noid", "pre_v3A_hmax32", "pre_v3A_idmax32"],
    "E_ctxhs": ["ctx_hs_d7", "ctx_hs_e1", "ctx_hs_e1k64", "ctx_hs_k128", "ctx_hs_k128t10", "ctx_hs_k256", "ctx_hs_mutual", "ctx_hs_t10", "seq_top3"],
    "E_catint": ["catint_S1"],
    "E_ittr": ["it_tr_coec", "it_tr_ctr", "it_tr_pos", "it_cand_cnt_train"],
    # new: item taste profiles from train labels (01_taste.py)
    "N_taste_u": ["tp_n", "tp_c", "tp_coec", "tp_cosC_u", "tp_cosS_u", "tp_dC_u", "tp_sur_u"],
    "N_taste_h": ["tp_n", "tp_c", "tp_coec", "tp_cosC_h", "tp_cosS_h", "tp_dC_h", "tp_sur_h"],
    "N_taste": ["tp_n", "tp_c", "tp_coec", "tp_cosC_u", "tp_cosS_u", "tp_dC_u", "tp_sur_u", "tp_cosC_h", "tp_cosS_h", "tp_dC_h",
                "tp_sur_h"],
}


def path_of(n, split):
    sfx = "val" if split == "val" else "test"
    for p in (FEATS / split / f"{n}.npy", Path("<repo>/nn/feat_cache") / split / f"{n}.npy",
              D / "derived" / split / f"{n}.npy", pp(f"{n}_{sfx}.npy"),
              Path("<repo>/eda/errors/category_intent/out") / f"{n}_{sfx}.npy"):
        if p.exists():
            return p
    return None


def load_cols(names, split):
    return np.stack([np.load(path_of(n, split)).astype(np.float32) for n in names], 1)


def leak_mask():
    """True for the val feeds whose non-zero user has a later feed in val or test."""
    fv = pl.read_parquet(FLAT / "val/feeds.parquet").select("fidx", "user_id", "timestamp")
    ft = pl.read_parquet(FLAT / "test/feeds.parquet").select("user_id", "timestamp")
    allf = pl.concat([fv.select("user_id", "timestamp"), ft]).rename({"timestamp": "ts2"})
    j = fv.filter(pl.col("user_id") != 0).join(allf, on="user_id").filter(pl.col("ts2") > pl.col("timestamp"))
    m = np.zeros(len(fv), bool)
    m[j["fidx"].unique().to_numpy()] = True
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="base")
    ap.add_argument("--extra", default="", help="comma list of feature names instead of a named group")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--rounds", type=int, default=300)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    tag = args.tag or args.group
    t0 = time.time()
    f, M, off = feed_meta("val")
    y = np.load(FLAT / "val/cand_target.npy").astype(np.float32)
    fidx = np.load(FLAT / "val/cand_fidx.npy")
    leak = leak_mask(); kf = ~leak
    base_feats = [n for n in RES4 if path_of(n, "val") is not None]
    extra = [x for x in args.extra.split(",") if x] if args.extra else GROUPS[args.group]
    miss = [n for n in extra if path_of(n, "val") is None]
    assert not miss, f"missing features {miss}"
    cols, names = [], []
    for s in SCORES:
        rz = feed_rank_z(off, M, np.load(pp(f"{s}_val.npy")).astype(np.float32))
        cols += [rz[:, 1:2], rz[:, 0:1]]; names += [f"{s}_z", f"{s}_rk"]
    X = np.hstack(cols + [load_cols(base_feats + extra, "val")])
    names += base_feats + extra
    zv = feed_rank_z(off, M, np.load(pp(f"{BASE_PRED}_val.npy")).astype(np.float32))[:, 1].astype(np.float64)
    lr = LogisticRegression(C=1e6).fit(zv[:, None], y)
    bm = float(lr.coef_[0, 0]) * zv + float(lr.intercept_[0])
    users = f["user_id"].to_numpy().copy()
    users[users == 0] = -np.arange(1, (users == 0).sum() + 1)
    p = dict(objective="binary:logistic", device="cpu", tree_method="hist", eta=0.02, max_depth=4, min_child_weight=20,
             subsample=0.8, colsample_bytree=0.8, reg_lambda=10, max_bin=256, nthread=16)
    z_oof = feed_rank_z(off, M, np.load(pp(f"{OOF_PRED}.npy")).astype(np.float32))[:, 1]
    pers, persb, oof_all = [], [], np.zeros(len(y))
    gain = {}
    for seed in range(args.seeds):
        p["seed"] = seed
        oof = np.zeros(len(y))
        for trf, vaf in GroupKFold(n_splits=5, shuffle=True, random_state=seed).split(np.arange(len(M)), groups=users):
            tr = np.isin(fidx, trf)
            b = xgb.train(p, xgb.DMatrix(X[tr], y[tr], feature_names=names, base_margin=bm[tr]), args.rounds)
            oof[~tr] = b.predict(xgb.DMatrix(X[~tr], feature_names=names, base_margin=bm[~tr]), output_margin=True)
            for k, v in b.get_score(importance_type="total_gain").items():
                gain[k] = gain.get(k, 0) + v
        per = ndcg_per_feed(oof, y, off, M)
        zs = feed_rank_z(off, M, oof.astype(np.float32))[:, 1]
        perb = ndcg_per_feed(0.5 * z_oof + 0.5 * zs, y, off, M)
        pers.append(per); persb.append(perb); oof_all += oof / args.seeds
        print(f"{tag} seed {seed}: stacker {per[kf].mean():.5f} blend {perb[kf].mean():.5f} ({time.time() - t0:.0f}s)", flush=True)
    (D / "out").mkdir(exist_ok=True)
    np.savez(D / "out" / f"qt_{tag}.npz", per=np.array(pers), perb=np.array(persb), leak=leak)
    np.save(D / "out" / f"qt_{tag}_oof.npy", oof_all.astype(np.float32))
    tot = sum(gain.values())
    top = sorted(gain.items(), key=lambda kv: -kv[1])
    print("gain share of extra feats:", {k: round(v / tot, 4) for k, v in top if k in extra})
    print("top-10 gain:", [(k, round(v / tot, 3)) for k, v in top[:10]])
    print(json.dumps(dict(tag=tag, feats=len(names), stacker=[float(x[kf].mean()) for x in pers],
                          blend=[float(x[kf].mean()) for x in persb], time=round(time.time() - t0))))


if __name__ == "__main__":
    main()
