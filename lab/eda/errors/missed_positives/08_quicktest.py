"""Quick paired test of new feature groups in the residual stacker (src/stack2.py protocol, leak-free reporting).

Base margin = fitted affine map of z(nn_final2_trainonly) (no val labels in the base). XGBoost binary:logistic on val,
5-fold GroupKFold by user (user 0 split by feed), same folds for every config within a seed. Baseline features =
stk2_res2_r3's compact list minus leaky ALS and minus files that no longer exist. Each config = baseline + one group.
Reports: OOF NDCG on leak-free feeds, paired delta vs baseline (mean over seeds of per-feed deltas) with SE, the Thu half,
and the v5-style blend 0.5 z(nn_final2_val_oof) + 0.5 z(stacker).
GPU: CUDA_VISIBLE_DEVICES=N only.
"""
import os, sys, json, time, argparse
sys.path.insert(0, "<repo>/src")
from common import *
from stack_val import feed_rank_z
from sklearn.model_selection import GroupKFold
import xgboost as xgb

D = Path("<repo>/eda/errors/missed_positives")
PREDS = Path("<repo>/preds")
SCORES = ["nn_final2_trainonly", "cb_tr300k_none", "lgb_tr300k_none", "xgb_rand50"]
BASE_FEATS = ["f1", "f2", "f3", "f4", "f5", "M", "H", "category_id", "maxsim32", "cos_u", "cos_h10", "cat_cnt_20", "emb_missing",
              "it_hist_all_luo", "it_hist_et1_luo", "it_hist_all_luo_rk", "seq_cv_fwd5", "seq_cv_invd20", "seq_cv_invd20_rk",
              "seq_cv_invd20_z", "seq_cv_norm", "seq_cv_pop", "seq_cv_seqs", "seq_cv_t1", "seq_cv_w5", "ctx_nd_c80", "ctx_nd_c90",
              "ctx_nd_c95", "ctx_nd_dupb_f1", "ctx_nd_dupb_h", "ctx_nd_dupb_u", "ctx_nd_hcent", "ctx_nd_knn_f1", "ctx_nd_knn_h",
              "ctx_nd_knn_h_dev", "ctx_nd_knn_u", "ctx_nd_knn_u_dev", "ctx_nd_max", "ctx_nd_mean", "ctx_nd_top5h", "ctx_nd_top5u",
              "ctx_nd_ucent"]
GROUPS = {
    "base": [],
    "G1_catconc": ["d_cat_run", "d_cat_run_e1", "d_cat_pur20", "d_cat_pur5", "d_top_cat20", "d_cat_last1", "d_cat_last3cnt",
                   "feed_cat_share", "d_ncat_feed", "cat_share_all", "cat_min_hpos", "cat_cnt_all"],
    "G2_supply": ["clu_ccos1024", "trd_skde_s6h", "trd_kde_sday", "d_dup95_h", "d_dup90_h", "d_dup95_hmin"],
    "G3_novel": ["d_f3_resid", "clu_hmin1024", "f3_rk", "f1_rk"],
    "G4_pre": ["pre_v3_fe", "pre_v3_new", "pre_v3_fe_noid", "pre_v3_hmax32", "pre_v3_id_contrib"],
}


def path_of(n, split):
    p = FEATS / split / f"{n}.npy"
    return p if p.exists() else D / "derived" / split / f"{n}.npy"


def load_cols(names, split):
    return np.stack([np.load(path_of(n, split)).astype(np.float32) for n in names], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", default="base,G1_catconc,G2_supply,G3_novel,G4_pre")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--rounds", type=int, default=300)
    ap.add_argument("--tag", default="qt")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    t0 = time.time()
    f, M, off = feed_meta("val")
    fr = np.load(D / "frame.npz")
    leakf = fr["leakf"]
    y = np.load(FLAT / "val/cand_target.npy").astype(np.float32)
    fidx = np.load(FLAT / "val/cand_fidx.npy")
    base_feats = [n for n in BASE_FEATS if path_of(n, "val").exists()]
    print("baseline feats used:", len(base_feats), "missing:", [n for n in BASE_FEATS if n not in base_feats], flush=True)
    cols = []
    for s in SCORES:
        rz = feed_rank_z(off, M, np.load(PREDS / f"{s}_val.npy").astype(np.float32))
        cols += [rz[:, 1:2], rz[:, 0:1]]
    Xs = np.hstack(cols + [load_cols(base_feats, "val")])
    names0 = [f"{s}_{k}" for s in SCORES for k in ("z", "rk")] + base_feats
    zv = feed_rank_z(off, M, np.load(PREDS / "nn_final2_trainonly_val.npy").astype(np.float32))[:, 1].astype(np.float64)
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(C=1e6).fit(zv[:, None], y)
    bm = float(lr.coef_[0, 0]) * zv + float(lr.intercept_[0])
    users = f["user_id"].to_numpy().copy()
    users[users == 0] = -np.arange(1, (users == 0).sum() + 1)
    p = dict(objective="binary:logistic", device=args.device, tree_method="hist", eta=0.02, max_depth=4, min_child_weight=20,
             subsample=0.8, colsample_bytree=0.8, reg_lambda=10, max_bin=256, nthread=16)
    z_oof = feed_rank_z(off, M, np.load(PREDS / "nn_final2_val_oof.npy").astype(np.float32))[:, 1]
    ts = f["timestamp"].to_numpy()
    thu = ts >= np.median(ts)  # val = Tue + Thu; later half
    kf = ~leakf
    res = {}
    per_feed = {}
    for g in args.groups.split(","):
        extra = GROUPS[g]
        X = np.hstack([Xs, load_cols(extra, "val")]) if extra else Xs
        names = names0 + extra
        pers, pers_bl = [], []
        oof_all = np.zeros(len(y))
        for seed in range(args.seeds):
            p["seed"] = seed
            oof = np.zeros(len(y))
            for trf, vaf in GroupKFold(n_splits=5, shuffle=True, random_state=seed).split(np.arange(len(M)), groups=users):
                tr = np.isin(fidx, trf)
                dtr = xgb.DMatrix(X[tr], y[tr], feature_names=names, base_margin=bm[tr])
                b = xgb.train(p, dtr, args.rounds)
                oof[~tr] = b.predict(xgb.DMatrix(X[~tr], feature_names=names, base_margin=bm[~tr]), output_margin=True)
            per = ndcg_per_feed(oof, y, off, M)
            zs = feed_rank_z(off, M, oof.astype(np.float32))[:, 1]
            perb = ndcg_per_feed(0.5 * z_oof + 0.5 * zs, y, off, M)
            pers.append(per); pers_bl.append(perb)
            oof_all += oof / args.seeds
            print(f"  {g} seed {seed}: stacker {per[kf].mean():.5f}  blend {perb[kf].mean():.5f}  ({time.time() - t0:.0f}s)", flush=True)
        per_feed[g] = (np.mean(pers, 0), np.mean(pers_bl, 0), [x[kf].mean() for x in pers], [x[kf].mean() for x in pers_bl])
        np.save(D / f"qt_{args.tag}_{g}_val.npy", oof_all.astype(np.float32))
    bl, blb, _, _ = per_feed["base"]
    print("\n=== results (leak-free feeds; paired deltas vs base, per-feed mean over seeds) ===")
    for g, (pf_, pfb, s_st, s_bl) in per_feed.items():
        d = pf_[kf] - bl[kf]; db = pfb[kf] - blb[kf]
        dthu = (pf_ - bl)[kf & thu]
        r = dict(stacker=float(pf_[kf].mean()), blend=float(pfb[kf].mean()), d_stacker=float(d.mean()), se=float(d.std() / np.sqrt(len(d))),
                 d_blend=float(db.mean()), se_blend=float(db.std() / np.sqrt(len(db))), d_thu=float(dthu.mean()),
                 se_thu=float(dthu.std() / np.sqrt(len(dthu))), per_seed_stacker=[round(x, 5) for x in s_st],
                 per_seed_blend=[round(x, 5) for x in s_bl])
        res[g] = r
        print(f"{g:12s} stacker {r['stacker']:.5f} d {r['d_stacker']:+.5f} ± {r['se']:.5f} | Thu d {r['d_thu']:+.5f} ± {r['se_thu']:.5f} | "
              f"blend {r['blend']:.5f} d {r['d_blend']:+.5f} ± {r['se_blend']:.5f} | seeds {r['per_seed_stacker']}")
    (D / f"qt_{args.tag}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
