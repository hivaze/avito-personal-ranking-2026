# Second-stage stacker fitted on val with 5 folds grouped by user. Inputs are scores of models that
# never saw val labels (train-only neural ensembles, trees fitted on train), each as a per-feed
# z-score and rank, plus a short list of features. With --base the neural score enters as a fixed
# base margin and XGBoost learns only the residual.
# How it was run: arguments as logged in lab/logs/res4_clean.log: --scores
# nn_final2_trainonly,xgb_x300_luo --feats <42 names> --objective binary:logistic --eta 0.02
# --depth 4 --mcw 20 --rounds 300 --seeds 3 --base nn_final2_trainonly --tag res4_clean.
# What it produced: preds/stk2_<tag>_{val,test}.npy and a json; out-of-fold val NDCG 0.38491 for
# res4_clean (results/residual_stacker_runs.csv).
# Kept as it ran: paths and names refer to the lab tree (see lab/ and docs/glossary.md).
# Original: lab/src/stack2.py
"""Compact stage-2 stacker on val (GroupKFold by user), XGBoost on GPU.

Inputs: scores of models that never saw val labels (train-only NN ensembles, train-fit GBDTs), each as raw / per-feed z /
per-feed rank, plus a short list of handcrafted features. Outputs OOF val + fold-averaged test predictions.
"""
import argparse, json, sys, time
sys.path.insert(0, "<repo>/src")
from common import *
from stack_val import feed_rank_z
from sklearn.model_selection import GroupKFold
import xgboost as xgb

PREDS = Path("<repo>/preds")


def build(split, scores, feats):
    f, M, off = feed_meta(split)
    sfx = "val" if split == "val" else "test"
    cols, names = [], []
    for s in scores:
        x = np.load(PREDS / f"{s}_{sfx}.npy").astype(np.float32)
        rz = feed_rank_z(off, M, x)
        cols += [rz[:, 1:2], rz[:, 0:1]]; names += [f"{s}_z", f"{s}_rk"]
    if feats:
        cols.append(load_feats(split, feats)); names += feats
    return np.hstack(cols), names, f, M, off


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True, help="comma-separated pred names (preds/<name>_{val,test}.npy)")
    ap.add_argument("--feats", default="", help="comma-separated feature names")
    ap.add_argument("--objective", default="binary:logistic")
    ap.add_argument("--eta", type=float, default=0.03)
    ap.add_argument("--depth", type=int, default=5)
    ap.add_argument("--mcw", type=float, default=20)
    ap.add_argument("--rounds", type=int, default=400)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--base", default="", help="pred name used as base margin (residual stacking), e.g. nn_final_trainonly")
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    t0 = time.time()
    scores = args.scores.split(","); feats = [x for x in args.feats.split(",") if x]
    Xv, names, f, M, off = build("val", scores, feats)
    Xt, _, ft, Mt, offt = build("test", scores, feats)
    y = np.load(FLAT / "val/cand_target.npy").astype(np.float32)
    fidx = np.load(FLAT / "val/cand_fidx.npy")
    users = f["user_id"].to_numpy().copy()
    users[users == 0] = -np.arange(1, (users == 0).sum() + 1)  # pooled unknown user: split by feed
    p = dict(objective=args.objective, device="cuda", tree_method="hist", eta=args.eta, max_depth=args.depth,
             min_child_weight=args.mcw, subsample=0.8, colsample_bytree=0.8, reg_lambda=10, max_bin=256)
    if args.objective.startswith("rank:"):
        p.update(lambdarank_pair_method="topk", lambdarank_num_pair_per_sample=300, lambdarank_score_normalization=False,
                 ndcg_exp_gain=False)
    bm_v = bm_t = None
    if args.base:
        zv = feed_rank_z(off, M, np.load(PREDS / f"{args.base}_val.npy").astype(np.float32))[:, 1].astype(np.float64)
        zt = feed_rank_z(offt, Mt, np.load(PREDS / f"{args.base}_test.npy").astype(np.float32))[:, 1].astype(np.float64)
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression(C=1e6).fit(zv[:, None], y)  # 2 parameters: scale and intercept of the base logit
        a, b0 = float(lr.coef_[0, 0]), float(lr.intercept_[0])
        if args.objective.startswith("rank:"):
            b0 = 0.0
        bm_v, bm_t = a * zv + b0, a * zt + b0
        print(f"base margin: {a:.3f} * z + {b0:.3f}", flush=True)
    oof = np.zeros(len(y)); te = np.zeros(len(Xt))
    dte = xgb.DMatrix(Xt, feature_names=names, base_margin=bm_t)
    for seed in range(args.seeds):
        p["seed"] = seed
        for trf, vaf in GroupKFold(n_splits=5, shuffle=True, random_state=seed).split(np.arange(len(M)), groups=users):
            trf.sort(); vaf.sort()
            tr = np.isin(fidx, trf)
            qtr = np.repeat(np.arange(len(trf)), M[trf])
            dtr = xgb.DMatrix(Xv[tr], y[tr], qid=qtr if args.objective.startswith("rank:") else None, feature_names=names,
                              base_margin=None if bm_v is None else bm_v[tr])
            b = xgb.train(p, dtr, args.rounds)
            dva = xgb.DMatrix(Xv[~tr], feature_names=names, base_margin=None if bm_v is None else bm_v[~tr])
            pv = b.predict(dva, output_margin=True)
            oof[~tr] += pv / args.seeds
            te += feed_rank_z(offt, Mt, b.predict(dte, output_margin=True).astype(np.float32))[:, 1] / (5 * args.seeds)
    per = ndcg_per_feed(oof, y, off, M)
    res = dict(tag=args.tag, args=vars(args), oof_ndcg=float(per.mean()), se=float(per.std() / np.sqrt(len(per))),
               time=round(time.time() - t0))
    np.save(PREDS / f"stk2_{args.tag}_val.npy", oof.astype(np.float32))
    np.save(PREDS / f"stk2_{args.tag}_test.npy", te.astype(np.float32))
    (PREDS / f"stk2_{args.tag}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res))


if __name__ == "__main__":
    main()
