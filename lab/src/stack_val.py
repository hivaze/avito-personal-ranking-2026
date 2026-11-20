"""Second stage on val (unbiased, same distribution as test): GroupKFold-by-user CV.

Inputs: handcrafted features (optional) + external model scores (preds/<name>_val.npy & _test.npy) used as
features (raw + within-feed rank/z) and/or as init_score (continue boosting from a train-fitted model).
Outputs OOF val NDCG and fold-averaged test predictions.
"""
import argparse, json, sys, time
sys.path.insert(0, "<repo>/src")
from common import *
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
import numba as nb

PREDS = Path("<repo>/preds")


@nb.njit(parallel=True, cache=True)
def feed_rank_z(off, M, x):
    out = np.empty((len(x), 2), np.float32)
    for f in nb.prange(len(M)):
        o, m = off[f], M[f]
        v = x[o:o + m]
        r = np.argsort(np.argsort(v))
        mu = v.mean(); sd = v.std() + 1e-9
        for k in range(m):
            out[o + k, 0] = r[k] / (m - 1) if m > 1 else 0.5
            out[o + k, 1] = (v[k] - mu) / sd
    return out


def build(split, feat_names, extra):
    f, M, off = feed_meta(split)
    cols, names = [], []
    if feat_names:
        cols.append(load_feats(split, feat_names)); names += feat_names
    for e in extra:
        x = np.load(PREDS / f"{e}_{split}.npy").astype(np.float32)
        rz = feed_rank_z(off, M, x)
        cols += [x[:, None], rz]; names += [e, f"{e}_rk", f"{e}_z"]
    return np.hstack(cols), names, M, off, f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extra", default="", help="comma-separated pred names (preds/<name>_{val,test}.npy)")
    ap.add_argument("--init", default="", help="pred name used as init_score")
    ap.add_argument("--no_feats", action="store_true")
    ap.add_argument("--family", default="", help="extra feature prefixes on top of the frozen base set")
    ap.add_argument("--exclude", default="")
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--leaves", type=int, default=31)
    ap.add_argument("--min_data", type=int, default=200)
    ap.add_argument("--rounds", type=int, default=5000)
    ap.add_argument("--nfold", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--objective", default="lambdarank")
    ap.add_argument("--fixed_rounds", type=int, default=0, help="if >0, no early stopping (avoids optimistic OOF)")
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    t0 = time.time()
    from eval_family import base_features
    feat_names = [] if args.no_feats else base_features()
    if args.family:
        feat_names += [n for n in all_feature_names("val") if n.startswith(tuple(args.family.split(","))) and n not in feat_names]
    if args.exclude:
        feat_names = [n for n in feat_names if not n.startswith(tuple(args.exclude.split(",")))]
    extra = [e for e in args.extra.split(",") if e]
    Xva, names, M, off, f = build("val", feat_names, extra)
    Xte, _, Mte, offte, _ = build("test", feat_names, extra)
    y = np.load(FLAT / "val/cand_target.npy").astype(np.float32)
    fidx = np.load(FLAT / "val/cand_fidx.npy")
    users = f["user_id"].to_numpy()
    init_va = np.load(PREDS / f"{args.init}_val.npy").astype(np.float64) if args.init else None
    init_te = np.load(PREDS / f"{args.init}_test.npy").astype(np.float64) if args.init else None
    p = dict(objective=args.objective, metric="ndcg", eval_at=[1000], learning_rate=args.lr, num_leaves=args.leaves,
             min_data_in_leaf=args.min_data, feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10,
             lambdarank_truncation_level=1000, num_threads=48, verbose=-1)
    cat = ["category_id"] if "category_id" in names else []
    oof = np.zeros(len(y)); te = np.zeros(len(Xte)); its = []
    for seed in range(args.seeds):
        p["seed"] = seed
        gkf = GroupKFold(n_splits=args.nfold, shuffle=True, random_state=seed) if args.seeds > 1 else GroupKFold(n_splits=args.nfold)
        for k, (trf, vaf) in enumerate(gkf.split(np.arange(len(M)), groups=users)):
            trf.sort(); vaf.sort()
            tr = np.isin(fidx, trf); va = ~tr
            dtr = lgb.Dataset(Xva[tr], y[tr], group=M[trf], init_score=None if init_va is None else init_va[tr],
                              feature_name=names, categorical_feature=cat)
            dva = lgb.Dataset(Xva[va], y[va], group=M[vaf], init_score=None if init_va is None else init_va[va], reference=dtr)
            if args.fixed_rounds:
                b = lgb.train(p, dtr, args.fixed_rounds); bi = args.fixed_rounds
            else:
                b = lgb.train(p, dtr, args.rounds, valid_sets=[dva], callbacks=[lgb.early_stopping(300, verbose=False)])
                bi = b.best_iteration
            its.append(bi)
            pv = b.predict(Xva[va], num_iteration=bi) + (0 if init_va is None else init_va[va])
            oof[va] += pv / args.seeds
            pt = b.predict(Xte, num_iteration=bi) + (0 if init_te is None else init_te)
            # average fold models on test via per-feed z-score to be scale-free
            te += feed_rank_z(offte, Mte, pt.astype(np.float32))[:, 1] / (args.nfold * args.seeds)
    s = ndcg(oof, y, off, M)
    per = ndcg_per_feed(oof, y, off, M)
    res = dict(tag=args.tag, args=vars(args), oof_ndcg=s, se=float(per.std() / np.sqrt(len(per))), iters=its, time=time.time() - t0)
    np.save(PREDS / f"stk_{args.tag}_val.npy", oof.astype(np.float32))
    np.save(PREDS / f"stk_{args.tag}_test.npy", te.astype(np.float32))
    (PREDS / f"stk_{args.tag}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res))


if __name__ == "__main__":
    main()
