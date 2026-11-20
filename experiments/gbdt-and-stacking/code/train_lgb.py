# LightGBM ranker trained on a sample of train feeds (production order) and early-stopped on val.
# It has three ways to treat the display position: ignore it, use it as a feature and fix it at
# inference, or use log CTR(position) as an offset during training only.
# How it was run: arguments as logged in lab/logs/tr300k_none.log, tr300k_feat.log and
# tr300k_offset.log: --n_feeds 300000 --pos_mode none|feat|offset --threads 48 --save_test --tag <tag>.
# What it produced: preds/lgb_<tag>_{val,test}.npy and a json with the val NDCG: 0.3682 (none),
# 0.3676 (feat, position fixed at 5), 0.3675 (offset). Table: results/early_trees.csv.
# These three runs used history counts that include the same user's later snapshots; the --luo
# flag, added afterwards, switches to the leak-safe counts.
# Kept as it ran: paths and names refer to the lab tree (see lab/ and docs/glossary.md).
# Original: lab/src/train_lgb.py
"""LightGBM ranker trained on a feed-subsample of (position-biased) train, early-stopped/evaluated on val (unbiased).

Position handling (--pos_mode):
  none    pos not used
  feat    pos (+ pos/M) as feature at train time; at inference pos is set to a constant (--pos_infer)
  offset  init_score = alpha * log(CTR(pos)) at train time (shallow additive position tower), no offset at inference
"""
import argparse, json, sys, time
sys.path.insert(0, "<repo>/src")
from common import *
import lightgbm as lgb

LEAKY_ON_TRAIN = ("it_tr_", "u_tr_")


def feed_rows(off, M, feeds_sel):
    return np.concatenate([np.arange(off[i], off[i] + M[i]) for i in feeds_sel]) if len(feeds_sel) < 50000 else \
        np.repeat(off[feeds_sel] - np.r_[0, np.cumsum(M[feeds_sel])[:-1]], M[feeds_sel]) + np.arange(M[feeds_sel].sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_feeds", type=int, default=300000)
    ap.add_argument("--recent_days", type=float, default=0, help="sample only from the last N days of train (0 = all)")
    ap.add_argument("--pos_mode", default="none", choices=["none", "feat", "offset"])
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--objective", default="lambdarank")
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--leaves", type=int, default=127)
    ap.add_argument("--rounds", type=int, default=3000)
    ap.add_argument("--min_data", type=int, default=200)
    ap.add_argument("--threads", type=int, default=64)
    ap.add_argument("--exclude", default="", help="comma-separated feature prefixes to drop")
    ap.add_argument("--family", default="", help="extra feature prefixes on top of the frozen base set")
    ap.add_argument("--no_tte", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--luo", action="store_true", help="use leave-user-later-out it_hist_* features")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--save_test", action="store_true")
    args = ap.parse_args()
    t0 = time.time()

    from eval_family import base_features
    names = [n for n in base_features() if not n.startswith(LEAKY_ON_TRAIN)]
    if args.family:
        names += [n for n in all_feature_names("val") if n.startswith(tuple(args.family.split(","))) and n not in names]
    if args.no_tte:
        names = [n for n in names if not n.startswith("tte_")]
    if args.exclude:
        names = [n for n in names if not n.startswith(tuple(args.exclude.split(",")))]
    if args.luo:
        names = swap_luo(names)
    ftr, Mtr, offtr = feed_meta("train")
    rng = np.random.default_rng(args.seed)
    cand = np.arange(len(Mtr))
    if args.recent_days:
        ts = ftr["timestamp"].to_numpy()
        cand = cand[ts >= ts.max() - args.recent_days * 86400]
    sel = np.sort(rng.choice(cand, size=min(args.n_feeds, len(cand)), replace=False))
    rows = feed_rows(offtr, Mtr, sel)
    Xtr = load_feats("train", names, rows)
    ytr = np.load(FLAT / "train/cand_target.npy", mmap_mode="r")[rows].astype(np.float32)
    pos = np.load(FLAT / "train/cand_pos.npy", mmap_mode="r")[rows].astype(np.float32)
    gtr = Mtr[sel]
    fval, Mva, offva = feed_meta("val")
    Xva = load_feats("val", names)
    yva = np.load(FLAT / "val/cand_target.npy").astype(np.float32)
    print(f"loaded train {Xtr.shape} val {Xva.shape} in {time.time()-t0:.0f}s", flush=True)

    init_tr = None
    feat_names = list(names)
    if args.pos_mode == "feat":
        Mrow = np.repeat(gtr, gtr).astype(np.float32)
        Xtr = np.hstack([Xtr, pos[:, None], (pos / Mrow)[:, None]])
        feat_names += ["pos", "pos_rel"]
    elif args.pos_mode == "offset":
        pc = np.bincount(pos.astype(int), weights=ytr, minlength=301) / np.maximum(np.bincount(pos.astype(int), minlength=301), 1)
        pc = np.maximum(pc, 1e-4)
        init_tr = args.alpha * np.log(pc[pos.astype(int)] / pc.mean())

    def val_matrix(pv):
        if args.pos_mode != "feat":
            return Xva
        Mrow = np.repeat(Mva, Mva).astype(np.float32)
        p = np.full(len(yva), pv, np.float32) if pv >= 0 else np.random.default_rng(1).random(len(yva)).astype(np.float32) * Mrow
        return np.hstack([Xva, p[:, None], (p / Mrow)[:, None]])

    params = dict(objective=args.objective, metric="ndcg", eval_at=[1000], learning_rate=args.lr, num_leaves=args.leaves,
                  min_data_in_leaf=args.min_data, feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10,
                  lambdarank_truncation_level=1000, num_threads=args.threads, verbose=-1, seed=args.seed, max_bin=255)
    if args.objective == "binary":
        params["metric"] = "ndcg"
    cat = ["category_id"] if "category_id" in feat_names else []
    dtr = lgb.Dataset(Xtr, ytr, group=gtr, init_score=init_tr, feature_name=feat_names, categorical_feature=cat,
                      free_raw_data=True)
    pv0 = 0.0 if args.pos_mode == "feat" else None
    dva = lgb.Dataset(val_matrix(pv0 if pv0 is not None else 0), yva, group=Mva, reference=dtr)
    b = lgb.train(params, dtr, args.rounds, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(100)])
    res = {"tag": vars(args), "best_iter": b.best_iteration, "train_time": time.time() - t0}
    out = Path("<repo>/preds"); out.mkdir(exist_ok=True)
    pvals = [0, 5, 20, 50, -1] if args.pos_mode == "feat" else [None]
    best = (-1, None)
    for pv in pvals:
        p = b.predict(val_matrix(pv if pv is not None else 0), num_iteration=b.best_iteration)
        s = ndcg(p, yva, offva, Mva)
        res[f"val_ndcg_pos{pv}"] = s
        print(f"val NDCG (pos_infer={pv}) = {s:.4f}", flush=True)
        if s > best[0]:
            best = (s, pv, p)
    np.save(out / f"lgb_{args.tag}_val.npy", best[2].astype(np.float32))
    res["best_val"] = best[0]; res["best_pos_infer"] = best[1]
    imp = sorted(zip(b.feature_importance("gain"), feat_names), reverse=True)
    res["top_features"] = [(n, float(g)) for g, n in imp[:30]]
    b.save_model(str(out / f"lgb_{args.tag}.txt"))
    if args.save_test:
        fte, Mte, offte = feed_meta("test")
        Xte = load_feats("test", names)
        if args.pos_mode == "feat":
            Mrow = np.repeat(Mte, Mte).astype(np.float32)
            pv = best[1]
            p = np.full(len(Xte), pv, np.float32) if pv >= 0 else np.random.default_rng(1).random(len(Xte)).astype(np.float32) * Mrow
            Xte = np.hstack([Xte, p[:, None], (p / Mrow)[:, None]])
        np.save(out / f"lgb_{args.tag}_test.npy", b.predict(Xte, num_iteration=b.best_iteration).astype(np.float32))
    (out / f"lgb_{args.tag}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "top_features"}), flush=True)
    for n, g in res["top_features"][:20]:
        print(f"  {g:12.0f} {n}")


if __name__ == "__main__":
    main()
