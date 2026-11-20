"""XGBoost GPU ranker on a large train feed-subsample (no position feature), early-stopped on val."""
import argparse, json, sys, time
sys.path.insert(0, "<repo>/src")
from common import *
from train_lgb import feed_rows, LEAKY_ON_TRAIN
from eval_family import base_features
import xgboost as xgb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_feeds", type=int, default=1000000)
    ap.add_argument("--objective", default="rank:ndcg")
    ap.add_argument("--pair_method", default="topk")
    ap.add_argument("--npairs", type=int, default=300)
    ap.add_argument("--eta", type=float, default=0.05)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--mcw", type=float, default=50)
    ap.add_argument("--rounds", type=int, default=4000)
    ap.add_argument("--family", default="")
    ap.add_argument("--no_tte", action="store_true")
    ap.add_argument("--sample", default="uniform", choices=["uniform", "random50", "pweight"],
                    help="uniform feeds | only random-like feeds (P(random)>0.5, M>=50) | uniform with group weight 0.2+P(random)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--luo", action="store_true", help="use leave-user-later-out it_hist_* features")
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    t0 = time.time()
    names = [n for n in base_features() if not n.startswith(LEAKY_ON_TRAIN)]
    if args.family:
        names += [n for n in all_feature_names("val") if n.startswith(tuple(args.family.split(","))) and n not in names]
    if args.no_tte:
        names = [n for n in names if not n.startswith("tte_")]
    if args.luo:
        names = swap_luo(names)
    ftr, Mtr, offtr = feed_meta("train")
    rng = np.random.default_rng(args.seed)
    prand = np.nan_to_num(np.load("<repo>/eda/labels_position/out/train_prandom.npz")["p_random"], nan=0.0)
    pool = np.arange(len(Mtr))
    if args.sample == "random50":
        pool = pool[(prand > 0.5) & (Mtr >= 50)]
    sel = np.sort(rng.choice(pool, size=min(args.n_feeds, len(pool)), replace=False))
    gw = (0.2 + prand[sel]).astype(np.float32) if args.sample == "pweight" else None
    rows = feed_rows(offtr, Mtr, sel)
    Xtr = load_feats("train", names, rows)
    ytr = np.load(FLAT / "train/cand_target.npy", mmap_mode="r")[rows].astype(np.float32)
    qtr = np.repeat(np.arange(len(sel)), Mtr[sel])
    fva, Mva, offva = feed_meta("val")
    Xva = load_feats("val", names)
    yva = np.load(FLAT / "val/cand_target.npy").astype(np.float32)
    qva = np.repeat(np.arange(len(Mva)), Mva)
    print(f"loaded {Xtr.shape} in {time.time()-t0:.0f}s", flush=True)
    dtr = xgb.QuantileDMatrix(Xtr, ytr, qid=qtr, weight=gw, feature_names=names, max_bin=256)
    del Xtr
    dva = xgb.QuantileDMatrix(Xva, yva, qid=qva, feature_names=names, ref=dtr)
    params = dict(objective=args.objective, device="cuda", tree_method="hist", eta=args.eta, max_depth=args.depth,
                  min_child_weight=args.mcw, subsample=0.8, colsample_bytree=0.7, reg_lambda=10, eval_metric="ndcg",
                  lambdarank_pair_method=args.pair_method, lambdarank_num_pair_per_sample=args.npairs,
                  ndcg_exp_gain=False, lambdarank_score_normalization=False, seed=args.seed, max_bin=256)
    b = xgb.train(params, dtr, args.rounds, evals=[(dva, "val")], early_stopping_rounds=200, verbose_eval=100)
    pv = b.predict(dva, iteration_range=(0, b.best_iteration + 1))
    s = ndcg(pv, yva, offva, Mva)
    out = Path("<repo>/preds")
    np.save(out / f"xgb_{args.tag}_val.npy", pv.astype(np.float32))
    fte, Mte, offte = feed_meta("test")
    dte = xgb.DMatrix(load_feats("test", names), feature_names=names)
    np.save(out / f"xgb_{args.tag}_test.npy", b.predict(dte, iteration_range=(0, b.best_iteration + 1)).astype(np.float32))
    b.save_model(str(out / f"xgb_{args.tag}.json"))
    imp = sorted(b.get_score(importance_type="gain").items(), key=lambda kv: -kv[1])[:30]
    res = dict(args=vars(args), val_ndcg=s, best_iter=b.best_iteration, time=time.time() - t0, top=imp)
    (out / f"xgb_{args.tag}.json.meta").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "top"}), flush=True)


if __name__ == "__main__":
    main()
