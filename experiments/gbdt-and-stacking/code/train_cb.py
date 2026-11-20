# CatBoost ranker on GPU (YetiRank by default), trained on a sample of train feeds and early-stopped
# on val, with the same feed-sampling options as train_xgb.py.
# How it was run: arguments as logged in lab/logs/cb_tr300k_luo.log and cb_tr500k_pw.log, for
# example --n_feeds 500000 --lr 0.1 --iters 3000 --sample pweight --gpu N --tag tr500k_pw.
# What it produced: preds/cb_<tag>_{val,test}.npy; val NDCG 0.36577 (300,000 feeds, leak-safe
# counts) and 0.36855 (500,000 feeds, weighted). The first run, cb_tr300k_none, saved its
# predictions and then stopped on a feature-importance error (lab/logs/cb_tr300k_none.log); its
# val NDCG 0.36997 is read from the saved file in results/cb26_ana_existing.log.
# Kept as it ran: paths and names refer to the lab tree (see lab/ and docs/glossary.md).
# Original: lab/src/train_cb.py
"""CatBoost ranker (GPU) trained on a train feed-subsample, early-stopped on val. Same position modes as train_lgb.py."""
import argparse, json, os, sys, time
sys.path.insert(0, "<repo>/src")
from common import *
from train_lgb import feed_rows, LEAKY_ON_TRAIN
from eval_family import base_features


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_feeds", type=int, default=300000)
    ap.add_argument("--pos_mode", default="none", choices=["none", "feat"])
    ap.add_argument("--loss", default="YetiRank")
    ap.add_argument("--lr", type=float, default=0.08)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--iters", type=int, default=4000)
    ap.add_argument("--gpu", default="4")
    ap.add_argument("--exclude", default="")
    ap.add_argument("--no_tte", action="store_true")
    ap.add_argument("--family", default="")
    ap.add_argument("--sample", default="uniform", choices=["uniform", "random50", "pweight"])
    ap.add_argument("--metric_period", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--luo", action="store_true", help="use leave-user-later-out it_hist_* features")
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    from catboost import CatBoostRanker, Pool
    t0 = time.time()
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
    prand = np.nan_to_num(np.load("<repo>/eda/labels_position/out/train_prandom.npz")["p_random"], nan=0.0)
    pool = np.arange(len(Mtr))
    if args.sample == "random50":
        pool = pool[(prand > 0.5) & (Mtr >= 50)]
    sel = np.sort(rng.choice(pool, size=min(args.n_feeds, len(pool)), replace=False))
    gw = (0.2 + prand[sel]).astype(np.float32) if args.sample == "pweight" else None
    rows = feed_rows(offtr, Mtr, sel)
    Xtr = load_feats("train", names, rows)
    ytr = np.load(FLAT / "train/cand_target.npy", mmap_mode="r")[rows].astype(np.float32)
    gid = np.repeat(np.arange(len(sel)), Mtr[sel])
    fva, Mva, offva = feed_meta("val")
    Xva = load_feats("val", names)
    yva = np.load(FLAT / "val/cand_target.npy").astype(np.float32)
    gva = np.repeat(np.arange(len(Mva)), Mva)
    fn = list(names)
    if args.pos_mode == "feat":
        pos = np.load(FLAT / "train/cand_pos.npy", mmap_mode="r")[rows].astype(np.float32)
        Xtr = np.hstack([Xtr, pos[:, None]]); Xva = np.hstack([Xva, np.zeros((len(Xva), 1), np.float32)]); fn += ["pos"]
    import pandas as pd
    def pool(X, y, g, w=None):
        df = pd.DataFrame(X, columns=fn)
        df["category_id"] = df["category_id"].astype(np.int32)
        return Pool(df, label=y, group_id=g, cat_features=["category_id"], group_weight=w)
    ptr, pva = pool(Xtr, ytr, gid, None if gw is None else np.repeat(gw, Mtr[sel])), pool(Xva, yva, gva)
    print(f"pools ready {time.time()-t0:.0f}s", flush=True)
    m = CatBoostRanker(loss_function=args.loss, eval_metric="NDCG:type=Base", iterations=args.iters, learning_rate=args.lr,
                       depth=args.depth, l2_leaf_reg=5, task_type="GPU", devices=args.gpu, random_seed=args.seed,
                       border_count=254, metric_period=args.metric_period, od_type="Iter", od_wait=300, use_best_model=True,
                       allow_writing_files=False, thread_count=32, gpu_ram_part=0.3)
    m.fit(ptr, eval_set=pva, verbose=200)
    pv = m.predict(pva)
    s = ndcg(pv, yva, offva, Mva)
    out = Path("<repo>/preds")
    np.save(out / f"cb_{args.tag}_val.npy", pv.astype(np.float32))
    fte, Mte, offte = feed_meta("test")
    Xte = load_feats("test", names)
    if args.pos_mode == "feat":
        Xte = np.hstack([Xte, np.zeros((len(Xte), 1), np.float32)])
    np.save(out / f"cb_{args.tag}_test.npy", m.predict(pool(Xte, None, np.repeat(np.arange(len(Mte)), Mte))).astype(np.float32))
    imp = sorted(zip(m.get_feature_importance(type="PredictionValuesChange"), fn), reverse=True)[:30]
    res = dict(args=vars(args), val_ndcg=s, best_iter=m.get_best_iteration(), time=time.time() - t0, top=[(n, float(g)) for g, n in imp])
    (out / f"cb_{args.tag}.json").write_text(json.dumps(res, indent=1))
    m.save_model(str(out / f"cb_{args.tag}.cbm"))
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
