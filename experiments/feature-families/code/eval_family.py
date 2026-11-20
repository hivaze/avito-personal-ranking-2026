# The tree harness that judged every feature family of the first round.
# It fits LightGBM lambdarank on the validation days with 5 folds grouped by user, once on a frozen base set
# and once on base + the new columns, and reports the paired per-feed NDCG difference with its standard error.
# Run as, for example (from lab/feats_v2/cal_/run_harness_subsets.sh):
#   python src/eval_family.py --drop hour --threads 16 --tag cal_ablation_drop_hour
# Every run appended one line to lab/reports/feature_evals.jsonl (120 runs; see ../results/harness_runs.csv).
# Kept as it ran: paths and names refer to the lab tree (see lab/ and docs/glossary.md).
# Original: lab/src/eval_family.py
"""Standardized feature-family evaluation on val (the only unbiased labels).

Protocol (identical for every family, so deltas are comparable):
  * LightGBM lambdarank, fixed rounds (no early stopping on the evaluated fold -> no optimistic bias)
  * GroupKFold by user_id, 5 folds, repeated over --seeds different fold shuffles
  * base = BASE_FEATURES (cached OOF per seed); candidate = base + family features
  * report mean OOF NDCG for both and the paired per-feed delta with its standard error

Usage:
  python src/eval_family.py --family cal_          # all val feature files whose name starts with "cal_"
  python src/eval_family.py --names a,b,c --tag mytest
  python src/eval_family.py --drop cat_ --tag no_cat   # ablation: base minus a prefix
Run with at most --threads threads (default 24); results appended to reports/feature_evals.jsonl.
"""
import argparse, json, os, sys, time
sys.path.insert(0, "<repo>/src")
from common import *
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

CACHE = Path("<repo>/data/eval_cache")
OUT = Path("<repo>/reports/feature_evals.jsonl")
# frozen base set: v1 handcrafted features (+ time-aware train TE). New families are evaluated on top of it.
V1_EXCLUDE = ("u_tr_", "it_tr_")  # superseded by tte_*
PARAMS = dict(objective="lambdarank", metric="ndcg", eval_at=[1000], learning_rate=0.03, num_leaves=31,
              min_data_in_leaf=200, feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10,
              lambdarank_truncation_level=1000, verbose=-1)
ROUNDS = 350


def base_features():
    names = json.loads((CACHE / "base_features.json").read_text()) if (CACHE / "base_features.json").exists() else None
    if names is None:
        names = [n for n in all_feature_names("val") if not n.startswith(V1_EXCLUDE)
                 and not n.startswith(("cal_", "trd_", "life_", "seq_", "clu_", "ctx_", "nn_", "x_"))]
        CACHE.mkdir(parents=True, exist_ok=True)
        (CACHE / "base_features.json").write_text(json.dumps(names))
    return names


def cv_oof(X, names, y, M, fidx, users, seed, threads):
    p = dict(PARAMS, num_threads=threads, seed=seed)
    oof = np.zeros(len(y))
    gkf = GroupKFold(n_splits=5, shuffle=True, random_state=seed)
    cat = ["category_id"] if "category_id" in names else []
    for trf, vaf in gkf.split(np.arange(len(M)), groups=users):
        trf.sort(); vaf.sort()
        tr = np.isin(fidx, trf)
        dtr = lgb.Dataset(X[tr], y[tr], group=M[trf], feature_name=names, categorical_feature=cat)
        b = lgb.train(p, dtr, ROUNDS)
        oof[~tr] = b.predict(X[~tr])
    return oof


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="", help="feature-name prefix(es), comma-separated")
    ap.add_argument("--names", default="", help="explicit comma-separated feature names")
    ap.add_argument("--drop", default="", help="prefixes to drop from base (ablation)")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--threads", type=int, default=24)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    t0 = time.time()
    f, M, off = feed_meta("val")
    y = np.load(FLAT / "val/cand_target.npy").astype(np.float32)
    fidx = np.load(FLAT / "val/cand_fidx.npy")
    users = f["user_id"].to_numpy()
    base = base_features()
    fam = []
    if args.family:
        pref = tuple(args.family.split(","))
        fam = [n for n in all_feature_names("val") if n.startswith(pref) and n not in base]
    if args.names:
        fam += [n for n in args.names.split(",") if n and n not in base]
    cand = [n for n in base if not (args.drop and n.startswith(tuple(args.drop.split(","))))] + fam
    tag = args.tag or args.family or args.names[:40] or f"drop_{args.drop}"
    Xb = None
    res_b, res_c, deltas = [], [], []
    for seed in range(args.seeds):
        cp = CACHE / f"base_oof_s{seed}.npy"
        lock = CACHE / f"base_oof_s{seed}.lock"
        if not cp.exists():
            try:  # first process computes the base OOF, concurrent ones wait for it (avoids redundant CV runs)
                os.close(os.open(lock, os.O_CREAT | os.O_EXCL))
                Xb = load_feats("val", base) if Xb is None else Xb
                ob = cv_oof(Xb, base, y, M, fidx, users, seed, args.threads)
                np.save(str(cp) + ".tmp.npy", ob)
                os.replace(str(cp) + ".tmp.npy", cp)
            except FileExistsError:
                while not cp.exists():
                    time.sleep(10)
        ob = np.load(cp)
        Xc = load_feats("val", cand)
        oc = cv_oof(Xc, cand, y, M, fidx, users, seed, args.threads)
        pb, pc = ndcg_per_feed(ob, y, off, M), ndcg_per_feed(oc, y, off, M)
        res_b.append(pb.mean()); res_c.append(pc.mean()); deltas.append(pc - pb)
    d = np.mean(deltas, axis=0)
    res = dict(tag=tag, n_family=len(fam), family=fam[:60], n_dropped=len(base) + len(fam) - len(cand),
               base_ndcg=float(np.mean(res_b)), cand_ndcg=float(np.mean(res_c)),
               delta=float(d.mean()), delta_se=float(d.std() / np.sqrt(len(d))),
               per_seed_delta=[float(c - b) for b, c in zip(res_b, res_c)], seconds=round(time.time() - t0))
    with open(OUT, "a") as fh:
        fh.write(json.dumps(res) + "\n")
    print(json.dumps(res))


if __name__ == "__main__":
    main()
