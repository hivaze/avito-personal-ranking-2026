"""Fast GPU screening (XGBoost rank:ndcg, same GroupKFold-by-user protocol as eval_family, 1-3 seeds).
Only used to pre-filter feature subsets; every claim in the report is re-checked with src/eval_family.py.
Usage: CUDA_VISIBLE_DEVICES=N python screen.py name1,name2,...  [--seeds 2] [--tag x] [--save_oof path]"""
import sys, json, argparse, time
sys.path.insert(0, "<repo>/src")
from common import *
import xgboost as xgb
from sklearn.model_selection import GroupKFold

SCR = Path("/tmp/scratchpad")
BASE = json.loads(Path("<repo>/data/eval_cache/base_features.json").read_text())
P = dict(objective="rank:ndcg", eval_metric="ndcg", tree_method="hist", device="cuda", learning_rate=0.05,
         max_depth=0, max_leaves=31, grow_policy="lossguide", min_child_weight=5, subsample=0.8,
         colsample_bytree=0.7, reg_lambda=10, lambdarank_pair_method="mean", lambdarank_num_pair_per_sample=4,
         max_bin=128)
ROUNDS = 400


def oof(X, y, M, fidx, users, seed):
    o = np.zeros(len(y))
    gkf = GroupKFold(n_splits=5, shuffle=True, random_state=seed)
    qid = fidx
    for trf, vaf in gkf.split(np.arange(len(M)), groups=users):
        tr = np.isin(fidx, trf)
        d = xgb.DMatrix(X[tr], y[tr], qid=qid[tr])
        b = xgb.train(dict(P, seed=seed), d, ROUNDS)
        o[~tr] = b.predict(xgb.DMatrix(X[~tr]))
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--tag", default="")
    ap.add_argument("--drop", default="")
    ap.add_argument("--save_oof", default="")
    a = ap.parse_args()
    t0 = time.time()
    f, M, off = feed_meta("val")
    y = np.load(FLAT / "val/cand_target.npy").astype(np.float32)
    fidx = np.load(FLAT / "val/cand_fidx.npy")
    users = f["user_id"].to_numpy()
    fam = [n for n in a.names.split(",") if n]
    base = [n for n in BASE if not (a.drop and n.startswith(tuple(a.drop.split(","))))]
    Xb = None
    ds, rb, rc = [], [], []
    for s in range(a.seeds):
        cp = SCR / f"xgb_base_oof_s{s}.npy"
        if cp.exists() and not a.drop:
            ob = np.load(cp)
        else:
            Xb = load_feats("val", BASE) if Xb is None else Xb
            ob = oof(Xb, y, M, fidx, users, s)
            np.save(cp, ob)
        Xc = load_feats("val", base + fam)
        oc = oof(Xc, y, M, fidx, users, s)
        if a.save_oof:
            np.save(f"{a.save_oof}_s{s}.npy", oc)
        pb, pc = ndcg_per_feed(ob, y, off, M), ndcg_per_feed(oc, y, off, M)
        ds.append(pc - pb); rb.append(pb.mean()); rc.append(pc.mean())
    d = np.mean(ds, axis=0)
    res = dict(tag=a.tag or a.names[:50], n=len(fam), base=float(np.mean(rb)), cand=float(np.mean(rc)),
               delta=float(d.mean()), se=float(d.std() / np.sqrt(len(d))), per_seed=[float(c - b) for b, c in zip(rb, rc)],
               sec=round(time.time() - t0))
    print(json.dumps(res), flush=True)
    with open(SCR / "screen_log.jsonl", "a") as fh:
        fh.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
