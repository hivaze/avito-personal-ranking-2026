"""GPU proxy of src/eval_family.py for fast screening (NOT used for final claims).
Same protocol: 5-fold GroupKFold by user (shuffle seed), base = frozen base set, candidate = base + names; paired per-feed delta.
XGBoost rank:ndcg on CUDA (CUDA_VISIBLE_DEVICES=N). Base OOF cached in data/trd_cache/xgb_base_oof_s{seed}.npy.
Usage: python screen_xgb.py name1,name2,... [--seeds 2] [--tag x]"""
import sys, json, time, argparse, os
sys.path.insert(0, "<repo>/src")
from common import *
import xgboost as xgb
from sklearn.model_selection import GroupKFold

CACHE = Path("<repo>/data/trd_cache")
BASE = json.loads(Path("<repo>/data/eval_cache/base_features.json").read_text())
P = dict(objective="rank:ndcg", eta=0.06, max_depth=0, max_leaves=31, grow_policy="lossguide", min_child_weight=5,
         subsample=0.8, colsample_bytree=0.7, reg_lambda=10, tree_method="hist", device="cuda", lambdarank_pair_method="topk",
         lambdarank_num_pair_per_sample=300, eval_metric="ndcg", max_bin=255)
ROUNDS = 300


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
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "5"
    ap = argparse.ArgumentParser(); ap.add_argument("names"); ap.add_argument("--seeds", type=int, default=2); ap.add_argument("--tag", default=""); ap.add_argument("--seed0", type=int, default=0)
    a = ap.parse_args()
    names = [n for n in a.names.split(",") if n]
    f, M, off = feed_meta("val")
    y = np.load(FLAT / "val/cand_target.npy").astype(np.float32)
    fidx = np.load(FLAT / "val/cand_fidx.npy")
    users = f["user_id"].to_numpy()
    t0 = time.time()
    Xb = None; ds = []; rb = []; rc = []
    for s in range(a.seed0, a.seed0 + a.seeds):
        cp = CACHE / f"xgb_base_oof_s{s}.npy"
        if cp.exists():
            ob = np.load(cp)
        else:
            Xb = load_feats("val", BASE) if Xb is None else Xb
            ob = oof(Xb, y, M, fidx, users, s); np.save(cp, ob)
        Xc = load_feats("val", BASE + names)
        oc = oof(Xc, y, M, fidx, users, s)
        pb, pc = ndcg_per_feed(ob, y, off, M), ndcg_per_feed(oc, y, off, M)
        rb.append(pb.mean()); rc.append(pc.mean()); ds.append(pc - pb)
    d = np.mean(ds, 0)
    res = dict(tag=a.tag or a.names[:50], n=len(names), base=float(np.mean(rb)), cand=float(np.mean(rc)), delta=float(d.mean()),
               se=float(d.std() / np.sqrt(len(d))), per_seed=[float(c - b) for b, c in zip(rb, rc)], sec=round(time.time() - t0))
    with open(CACHE / "xgb_screen.jsonl", "a") as fh:
        fh.write(json.dumps(res) + "\n")
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
