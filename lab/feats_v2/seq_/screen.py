"""Cheap screening (NOT the official harness): same LightGBM params as src/eval_family.py.
  python screen.py imp <names|prefixes>       gain importance of a single fit on all val (base + features)
  python screen.py cv <tag> <names|prefixes> [seed]  1-seed 5-fold GroupKFold delta vs cached base OOF (same seed)
"""
import sys, json
sys.path.insert(0, "<repo>/src")
from common import *
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from seq_lib import WORK, log

import eval_family as EF

TH = 16


def resolve(spec):
    allv = all_feature_names("val")
    out = []
    for s in spec.split(","):
        if s in allv:
            out.append(s)
        else:
            out += [n for n in allv if n.startswith(s)]
    return list(dict.fromkeys(out))


def main():
    mode = sys.argv[1]
    f, M, off = feed_meta("val")
    y = np.load(FLAT / "val/cand_target.npy").astype(np.float32)
    fidx = np.load(FLAT / "val/cand_fidx.npy")
    users = f["user_id"].to_numpy()
    base = json.loads((Path("<repo>/data/eval_cache") / "base_features.json").read_text())
    if mode == "imp":
        fam = [n for n in resolve(sys.argv[2]) if n not in base]
        names = base + fam
        X = load_feats("val", names)
        p = dict(EF.PARAMS, num_threads=TH, seed=0)
        b = lgb.train(p, lgb.Dataset(X, y, group=M, feature_name=names, categorical_feature=["category_id"]), EF.ROUNDS)
        g = b.feature_importance("gain"); g = g / g.sum()
        order = np.argsort(-g)
        print("rank feature gain_share")
        for r, i in enumerate(order):
            if names[i] in fam or r < 25:
                print(f"{r:4d} {names[i]:28s} {g[i]:.4f} {'*' if names[i] in fam else ''}")
        print("family total gain share", g[[names.index(n) for n in fam]].sum())
    elif mode == "cv":
        tag = sys.argv[2]
        fam = [n for n in resolve(sys.argv[3]) if n not in base]
        seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0
        cp = WORK / f"screen_base_oof_s{seed}.npy"
        official = Path("<repo>/data/eval_cache") / f"base_oof_s{seed}.npy"
        if official.exists():
            ob = np.load(official)
        elif cp.exists():
            ob = np.load(cp)
        else:
            ob = EF.cv_oof(load_feats("val", base), base, y, M, fidx, users, seed, TH)
            np.save(cp, ob)
        oc = EF.cv_oof(load_feats("val", base + fam), base + fam, y, M, fidx, users, seed, TH)
        pb, pc = ndcg_per_feed(ob, y, off, M), ndcg_per_feed(oc, y, off, M)
        d = pc - pb
        res = dict(tag=tag, seed=seed, n=len(fam), base=float(pb.mean()), cand=float(pc.mean()),
                   delta=float(d.mean()), se=float(d.std() / np.sqrt(len(d))))
        print(json.dumps(res))
        with open(WORK / "screen.jsonl", "a") as fh:
            fh.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
