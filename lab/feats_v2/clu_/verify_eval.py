"""Verifier's robustness evaluation for clu_ KEEP features (independent of eval_family's single fixed column order).

modes:
  cv    R replicates: GroupKFold(5, by user) with fold seed 1000+r, LightGBM seed 1000+r, and a random column permutation
        per replicate (same permutation of base columns for base and cand; family columns inserted at random positions).
        Fresh base model in every replicate (paired per-feed delta).
  time  train on val feeds of Tue Sep 22 (MSK), evaluate on Thu Sep 24 (and the reverse), R LightGBM seeds / column
        permutations, base vs base+family, paired per-feed delta.
Same LightGBM params and rounds as src/eval_family.py.
usage: python verify_eval.py <mode> <R | r0:r1> <names comma-separated> [threads]
"""
import json, sys, time
sys.path.insert(0, "<repo>/src")
from common import *
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from eval_family import PARAMS, ROUNDS, base_features

t0 = time.time()
mode, rr, fam = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
r0, r1 = (int(v) for v in rr.split(":")) if ":" in rr else (0, int(rr))
R = r1 - r0
threads = int(sys.argv[4]) if len(sys.argv) > 4 else 16
f, M, off = feed_meta("val")
y = np.load(FLAT / "val/cand_target.npy").astype(np.float32)
fidx = np.load(FLAT / "val/cand_fidx.npy")
users = f["user_id"].to_numpy()
ts = f["timestamp"].to_numpy().astype(np.int64)
day = (ts + 3 * 3600) // 86400 - 20702   # 16 = Tue Sep 22, 18 = Thu Sep 24
base = base_features()
Xb = load_feats("val", base)
Xf = load_feats("val", fam)
names_all = base + fam
X_all = np.concatenate([Xb, Xf], axis=1)
del Xb, Xf
nb_ = len(base)


def fit_pred(cols, seed, trf, tef):
    names = [names_all[c] for c in cols]
    p = dict(PARAMS, num_threads=threads, seed=seed)
    cat = ["category_id"] if "category_id" in names else []
    tr = np.isin(fidx, trf); te = np.isin(fidx, tef)
    Xs = X_all[:, cols]
    b = lgb.train(p, lgb.Dataset(Xs[tr], y[tr], group=M[trf], feature_name=names, categorical_feature=cat), ROUNDS)
    return te, b.predict(Xs[te])


def perms(rng):
    pb = rng.permutation(nb_)
    cand = list(pb)
    for j in range(len(fam)):
        cand.insert(int(rng.integers(0, len(cand) + 1)), nb_ + j)
    return list(pb), cand


res = []
if mode == "cv":
    D = []
    for r in range(r0, r1):
        seed = 1000 + r
        rng = np.random.default_rng(seed)
        cb, cc = perms(rng)
        ob = np.zeros(len(y)); oc = np.zeros(len(y))
        for trf, vaf in GroupKFold(n_splits=5, shuffle=True, random_state=seed).split(np.arange(len(M)), groups=users):
            trf.sort(); vaf.sort()
            te, p = fit_pred(cb, seed, trf, vaf); ob[te] = p
            te, p = fit_pred(cc, seed, trf, vaf); oc[te] = p
        pb, pc = ndcg_per_feed(ob, y, off, M), ndcg_per_feed(oc, y, off, M)
        D.append(pc - pb)
        res.append(dict(r=r, base=float(pb.mean()), cand=float(pc.mean()), delta=float((pc - pb).mean())))
        print(json.dumps(res[-1]), f"{time.time()-t0:.0f}s", flush=True)
    d = np.mean(D, axis=0)
    per = np.array([x["delta"] for x in res])
    out = dict(mode=mode, fam=fam, R=R, delta=float(d.mean()), feed_se=float(d.std() / np.sqrt(len(d))),
               rep_mean=float(per.mean()), rep_se=float(per.std(ddof=1) / np.sqrt(R)) if R > 1 else None,
               n_pos=int((per > 0).sum()), per=per.round(5).tolist())
else:
    out = dict(mode=mode, fam=fam, R=R)
    for a, b_ in ((16, 18), (18, 16)):
        trf = np.where(day == a)[0]; tef = np.where(day == b_)[0]
        D = []; per = []
        for r in range(r0, r1):
            seed = 2000 + r
            rng = np.random.default_rng(seed)
            cb, cc = perms(rng)
            te, pbv = fit_pred(cb, seed, trf, tef)
            _, pcv = fit_pred(cc, seed, trf, tef)
            ot = off[tef] - off[tef[0]]
            # tef are contiguous? rows sorted by (timestamp, feed_id) -> feeds of one day are contiguous
            assert np.all(np.diff(tef) == 1)
            yt = y[te]
            pb, pc = ndcg_per_feed(pbv, yt, ot, M[tef]), ndcg_per_feed(pcv, yt, ot, M[tef])
            D.append(pc - pb); per.append(float((pc - pb).mean()))
            print(f"train day {a} -> test day {b_} r={r} base={pb.mean():.5f} cand={pc.mean():.5f} delta={per[-1]:+.5f}",
                  f"{time.time()-t0:.0f}s", flush=True)
        d = np.mean(D, axis=0); per = np.array(per)
        out[f"{a}->{b_}"] = dict(n_test_feeds=int(len(tef)), delta=float(d.mean()), feed_se=float(d.std() / np.sqrt(len(d))),
                                rep_se=float(per.std(ddof=1) / np.sqrt(R)) if R > 1 else None, per=per.round(5).tolist())
print("RESULT", json.dumps(out), flush=True)
