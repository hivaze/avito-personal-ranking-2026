"""Verifier: do the KEEP features help a LightGBM trained on (position-biased) TRAIN feeds? 100K random train feeds
(seed s), fixed 250 rounds, lr 0.05, 63 leaves, no early stopping; base vs base+KEEP6 with the same feeds/seed;
paired per-feed delta on val. usage: python verify_trainlgb.py <seed> <threads>"""
import sys, json, time
sys.path.insert(0, "<repo>/src")
from common import *
import lightgbm as lgb
from eval_family import base_features
t0 = time.time()
seed, threads = int(sys.argv[1]), int(sys.argv[2])
KEEP = ["clu_hcnt1024", "clu_hmin1024", "clu_hrw4096", "clu_het14096", "clu_pmi1024_mean", "clu_pmi11024_mean"]
base = [n for n in base_features() if not n.startswith(("it_tr_", "u_tr_"))]
ft, Mt, offt = feed_meta("train"); rng = np.random.default_rng(100 + seed)
sel = np.sort(rng.choice(len(Mt), 100_000, replace=False))
rows = np.concatenate([np.arange(offt[i], offt[i] + Mt[i]) for i in sel])
ytr = np.load(FLAT / "train/cand_target.npy", mmap_mode="r")[rows].astype(np.float32)
fv, Mv, offv = feed_meta("val"); yv = np.load(FLAT / "val/cand_target.npy").astype(np.float32)
P = dict(objective="lambdarank", metric="ndcg", eval_at=[1000], learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
         feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10, lambdarank_truncation_level=1000,
         num_threads=threads, verbose=-1, seed=seed)
res = {}
for nm, names in (("base", base), ("cand", base + KEEP)):
    Xtr = load_feats("train", names, rows); Xv = load_feats("val", names)
    b = lgb.train(P, lgb.Dataset(Xtr, ytr, group=Mt[sel], feature_name=names, categorical_feature=["category_id"]), 250)
    res[nm] = ndcg_per_feed(b.predict(Xv), yv, offv, Mv)
    if nm == "cand":
        imp = dict(zip(names, b.feature_importance("gain"))); tot = sum(imp.values())
        rk = {n: i + 1 for i, (n, _) in enumerate(sorted(imp.items(), key=lambda x: -x[1]))}
        print("KEEP gain share / rank:", {k: (round(imp[k] / tot, 4), rk[k]) for k in KEEP})
    print(nm, f"{res[nm].mean():.5f}", f"{time.time()-t0:.0f}s", flush=True)
d = res["cand"] - res["base"]
print("RESULT", json.dumps(dict(seed=seed, base=float(res["base"].mean()), cand=float(res["cand"].mean()),
                                delta=float(d.mean()), se=float(d.std() / np.sqrt(len(d))))))
