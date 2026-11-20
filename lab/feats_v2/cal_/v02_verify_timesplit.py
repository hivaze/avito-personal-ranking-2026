"""VERIFIER: time-split check on val. Fit LightGBM (same PARAMS / ROUNDS as src/eval_family.py) on val Tue Sep 22 feeds and
evaluate on Thu Sep 24 feeds, and the reverse. Configurations: base, base+subset, base-hour, base+noise control.
Reports mean NDCG (pessimistic ties, src/metrics.py) and paired per-feed delta vs base, averaged over seeds, SE over feeds.
Usage: python v02_verify_timesplit.py <seeds> <threads> name1=feat,feat;name2=...  (configs; '-hour' means drop hour)"""
import sys, json, time
sys.path.insert(0, "<repo>/src")
sys.path.insert(0, "<repo>/feats_v2/cal_")
from common import feed_meta, load_feats, FLAT
from metrics import ndcg_per_feed
import numpy as np, lightgbm as lgb, json as js
import importlib.util
spec = importlib.util.spec_from_file_location("ef", "<repo>/src/eval_family.py"); ef = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ef)
PARAMS, ROUNDS = ef.PARAMS, ef.ROUNDS
base = js.loads(open("<repo>/data/eval_cache/base_features.json").read())

n_seeds = int(sys.argv[1]); threads = int(sys.argv[2])
configs = {"base": list(base)}
for item in sys.argv[3].split(";"):
    nm, spec_ = item.split("=")
    feats = list(base)
    for tok in spec_.split(","):
        if tok.startswith("-"):
            feats.remove(tok[1:])
        elif tok:
            feats.append(tok)
    configs[nm] = feats

f, M, off = feed_meta("val")
y = np.load(FLAT / "val/cand_target.npy").astype(np.float32)
fidx = np.load(FLAT / "val/cand_fidx.npy")
day = ((f["timestamp"].to_numpy().astype(np.int64) + 3 * 3600) // 86400)
# val = Tue Sep 22 + Thu Sep 24 (MSK); 1 feed spills into Wed 00:xx and 2 into Fri 00:xx -> attach to the preceding day
d22 = int(np.datetime64("2026-09-22", "D").astype(np.int64)); d24 = d22 + 2
day = np.where(day <= d22 + 1, d22, d24)
days = np.unique(day); assert len(days) == 2, days
print("val days (MSK day numbers)", days, [str(np.datetime64(int(d), "D")) for d in days], "feeds", [(day == d).sum() for d in days], flush=True)
allnames = sorted(set(sum(configs.values(), [])))
X = load_feats("val", allnames); col = {n: i for i, n in enumerate(allnames)}

res = {}
for di, (dtr, dev) in enumerate([(days[0], days[1]), (days[1], days[0])]):
    trf = np.where(day == dtr)[0]; evf = np.where(day == dev)[0]
    tr = np.isin(fidx, trf); ev = ~tr
    Mev = M[evf]; offev = np.r_[0, np.cumsum(Mev)[:-1]]
    yev = y[ev]
    per = {}
    for cn, feats in configs.items():
        ci = [col[n] for n in feats]
        cat = ["category_id"] if "category_id" in feats else []
        pf = []
        for seed in range(n_seeds):
            t0 = time.time()
            p = dict(PARAMS, num_threads=threads, seed=seed)
            dtr_ = lgb.Dataset(X[tr][:, ci], y[tr], group=M[trf], feature_name=feats, categorical_feature=cat)
            b = lgb.train(p, dtr_, ROUNDS)
            pf.append(ndcg_per_feed(b.predict(X[ev][:, ci]), yev, offev, Mev))
            print(f"dir{di} {cn} seed{seed} ndcg={pf[-1].mean():.5f} ({time.time() - t0:.0f}s)", flush=True)
        per[cn] = np.array(pf)  # [seeds, feeds]
    tag = f"{str(np.datetime64(int(dtr), 'D'))}->{str(np.datetime64(int(dev), 'D'))}"
    res[tag] = {}
    for cn in configs:
        d = per[cn].mean(0) - per["base"].mean(0)
        res[tag][cn] = dict(ndcg=float(per[cn].mean()), delta=float(d.mean()), se=float(d.std() / np.sqrt(len(d))),
                            per_seed=[float(a - b) for a, b in zip(per[cn].mean(1), per["base"].mean(1))])
        print(tag, cn, json.dumps(res[tag][cn]), flush=True)
    res[tag]["_per"] = {cn: per[cn].mean(0) for cn in configs}
# pooled over both directions (independent eval feeds)
print("POOLED over both directions:")
pooled = {}
for cn in configs:
    d = np.concatenate([res[t]["_per"][cn] - res[t]["_per"]["base"] for t in res if not t.startswith("_")])
    pooled[cn] = dict(delta=float(d.mean()), se=float(d.std() / np.sqrt(len(d))))
    print(cn, json.dumps(pooled[cn]))
out = {t: {c: v for c, v in r.items() if c != "_per"} for t, r in res.items()}
out["pooled"] = pooled
json.dump(out, open("<repo>/feats_v2/cal_/out/verify/v02_timesplit.json", "w"), indent=1)
