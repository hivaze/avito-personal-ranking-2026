"""Time-split check: train LightGBM (eval_family PARAMS, ROUNDS) on one val day, evaluate on the other, base vs base+family.
Usage: timesplit.py <dir: tue2thu|thu2tue> <seeds> ; prints per-seed deltas, paired per-feed SE, seed-ensemble deltas."""
import sys, json
sys.path.insert(0, "<repo>/src")
from common import *
import lightgbm as lgb
from eval_family import PARAMS, ROUNDS, base_features
SCR = Path("/tmp/scratchpad")
K = "ctx_hn_soft05,ctx_hn_rrw,ctx_hn_e1_rrw,ctx_hn_top1w,ctx_hn_margin,ctx_hn_bestrk,ctx_hs_t10,ctx_hs_d7,ctx_hs_k128,ctx_hs_e1,ctx_hs_k256,ctx_hs_e1k64,ctx_hs_k128t10,ctx_hs_mutual,ctx_nd_knn_u_dev,ctx_nd_ucent,ctx_nd_knn_h_dev,ctx_nd_knn_h,ctx_nd_max,ctx_nd_hcent,ctx_nd_knn_f1,ctx_nd_knn_u,ctx_nd_top5u,ctx_nd_top5h".split(",")
direc, seeds = sys.argv[1], [int(s) for s in sys.argv[2].split(",")]
f, M, off = feed_meta("val")
y = np.load(FLAT / "val/cand_target.npy").astype(np.float32)
fidx = np.load(FLAT / "val/cand_fidx.npy")
day = ((f["timestamp"].to_numpy() + 3 * 3600) // 86400)
day = np.where(day <= day.min() + 1, 0, 1); d0, d1 = 0, 1  # Tue22 (+1 feed after midnight) vs Thu24 (+2 feeds after midnight)
print("days", np.unique(day, return_counts=True))
trd, ted = (d0, d1) if direc == "tue2thu" else (d1, d0)
users = f["user_id"].to_numpy()
tr_users = set(users[(day == trd) & (users != 0)].tolist())
trf = np.where(day == trd)[0]
tef = np.where((day == ted) & ~np.isin(users, list(tr_users)))[0]
print(direc, "train feeds", len(trf), "eval feeds", len(tef), "(excluded overlapping users:", int(((day == ted)).sum() - len(tef)), ")")
tr = np.isin(fidx, trf); te = np.isin(fidx, tef)
Mte = M[tef]; offte = np.concatenate([[0], np.cumsum(Mte)[:-1]])
base = base_features()
Xb = load_feats("val", base)
Xk = load_feats("val", sorted(K))
rng = np.random.default_rng(12345)
perm = np.argsort(fidx.astype(np.float64) + rng.random(len(fidx)), kind="stable")
variants = {"base": (Xb, base), "k24alpha": (np.concatenate([Xb, Xk], 1), base + sorted(K)),
            "k24shuf1": None, "placebo": (np.concatenate([Xb, Xk[perm]], 1), base + sorted(K))}
o1 = list(np.random.default_rng(1).permutation(K)); idx1 = [sorted(K).index(n) for n in o1]
variants["k24shuf1"] = (np.concatenate([Xb, Xk[:, idx1]], 1), base + o1)
res = {v: [] for v in variants}
cat = ["category_id"]
for s in seeds:
    for v, (X, names) in variants.items():
        p = dict(PARAMS, num_threads=16, seed=s)
        b = lgb.train(p, lgb.Dataset(X[tr], y[tr], group=M[trf], feature_name=names, categorical_feature=cat), ROUNDS)
        pr = b.predict(X[te]); res[v].append(pr)
        np.save(SCR / "oof" / f"tsfix_{direc}_{v}_s{s}.npy", pr)
    pb = ndcg_per_feed(res["base"][-1], y[te], offte, Mte)
    print("seed", s, "base", round(pb.mean(), 5), " ".join(f"{v} {ndcg_per_feed(res[v][-1], y[te], offte, Mte).mean() - pb.mean():+.5f}" for v in variants if v != "base"), flush=True)
fl = np.repeat(np.arange(len(tef)), Mte)
def fz(x):
    mu = np.bincount(fl, weights=x, minlength=len(Mte)) / Mte; d = x - mu[fl]
    return d / (np.sqrt(np.bincount(fl, weights=d * d, minlength=len(Mte)) / Mte)[fl] + 1e-9)
out = {}
for v in variants:
    if v == "base": continue
    ds = [ndcg_per_feed(c, y[te], offte, Mte) - ndcg_per_feed(b, y[te], offte, Mte) for b, c in zip(res["base"], res[v])]
    d = np.mean(ds, 0)
    de = ndcg_per_feed(np.mean([fz(c) for c in res[v]], 0), y[te], offte, Mte) - ndcg_per_feed(np.mean([fz(b) for b in res["base"]], 0), y[te], offte, Mte)
    out[v] = dict(mean_delta=round(float(d.mean()), 5), se=round(float(d.std() / np.sqrt(len(d))), 5),
                  per_seed=[round(float(x.mean()), 5) for x in ds], ens_delta=round(float(de.mean()), 5), ens_se=round(float(de.std() / np.sqrt(len(de))), 5))
print(json.dumps(dict(direction=direc, n_eval_feeds=len(tef), base_mean=float(np.mean([ndcg_per_feed(b, y[te], offte, Mte).mean() for b in res["base"]])), **out)))
