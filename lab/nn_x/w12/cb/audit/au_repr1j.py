"""audit 3h: JOINT representation test (the residual form can understate an input, COVERAGE_MAP row 323). One LightGBM, the stage-1
params and rounds, on the 177 tabular features PLUS new blocks -> z(W16). Control = stage 1 (out/s1_<set>.npy, same rows, params, rounds).
usage: au_repr1j.py <target V|H> <blocks comma list from emb,seq,prod,trd> [rounds]. No label in the fit. CPU, 12 threads."""
import time
from au_common import *
sys.path.insert(0, str(ROOT / "nn_x/cb26"))
import lightgbm as lgb
from cbdata import load_tab
from cb_member import member_features
from au_blocks import SEQ_NAMES
t0 = time.time(); log = lambda *a: print(f"[{time.time()-t0:6.0f}s]", *a, flush=True)
tgt = sys.argv[1]; blocks = sys.argv[2].split(","); ROUNDS = int(sys.argv[3]) if len(sys.argv) > 3 else 800
LR = float(sys.argv[4]) if len(sys.argv) > 4 else 0.1
EVS = set(int(x) for x in sys.argv[5].split(",")) if len(sys.argv) > 5 else {200, 400, 600}
tag = f"s1j_{tgt}_{'+'.join(blocks)}" + ("" if LR == 0.1 else f"_lr{LR}")
m = Meta(); names = member_features(); SP = dict(V="val", H="hold", T="test"); TAB = dict(V="val", H="trH", T="test")
Z = {k: w16_pool(m, SP[k]) for k in SP}
TRD = sorted(p.stem for p in (ROOT / "data/feats/val").glob("trd_*.npy"))


def mat(k):
    parts = [load_tab(TAB[k], names)]; ns = list(names)
    for b in blocks:
        if b == "emb": parts.append(np.load(CACHE / f"blk_{k}_emb.npy").astype(np.float32)); ns += [f"emb{i}" for i in range(128)]
        elif b == "seq": parts.append(np.load(CACHE / f"blk_{k}_seq.npy")); ns += list(SEQ_NAMES)
        elif b == "prod": parts.append(np.load(CACHE / f"blk_{k}_prod.npy").astype(np.float32)); ns += [f"pu{i}" for i in range(128)] + [f"ph{i}" for i in range(128)]
        elif b == "trd": parts.append(load_tab(TAB[k], TRD)); ns += TRD
    return np.concatenate(parts, 1), ns


tr = [k for k in SP if k != tgt]
Xs = [mat(k) for k in tr]; ns = Xs[0][1]
d = lgb.Dataset(np.concatenate([x[0] for x in Xs]), np.concatenate([Z[k] for k in tr]), feature_name=ns, categorical_feature=["category_id"], free_raw_data=True); del Xs
Xe, _ = mat(tgt); log("matrices", Xe.shape, "train sets", tr)
P = dict(objective="regression", learning_rate=LR, num_leaves=255, min_data_in_leaf=100, feature_fraction=0.5, bagging_fraction=0.5,
         bagging_freq=1, lambda_l2=10, max_bin=255, num_threads=12, verbose=-1, seed=0)
sp = SP[tgt]; off, M, y = m.off[sp], m.M[sp], m.y[sp]
s1 = np.load(OUT / f"s1_{tgt}.npy"); e1 = ndcg_fast(s1, y, off, M); ew = ndcg_fast(Z[tgt], y, off, M); res = {}
log(f"control (stage 1, 177 features, {ROUNDS} rounds): ndcg {e1.mean():.5f} rho {corr(m.z(sp, s1), Z[tgt]):.4f}; W16 {ew.mean():.5f}")
def cb(env):
    it = env.iteration + 1
    if it in EVS or it == ROUNDS:
        p = env.model.predict(Xe, num_iteration=it, num_threads=12).astype(np.float32); e = ndcg_fast(p, y, off, M); zp = m.z(sp, p)
        eb = ndcg_fast(0.9 * Z[tgt] + 0.1 * zp, y, off, M)
        r = dict(rho=corr(zp, Z[tgt]), r2=float(1 - np.mean((Z[tgt] - p) ** 2) / np.mean(Z[tgt] ** 2)), ndcg=float(e.mean()), d_vs_s1=float((e - e1).mean()), d_vs_s1_se=se(e - e1),
                 gap=float((ew - e).mean()), gap_se=se(ew - e), blend01=float((eb - ew).mean()), blend01_se=se(eb - ew))
        res[it] = r; log(f"round {it}: rho {r['rho']:.4f} R2 {r['r2']:.4f} ndcg {r['ndcg']:.5f} | vs stage 1 (round {ROUNDS}) {r['d_vs_s1']:+.5f} +- {r['d_vs_s1_se']:.5f} | gap to W16 {r['gap']:.5f} +- {r['gap_se']:.5f} | 0.9 W16 + 0.1 this {r['blend01']:+.5f} +- {r['blend01_se']:.5f}")
        if it == ROUNDS: np.save(OUT / f"{tag}.npy", p)
        json.dump(res, open(OUT / f"{tag}.json", "w"), indent=1)
b = lgb.train(P, d, ROUNDS, callbacks=[cb])
g = b.feature_importance("gain"); tot = g.sum(); grp = lambda f: float(sum(g[i] for i, n in enumerate(ns) if f(n)) / tot)
res["gain"] = dict(tab177=grp(lambda n: n in set(names)), emb=grp(lambda n: n.startswith("emb") and n[3:].isdigit()), seq=grp(lambda n: n.startswith("sq_")), prod=grp(lambda n: n[:2] in ("pu", "ph") and n[2:].isdigit()), trd=grp(lambda n: n in set(TRD)))
log("gain share by block:", res["gain"]); json.dump(res, open(OUT / f"{tag}.json", "w"), indent=1); log("DONE")
