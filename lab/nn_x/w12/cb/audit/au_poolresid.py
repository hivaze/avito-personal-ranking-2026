"""audit 5a: pool-residual probe (see PREDECLARED_audit.txt, block of 00:05Z). CPU, 12 threads. usage: au_poolresid.py <blocks comma list>"""
import time
from au_common import *
sys.path.insert(0, str(ROOT / "nn_x/cb26"))
import lightgbm as lgb
from cbdata import load_tab
from cb_member import member_features
from au_blocks import SEQ_NAMES
t0 = time.time(); log = lambda *a: print(f"[{time.time()-t0:6.0f}s]", *a, flush=True)
m = Meta(); names = member_features(); SP = dict(V="val", H="hold"); TAB = dict(V="val", H="trH")
Z = {k: w16_pool(m, SP[k]) for k in SP}
beta = {k: float(clogit(Z[k][:, None], m.y[SP[k]].astype(np.float64), m.off[SP[k]], m.M[SP[k]])[0][0]) for k in SP}; log("beta", beta)


def mat(k, b):
    parts, ns = [], []
    if b in ("same177", "both"): parts.append(load_tab(TAB[k], names)); ns += list(names)
    if b in ("all_new", "both"):
        parts += [np.load(CACHE / f"blk_{k}_emb.npy").astype(np.float32), np.load(CACHE / f"blk_{k}_seq.npy"), np.load(CACHE / f"blk_{k}_prod.npy").astype(np.float32)]
        ns += [f"emb{i}" for i in range(128)] + list(SEQ_NAMES) + [f"pu{i}" for i in range(128)] + [f"ph{i}" for i in range(128)]
    fid = np.repeat(np.arange(len(m.M[SP[k]])), m.M[SP[k]])
    parts.append(np.stack([Z[k], rank_pct(m, SP[k], Z[k]), np.log(m.M[SP[k]][fid]).astype(np.float32)], 1)); ns += ["z_w16", "rk_w16", "logM"]
    return np.concatenate(parts, 1), ns


P = dict(objective="lambdarank", lambdarank_truncation_level=1000, learning_rate=0.03, num_leaves=31, min_data_in_leaf=500, feature_fraction=0.3, bagging_fraction=0.5,
         bagging_freq=1, lambda_l2=50, max_bin=63, num_threads=12, verbose=-1, seed=0)
res = {}
for b in sys.argv[1].split(","):
    for tr, rd in (("H", "V"), ("V", "H")):
        X, ns = mat(tr, b); sp = SP[tr]
        d = lgb.Dataset(X, m.y[sp], group=m.M[sp], init_score=beta[tr] * Z[tr].astype(np.float64), feature_name=ns, categorical_feature=["category_id"] if "category_id" in ns else "auto", free_raw_data=True)
        bst = lgb.train(P, d, 150); del X, d
        Xe, _ = mat(rd, b); spr = SP[rd]; y, off, M = m.y[spr], m.off[spr], m.M[spr]; e0 = ndcg_fast(Z[rd], y, off, M); out = {}
        for it in (50, 100, 150):
            f = bst.predict(Xe, num_iteration=it, raw_score=True, num_threads=12); e = ndcg_fast((beta[tr] * Z[rd] + f).astype(np.float32), y, off, M); dd = e - e0
            out[it] = (float(dd.mean()), se(dd)); log(f"block {b:8s} train {tr} labels -> read {spr}: round {it}: delta vs W16 {dd.mean():+.5f} +- {se(dd):.5f} (z {dd.mean()/se(dd):+.2f}); sd of f {f.std():.4f}")
        g = bst.feature_importance("gain"); o = np.argsort(-g)[:8]; log("   top gain:", [(ns[i], round(float(g[i] / g.sum()), 3)) for i in o])
        res[f"{b}_{tr}to{rd}"] = out; json.dump(res, open(OUT / "poolresid.json", "w"), indent=1); del Xe
log("DONE")
