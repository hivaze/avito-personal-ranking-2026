"""audit 3c: representation test, stage 2. Residual r = z(W16) - stage-1 tree fit (out of sample on every set). A second LightGBM is
fitted on r from ONE block of inputs (+ the stage-1 prediction, M, H as conditioning columns) and read on a feed set it never saw.
 direction V: train on H + T, read on val;   direction H: train on V + T, read on the holdout.
No label is used in any fit; labels only score the result. usage: au_repr2.py <direction V|H> <block,block,...> [rounds]   CPU, 12 threads."""
import time
from au_common import *
sys.path.insert(0, str(ROOT / "nn_x/cb26"))
import lightgbm as lgb
from cbdata import load_tab
from cb_member import member_features
from au_blocks import SEQ_NAMES
t0 = time.time(); log = lambda *a: print(f"[{time.time()-t0:6.0f}s]", *a, flush=True)
tgt = sys.argv[1]; blocks = sys.argv[2].split(","); ROUNDS = int(sys.argv[3]) if len(sys.argv) > 3 else 300
m = Meta(); SP = dict(V="val", H="hold", T="test"); TAB = dict(V="val", H="trH", T="test")
tr = [k for k in SP if k != tgt]
Z = {k: w16_pool(m, SP[k]) for k in SP}; S1 = {k: np.load(OUT / f"s1_{k}.npy") for k in SP}
R = {k: (Z[k] - S1[k]).astype(np.float32) for k in SP}
names177 = member_features()


def feedcorr(a, b, off, M):
    fid = np.repeat(np.arange(len(M)), M); n = M.astype(np.float64)
    ma = np.bincount(fid, a, len(M)) / n; mb = np.bincount(fid, b, len(M)) / n
    da = a - ma[fid]; db = b - mb[fid]
    num = np.bincount(fid, da * db, len(M)); den = np.sqrt(np.bincount(fid, da * da, len(M)) * np.bincount(fid, db * db, len(M)))
    return float(np.mean(num[den > 0] / den[den > 0]))


def cond(k):
    sp = SP[k]; fid = np.repeat(np.arange(len(m.M[sp])), m.M[sp])
    Hf = (m.fv["H"].to_numpy() if k == "V" else m.ft["H"].to_numpy() if k == "T" else np.load(CACHE / "ctx_hold.npz")["H"]).astype(np.float32)
    return np.stack([S1[k], m.M[sp][fid].astype(np.float32), Hf[fid]], 1), ["s1_pred", "c_M", "c_H"]


def block(k, b):
    if b == "same177": return load_tab(TAB[k], names177), list(names177)
    if b == "emb": return np.load(CACHE / f"blk_{k}_emb.npy").astype(np.float32), [f"emb{i}" for i in range(128)]
    if b == "seq": return np.load(CACHE / f"blk_{k}_seq.npy"), list(SEQ_NAMES)
    if b == "prod": return np.load(CACHE / f"blk_{k}_prod.npy").astype(np.float32), [f"pu{i}" for i in range(128)] + [f"ph{i}" for i in range(128)]
    if b == "trd":
        tn = sorted(p.stem for p in (ROOT / "data/feats/val").glob("trd_*.npy")); return load_tab(TAB[k], tn), tn
    if b == "all_new":
        parts = [block(k, x) for x in ("emb", "seq", "prod")]
        return np.concatenate([p[0] for p in parts], 1), sum([p[1] for p in parts], [])
    raise SystemExit(b)


P = dict(objective="regression", learning_rate=0.1, num_leaves=255, min_data_in_leaf=200, feature_fraction=0.5, bagging_fraction=0.3,
         bagging_freq=1, lambda_l2=10, max_bin=63, num_threads=12, verbose=-1, seed=0)
sp = SP[tgt]; off, M = m.off[sp], m.M[sp]; y = m.y[sp]
e_w = ndcg_fast(Z[tgt], y, off, M); e_1 = ndcg_fast(S1[tgt], y, off, M)
base = dict(rho=corr(m.z(sp, S1[tgt]), Z[tgt]), feedcorr=feedcorr(S1[tgt].astype(np.float64), Z[tgt].astype(np.float64), off, M), ndcg=float(e_1.mean()), ndcg_w16=float(e_w.mean()),
            resid_var=float(np.mean(R[tgt] ** 2)))
log(f"direction {tgt}: train {'+'.join(tr)} -> read {sp} ({len(M)} feeds). stage 1 alone: {json.dumps({k: round(v, 5) for k, v in base.items()})}")
resf = OUT / f"s2_{tgt}.json"; res = json.load(open(resf)) if resf.exists() else {}
res["stage1"] = base
for b in blocks:
    Xs, ns = [], None
    for k in tr:
        c, cn = cond(k); x, xn = block(k, b); Xs.append(np.concatenate([c, x], 1)); ns = cn + xn
    cat = ["category_id"] if "category_id" in ns else "auto"
    d = lgb.Dataset(np.concatenate(Xs), np.concatenate([R[k] for k in tr]), feature_name=ns, categorical_feature=cat, free_raw_data=True); del Xs
    c, _ = cond(tgt); x, _ = block(tgt, b); Xe = np.concatenate([c, x], 1); out = {}
    def cb(env, b=b, Xe=Xe, out=out):
        it = env.iteration + 1
        if it in (100, 200) or it == ROUNDS:
            p2 = env.model.predict(Xe, num_iteration=it, num_threads=12).astype(np.float32); s = S1[tgt] + p2
            e_2 = ndcg_fast(s, y, off, M); zs = m.z(sp, s)
            eb = ndcg_fast(0.9 * Z[tgt] + 0.1 * zs, y, off, M); eb1 = ndcg_fast(0.9 * Z[tgt] + 0.1 * m.z(sp, S1[tgt]), y, off, M)
            r = dict(r2_resid=float(1 - np.mean((R[tgt] - p2) ** 2) / np.mean(R[tgt] ** 2)), rho=corr(zs, Z[tgt]), feedcorr=feedcorr(s.astype(np.float64), Z[tgt].astype(np.float64), off, M),
                     ndcg=float(e_2.mean()), d_ndcg=float((e_2 - e_1).mean()), d_ndcg_se=se(e_2 - e_1), gap_left=float((e_w - e_2).mean()), gap_left_se=se(e_w - e_2),
                     blend01=float((eb - e_w).mean()), blend01_se=se(eb - e_w), blend01_s1=float((eb1 - e_w).mean()), blend01_s1_se=se(eb1 - e_w))
            out[it] = r; log(f"  block {b:8s} ({Xe.shape[1]} cols) round {it}: R2(resid) {r['r2_resid']:.4f} rho {r['rho']:.4f} feedcorr {r['feedcorr']:.4f} ndcg {r['ndcg']:.5f} "
                             f"(vs stage 1 {r['d_ndcg']:+.5f} +- {r['d_ndcg_se']:.5f}; gap to W16 left {r['gap_left']:.5f} +- {r['gap_left_se']:.5f}) | 0.9 W16 + 0.1 this: {r['blend01']:+.5f} +- {r['blend01_se']:.5f}")
            if it == ROUNDS: np.save(OUT / f"s2_{tgt}_{b}.npy", p2)
    bst = lgb.train(P, d, ROUNDS, callbacks=[cb])
    g = bst.feature_importance("gain"); o = np.argsort(-g)[:15]
    out["top_gain"] = [(ns[i], round(float(g[i] / g.sum()), 4)) for i in o]; log("   top gain share:", out["top_gain"][:10])
    if b == "all_new":
        grp = lambda pre: float(sum(g[i] for i, n in enumerate(ns) if n.startswith(pre)) / g.sum())
        out["gain_by_block"] = dict(cond=grp("s1_pred") + float(g[ns.index("c_M")] + g[ns.index("c_H")]) / float(g.sum()), emb=grp("emb"), seq=grp("sq_"), prod_user=grp("pu"), prod_hist=grp("ph"))
        log("   gain by block:", out["gain_by_block"])
    res[b] = out; json.dump(res, open(resf, "w"), indent=1); del d, Xe
log("DONE")
