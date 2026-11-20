"""audit 3e: what the NN-minus-tree residual is made of, and the reverse test. Light: 2 threads (the one LightGBM fit uses 12; run it alone).
 r = z(W16) - stage-1 tree fit (out of sample).  usage: au_resid.py scan | item | reverse"""
from au_common import *
mode = sys.argv[1]
if mode != "reverse": light(2)
sys.path.insert(0, str(ROOT / "nn_x/cb26"))
from cbdata import load_tab
from cb_member import member_features
m = Meta(); SP = dict(V="val", H="hold", T="test")
Z = {k: w16_pool(m, SP[k]) for k in SP}


def wcorr(a, b, fid, nF, M):
    """pooled within-feed correlation (both centred per feed)."""
    a = np.nan_to_num(np.asarray(a, np.float64)); b = np.asarray(b, np.float64)
    a = a - (np.bincount(fid, a, nF) / M)[fid]; b = b - (np.bincount(fid, b, nF) / M)[fid]
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-30))


if mode == "scan":
    S1 = {k: np.load(OUT / f"s1_{k}.npy") for k in SP}
    for k in ("V", "H"):
        sp = SP[k]; off, M, y = m.off[sp], m.M[sp], m.y[sp]; nF = len(M); fid = np.repeat(np.arange(nF), M)
        r = Z[k] - S1[k]; c = dict(np.load(CACHE / f"ctx_{sp}.npz"))
        print(f"\n== {sp}: residual r = z(W16) - tree fit. var(r) {np.mean(r**2):.4f} (z has var 1), rho(fit, W16) {corr(m.z(sp, S1[k]), Z[k]):.4f}")
        print(f"   within-feed corr with the label: W16 {wcorr(Z[k], y, fid, nF, M):.4f}, tree fit {wcorr(S1[k], y, fid, nF, M):.4f}, residual {wcorr(r, y, fid, nF, M):.4f}")
        P = y > 0; pop = np.nan_to_num(c["c_it_hist_all_luo"]); cc = np.nan_to_num(c["c_cat_cnt_all"]); Hrow = c["H"][fid]; rk = np.nan_to_num(c["c_seq_max_all_rk"], nan=-1)
        print("   mean residual of POSITIVES (z units; > 0 = the NN lifts these positives above what the tree fit reproduces) and residual variance of all candidates, by bin")
        def line(lab, mk):
            if (mk & P).sum() < 30: return
            print(f"     {lab:28s} n_pos {int((mk & P).sum()):6d}  mean r on positives {r[mk & P].mean():+.4f} +- {r[mk & P].std()/np.sqrt((mk & P).sum()):.4f}   var(r) all cands {np.mean(r[mk]**2):.4f}  share of cands {mk.mean():.3f}")
        line("all", np.ones(len(r), bool))
        for lab, mk in [("cold (pop 0)", pop <= 0), ("pop 1-9", (pop > 0) & (pop < 10)), ("pop 10-99", (pop >= 10) & (pop < 100)), ("pop 100+", pop >= 100)]: line(lab, mk)
        for lab, mk in [("category unseen (H>0)", (Hrow > 0) & (cc <= 0)), ("category seen 1-9x", (cc > 0) & (cc < 10)), ("category seen 10+x", cc >= 10)]: line(lab, mk)
        for lab, mk in [("hist-sim rank bottom half", (rk >= 0) & (rk < 0.5)), ("hist-sim rank 50-80%", (rk >= 0.5) & (rk < 0.8)), ("hist-sim rank 80-95%", (rk >= 0.8) & (rk < 0.95)), ("hist-sim rank top 5%", rk >= 0.95)]: line(lab, mk)
        # correlation scan: the 177 used features and every other feature on disk for this split
        names = member_features(); used = set(names)
        if k == "V":
            allv = sorted({p.stem for p in (ROOT / "data/feats/val").glob("*.npy")} | {p.stem for p in (ROOT / "data/feats_new/val").glob("*.npy")})
            rows = []
            for n in allv:
                pth = ROOT / "data/feats/val" / f"{n}.npy"
                if not pth.exists(): pth = ROOT / "data/feats_new/val" / f"{n}.npy"
                x = np.load(pth, mmap_mode="r")
                if x.ndim != 1 or len(x) != len(r): continue
                x = np.asarray(x, np.float32)
                if not np.isfinite(x).any(): continue
                rows.append((wcorr(x, r, fid, nF, M), wcorr(x, Z[k], fid, nF, M), n, n in used))
            rows.sort(key=lambda t: -abs(t[0]))
            print("   within-feed corr of the residual with single features (val). used = one of the tree's 177 inputs. top 12 used, top 20 unused:")
            for cr, cz, n, u in [t for t in rows if t[3]][:12]: print(f"     used   {n:26s} corr(resid) {cr:+.4f}   corr(W16) {cz:+.4f}")
            for cr, cz, n, u in [t for t in rows if not t[3]][:20]: print(f"     unused {n:26s} corr(resid) {cr:+.4f}   corr(W16) {cz:+.4f}")
        # new-input blocks: single-column correlations (max |corr| over the block's columns)
        sq = np.load(CACHE / f"blk_{k}_seq.npy", mmap_mode="r"); from au_blocks import SEQ_NAMES
        cs = sorted(((wcorr(np.asarray(sq[:, j]), r, fid, nF, M), SEQ_NAMES[j]) for j in range(sq.shape[1])), key=lambda t: -abs(t[0]))
        print("   new per-event columns, top 8 |within-feed corr| with the residual:", [(n, round(c_, 4)) for c_, n in cs[:8]])
        em = np.load(CACHE / f"blk_{k}_emb.npy", mmap_mode="r"); ce = sorted((abs(wcorr(np.asarray(em[:, j], np.float32), r, fid, nF, M)) for j in range(128)), reverse=True)
        # linear probe: ridge of the residual on the per-feed-centred raw embedding (in-sample R2 upper bound for a LINEAR item-content main effect)
        E = np.asarray(em, np.float32); E = E - np.stack([np.bincount(fid, E[:, j], nF) / M for j in range(128)], 1)[fid]
        half = (np.arange(nF) % 2 == 0)[fid]; A = E[half].T @ E[half] + 1e2 * np.eye(128); w = np.linalg.solve(A, E[half].T @ r[half]); pr = E[~half] @ w
        print(f"   raw embedding dims: max single-dim |corr| {ce[0]:.4f}, median {np.median(ce):.4f}; ridge on 128 per-feed-centred dims (fit even feeds, read odd feeds): R2 of the residual {1 - np.mean((r[~half] - pr)**2) / np.mean(r[~half]**2):.4f}")

if mode == "item":
    S1 = {k: np.load(OUT / f"s1_{k}.npy") for k in SP}
    it = np.concatenate([np.load(CACHE / "ctx_val.npz")["c_item"], np.load(CACHE / "ctx_hold.npz")["c_item"], np.load(ROOT / "data/flat/test/cand_item_id.npy").astype(np.int64)])
    r = np.concatenate([Z[k] - S1[k] for k in ("V", "H", "T")]); zz = np.concatenate([Z[k] for k in ("V", "H", "T")]); s1 = np.concatenate([S1[k] for k in ("V", "H", "T")])
    rng = np.random.default_rng(0); perm = rng.permutation(len(it)); o = perm[np.argsort(it[perm], kind="stable")]; its = it[o]
    first = np.r_[True, its[1:] != its[:-1]]; start = np.where(first)[0]; cnt = np.diff(np.r_[start, len(its)])
    a = o[start[cnt >= 2]]; b = o[start[cnt >= 2] + 1]   # two random occurrences of the same item (different feeds: an item appears once per feed)
    print(f"items with >= 2 occurrences across V+H+T: {len(a)} of {len(start)} items ({cnt[cnt>=2].sum()/len(it):.3f} of rows)")
    for lab, x in (("W16 z", zz), ("tree fit", s1), ("residual r", r)):
        print(f"   same item, two different feeds: corr of {lab:11s} {np.corrcoef(x[a], x[b])[0,1]:.4f}")
    for lo, hi in ((2, 3), (3, 6), (6, 21), (21, 10**9)):
        mk = (cnt[cnt >= 2] >= lo) & (cnt[cnt >= 2] < hi)
        if mk.sum() > 100: print(f"   items with {lo}-{hi-1} occurrences ({mk.sum()} items): corr of residual {np.corrcoef(r[a][mk], r[b][mk])[0,1]:.4f}")
    print("   reading: the correlation of the residual across two feeds of the same item = the share of residual variance that is an ITEM-level effect (identity or content), the ceiling for any per-item input")

if mode == "reverse":
    import lightgbm as lgb
    key = [k for k in TREES if k.startswith("cb26_e3")][0]
    ZT = {k: m.z(SP[k], np.load(TREES[key][SP[k]])) for k in SP}
    def feats(k):
        sp = SP[k]; _, mem = w16_pool(m, sp, members=True); mem = np.asarray(mem).T; fid = np.repeat(np.arange(len(m.M[sp])), m.M[sp])
        return np.concatenate([mem, Z[k][:, None], rank_pct(m, sp, Z[k])[:, None], mem.std(1, keepdims=True), np.log(m.M[sp][fid])[:, None].astype(np.float32)], 1)
    for tgt in ("V", "H"):
        tr = [k for k in SP if k != tgt]; sp = SP[tgt]; off, M = m.off[sp], m.M[sp]
        d = lgb.Dataset(np.concatenate([feats(k) for k in tr]), np.concatenate([ZT[k] for k in tr]))
        b = lgb.train(dict(objective="regression", learning_rate=0.1, num_leaves=127, min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.5, bagging_freq=1, lambda_l2=10,
                           max_bin=255, num_threads=12, verbose=-1, seed=0), d, 300)
        p = b.predict(feats(tgt), num_threads=12).astype(np.float32); u = ZT[tgt] - p; nF = len(M); fid = np.repeat(np.arange(nF), M)
        print(f"\n== reverse, read on {sp}: tree (cb26_e3) z predicted from NN-only inputs (16 member z, pool z, pool rank, member sd, log M); train {'+'.join(tr)}")
        print(f"   linear: rho(tree, W16) {corr(ZT[tgt], Z[tgt]):.4f} (R2 {corr(ZT[tgt], Z[tgt])**2:.4f});  LightGBM on NN outputs: rho {corr(m.z(sp, p), ZT[tgt]):.4f}, R2 {1 - np.mean(u**2)/np.mean(ZT[tgt]**2):.4f}")
        if sp in m.y:
            y = m.y[sp]; print(f"   within-feed corr with the label: tree {wcorr(ZT[tgt], y, fid, nF, M):.4f}, NN-reproduced part {wcorr(p, y, fid, nF, M):.4f}, tree-only residual {wcorr(u, y, fid, nF, M):.4f}")
        if tgt == "V":
            names = member_features(); rows = sorted(((wcorr(load_tab("val", [n])[:, 0], u, fid, nF, M), n) for n in names), key=lambda t: -abs(t[0]))
            print("   tree-only residual: top 15 within-feed corr with the tree's own features:", [(n, round(c_, 3)) for c_, n in rows[:15]])
            np.save(OUT / "rev_V_treeonly.npy", u)
