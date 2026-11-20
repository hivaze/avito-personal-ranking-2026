"""audit 3f: does a block of new inputs close the gap where it lives? Positive-level relative loss vs W16 (val and holdout) for
the stage-1 tree fit (177 features -> W16 score), stage 1 + each stage-2 block, and the tree models of record. Light: 2 threads."""
from au_common import *
light(2)
m = Meta()


def contrib(s, y, off, M):
    n = len(M); fid = np.repeat(np.arange(n), M)
    order = np.lexsort((y, -s.astype(np.float64), fid))
    rank = np.empty(len(s), np.int64); rank[order] = np.arange(len(s)) - np.repeat(off, M)
    npos = np.bincount(fid, weights=y, minlength=n)
    cs = np.r_[0, np.cumsum(1.0 / np.log2(np.arange(M.max()) + 2.0))]
    c = np.where(y > 0, 1.0 / np.log2(rank + 2.0) / np.maximum(cs[npos.astype(np.int64)][fid], 1e-12), 0.0)
    return c, fid


for k, sp in (("V", "val"), ("H", "hold")):
    y, off, M = m.y[sp], m.off[sp], m.M[sp]; c = dict(np.load(CACHE / f"ctx_{sp}.npz")); nF = len(M)
    S = {"W16": w16_pool(m, sp)}
    if (OUT / f"s1_{k}.npy").exists():
        s1 = np.load(OUT / f"s1_{k}.npy"); S["s1 (177 feats)"] = s1
        for b in ("same177", "emb", "seq", "prod", "all_new"):
            f = OUT / f"s2_{k}_{b}.npy"
            if f.exists(): S[f"s1+{b}"] = s1 + np.load(f)
    S["cb26_e3"] = m.z(sp, np.load(TREES[[t for t in TREES if t.startswith("cb26_e3")][0]][sp]))
    S["K134 r600"] = m.z(sp, np.load(TREES[[t for t in TREES if t.startswith("lgb_K_xm_134118")][0]][sp]))
    suf = 'val' if sp == 'val' else 'trH'
    for nm in ("KL0_60k_r600", "KL0_60k_r2400", "KL1_60k_r1000"):
        if (OUT / f"{nm}_{suf}.npy").exists(): S[nm] = np.load(OUT / f"{nm}_{suf}.npy")
    for nm, lab in ((f"s1j_{k}_emb+seq+prod", "joint 177+new r800"), (f"s1j_{k}_none_lr0.2", "177 only, lr0.2 r1000"), (f"s1j_{k}_emb+seq+prod_lr0.2", "joint, lr0.2 r1000")):
        if (OUT / f"{nm}.npy").exists(): S[lab] = np.load(OUT / f"{nm}.npy")
    C = {}; fid = None
    for n, s in S.items(): C[n], fid = contrib(np.asarray(s, np.float32), y, off, M)
    E = {n: np.where(np.bincount(fid, y, nF) > 0, np.bincount(fid, C[n], nF), 1.0) for n in C}
    names = [n for n in S if n != "W16"]
    print(f"\n================ {sp.upper()} ({nF} feeds): NDCG and gap to W16 {E['W16'].mean():.5f}")
    for n in names: print(f"   {n:20s} {E[n].mean():.5f}   gap {fmt(E['W16'] - E[n])}   rho to W16 {corr(m.z(sp, S[n]), S['W16']):.4f}")
    P = y > 0; pop = np.nan_to_num(c["c_it_hist_all_luo"]); cc = np.nan_to_num(c["c_cat_cnt_all"]); Hrow = c["H"][fid]; rk = np.nan_to_num(c["c_seq_max_all_rk"], nan=-1)
    groups = [("all positives", np.ones(len(y), bool)), ("cold (pop 0)", pop <= 0), ("pop 1-9", (pop > 0) & (pop < 10)), ("pop 10-99", (pop >= 10) & (pop < 100)), ("pop 100+", pop >= 100),
              ("category unseen (H>0)", (Hrow > 0) & (cc <= 0)), ("category seen 1-9x", (cc > 0) & (cc < 10)), ("category seen 10+x", cc >= 10),
              ("hist-sim bottom half", (rk >= 0) & (rk < 0.5)), ("hist-sim 50-80%", (rk >= 0.5) & (rk < 0.8)), ("hist-sim 80-95%", (rk >= 0.8) & (rk < 0.95)), ("hist-sim top 5%", rk >= 0.95)]
    print("   relative loss of the positives' NDCG contribution vs W16, % (share of the model's total gap in brackets)")
    print(f"   {'bin':24s} {'share pos':>9s}" + "".join(f" | {n[:21]:>21s}" for n in names))
    for lab, mk in groups:
        mk = mk & P; s = f"   {lab:24s} {mk.sum()/P.sum():9.3f}"
        for n in names:
            tot = (E["W16"] - E[n]).sum()
            s += f" | {100*(1 - C[n][mk].sum()/C['W16'][mk].sum()):10.1f}% ({100*(C['W16'][mk] - C[n][mk]).sum()/tot:5.1f})"
        print(s)
