"""(a) re-run the builder kernels on val, test and two train slices and compare to the stored files;
(b) independent plain-numpy reimplementation of several KEEP features on random val/test feeds."""
import sys, json
sys.path.insert(0, "<repo>/feats_v2/ctx_")
import numba as nb, numpy as np
nb.set_num_threads(16)
from build_ctx import load_meta, load_item_emb, FLAT, FEATS, _nd_kernel, ND_NAMES
from build_ctx3 import _hn_kernel, NAMES as HN_NAMES
from build_ctx4 import _hs_kernel, VARIANTS as HS_VAR
KEEP = set("ctx_hn_soft05,ctx_hn_rrw,ctx_hn_e1_rrw,ctx_hn_top1w,ctx_hn_margin,ctx_hn_bestrk,ctx_hs_t10,ctx_hs_d7,ctx_hs_k128,ctx_hs_e1,ctx_hs_k256,ctx_hs_e1k64,ctx_hs_k128t10,ctx_hs_mutual,ctx_nd_knn_u_dev,ctx_nd_ucent,ctx_nd_knn_h_dev,ctx_nd_knn_h,ctx_nd_max,ctx_nd_hcent,ctx_nd_knn_f1,ctx_nd_knn_u,ctx_nd_top5u,ctx_nd_top5h".split(","))
def run(split, fa=None, fb=None):
    feeds, M, H, co, ho = load_meta(split)
    if fa is None: fa, fb = 0, len(M)
    c0, c1 = co[fa], co[fb - 1] + M[fb - 1]; h0, h1 = ho[fa], ho[fb - 1] + H[fb - 1]
    M, H = M[fa:fb], H[fa:fb]; co = co[fa:fb] - c0; ho = ho[fa:fb] - h0
    table = np.ascontiguousarray(load_item_emb(split))
    ld = lambda p, a, b: np.asarray(np.load(p, mmap_mode="r")[a:b])
    cemb = ld(FLAT / split / "cand_emb_idx.npy", c0, c1).astype(np.int64)
    hemb = ld(FLAT / split / "hist_emb_idx.npy", h0, h1).astype(np.int64)
    het = ld(FLAT / split / "hist_event_type.npy", h0, h1)
    ue = ld(FLAT / split / "user_emb.npy", fa, fb)
    cu = ld(FEATS / split / "cos_u.npy", c0, c1).astype(np.float32); ms = ld(FEATS / split / "maxsim32.npy", c0, c1).astype(np.float32); f1 = ld(FEATS / split / "f1.npy", c0, c1).astype(np.float32)
    N = c1 - c0; new = {}
    out = np.empty((N, 18), np.float32); fo = np.empty((len(M), 4), np.float32)
    _nd_kernel(co, M, ho, H, cemb, hemb, table, ue, cu, ms, f1, fo, out)
    for i, n in enumerate(ND_NAMES):
        if n in KEEP: new[n] = out[:, i].copy()
    for suf, (Kv, dec, ev, cols) in {"": (32, .85, 0, [1, 3, 4, 5]), "soft": (32, .85, 0, [6]), "_e1": (32, .85, 1, [3])}.items():
        hm = np.ones(len(hemb), np.bool_) if ev == 0 else (het == ev)
        o = np.empty((N, 8), np.float32); _hn_kernel(co, M, ho, H, cemb, hemb, hm, table, Kv, dec, o)
        for i in cols:
            nm = HN_NAMES[i].replace("ctx_hn_", f"ctx_hn{suf}_") if suf.startswith("_") else HN_NAMES[i]
            new[nm] = o[:, i].copy()
    for nm in ["ctx_hs_t10", "ctx_hs_d7", "ctx_hs_k128", "ctx_hs_e1", "ctx_hs_k256", "ctx_hs_e1k64", "ctx_hs_k128t10"]:
        Kv, dec, tau, ev = HS_VAR[nm]
        hm = np.ones(len(hemb), np.bool_) if ev == 0 else (het == ev)
        a = np.empty(N, np.float32); b = np.empty(N, np.float32)
        _hs_kernel(co, M, ho, H, cemb, hemb, hm, table, Kv, dec, tau, a, b)
        new[nm] = a
        if nm == "ctx_hs_t10": new["ctx_hs_mutual"] = b
    assert set(new) == KEEP, set(KEEP) - set(new)
    worst = {}
    for n, v in new.items():
        s = ld(FEATS / split / f"{n}.npy", c0, c1)
        nan_mis = int((np.isnan(s) != np.isnan(v)).sum()); g = ~np.isnan(s) & ~np.isnan(v)
        worst[n] = (nan_mis, float(np.abs(s[g] - v[g]).max()) if g.any() else 0.0)
    bad = {n: w for n, w in worst.items() if w[0] > 0 or w[1] > 1e-3}
    print(split, fa, fb, "rows", N, "features", len(worst), "max|diff|", max(w[1] for w in worst.values()), "nan mismatches", sum(w[0] for w in worst.values()), "BAD:", bad, flush=True)
run("val"); run("test")
f, *_ = load_meta("train"); n = len(f)
run("train", 0, 20000); run("train", n // 2, n // 2 + 20000); run("train", n - 20000, n)

# ---------------- (b) independent numpy reimplementation
def emb(split):
    return np.asarray(load_item_emb(split)).astype(np.float32)
def check(split, nfeeds=300):
    feeds, M, H, co, ho = load_meta(split)
    T = load_item_emb(split)
    cemb = np.load(FLAT / split / "cand_emb_idx.npy", mmap_mode="r"); hemb = np.load(FLAT / split / "hist_emb_idx.npy", mmap_mode="r")
    het = np.load(FLAT / split / "hist_event_type.npy", mmap_mode="r"); ue = np.load(FLAT / split / "user_emb.npy", mmap_mode="r")
    cu = np.load(FEATS / split / "cos_u.npy", mmap_mode="r")
    st = {n: np.load(FEATS / split / f"{n}.npy", mmap_mode="r") for n in KEEP}
    rng = np.random.default_rng(7); fs = rng.choice(np.where((H > 0) & (M > 1))[0], nfeeds, replace=False)
    err = {}
    def nrm(x): return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-6)
    for fi in fs:
        ci = np.asarray(cemb[co[fi]:co[fi] + M[fi]]); hi = np.asarray(hemb[ho[fi]:ho[fi] + H[fi]]); he = np.asarray(het[ho[fi]:ho[fi] + H[fi]])
        ok = ci >= 0
        if ok.sum() < 2: continue
        V = nrm(np.asarray(T[np.where(ok, ci, 0)]).astype(np.float32)); V[~ok] = 0
        def hist(K, ev=0):
            sel = (hi >= 0) & ((he == ev) if ev else True)
            return nrm(np.asarray(T[hi[sel][:K]]).astype(np.float32))
        def mass(K, dec, tau, ev=0):
            Hm = hist(K, ev)
            if len(Hm) == 0: return None
            S = V[ok] @ Hm.T; w = dec ** np.arange(len(Hm))
            P = np.exp((S - S.max(0)) / tau); P /= P.sum(0)
            r = np.full(M[fi], np.nan); r[ok] = (P * w).sum(1) / w.sum() * M[fi]; return r
        ref = {}
        for nm, (K, dec, tau, ev) in HS_VAR.items():
            if nm in KEEP: ref[nm] = mass(K, dec, tau, ev)
        ref["ctx_hn_soft05"] = mass(32, .85, .05)
        Hm = hist(32); S = V @ Hm.T; S[~ok] = -9; w = .85 ** np.arange(len(Hm)); m = M[fi]
        R = np.argsort(np.argsort(-S, axis=0, kind="stable"), axis=0, kind="stable")  # rank of each cand per history item
        r_rrw = (w / (1 + R)).sum(1) / w.sum()
        nn = S.argmax(0)  # nearest candidate per history item
        top1w = np.bincount(nn, weights=w, minlength=m) / w.sum()
        hb = S.argmax(1)
        mutual = np.zeros(m)
        for j in range(len(Hm)):
            if hb[nn[j]] == j: mutual[nn[j]] += w[j]
        mutual /= w.sum()
        jb = S.argmax(1)
        bestrk = R[np.arange(m), jb] / (m - 1)
        margin = np.array([S[i, jb[i]] - np.max(np.delete(S[:, jb[i]], i)) for i in range(m)])
        for nm, v in [("ctx_hn_rrw", r_rrw), ("ctx_hn_top1w", top1w), ("ctx_hs_mutual", mutual), ("ctx_hn_bestrk", bestrk), ("ctx_hn_margin", margin)]:
            v = v.astype(np.float64); v[~ok] = np.nan; ref[nm] = v
        # nd: ucent, knn_u, top5u, max
        u = np.asarray(ue[fi]).astype(np.float32); un = np.linalg.norm(u)
        fm = V[ok].sum(0); fm /= np.linalg.norm(fm)
        if un > 0:
            r = np.full(m, np.nan); r[ok] = V[ok] @ (u / un) - V[ok] @ fm; ref["ctx_nd_ucent"] = r
        SS = V @ V.T
        cuf = np.asarray(cu[co[fi]:co[fi] + m]).astype(np.float64)
        knn = np.full(m, np.nan); mx = np.full(m, np.nan); t5 = np.full(m, np.nan)
        cus = np.where(ok & ~np.isnan(cuf), cuf, -9)
        order = np.argsort(-cus, kind="stable")
        for i in np.where(ok)[0]:
            o = np.where(ok & (np.arange(m) != i))[0]
            wk = np.exp(20 * (SS[i, o] - 1)); g = ~np.isnan(cuf[o])
            knn[i] = (wk[g] * cuf[o][g]).sum() / wk[g].sum(); mx[i] = SS[i, o].max()
            top = [j for j in order if j != i and cus[j] > -5][:5]; t5[i] = SS[i, top].mean()
        ref["ctx_nd_knn_u"] = knn; ref["ctx_nd_max"] = mx; ref["ctx_nd_top5u"] = t5
        for nm, v in ref.items():
            if v is None: continue
            s = np.asarray(st[nm][co[fi]:co[fi] + m]).astype(np.float64)
            g = ~np.isnan(v) & ~np.isnan(s)
            e = err.setdefault(nm, [0.0, 0, 0])
            e[0] = max(e[0], float(np.abs(v[g] - s[g]).max()) if g.any() else 0.0); e[1] += int((np.isnan(v) != np.isnan(s)).sum()); e[2] += int(g.sum())
    print(split, "independent reimplementation (max|diff|, nan mismatches, rows):")
    for k, v in sorted(err.items()): print(f"  {k:18s} {v[0]:.2e} {v[1]} {v[2]}")
check("val"); check("test")
