"""Build clu_ features for a split.

usage: python build.py <split[,split]> <group[,group]>
groups:
  hist   label-free: user's history histogram over item clusters (K=256/1024/4096), feed share of the cluster,
         cosine of the candidate to its centroid, PCA top-16 of the item embedding
  te     item-cluster CTR / COEC from train labels (position-debiased expected clicks; time-aware for train rows)
  uxi    user-cluster x item-cluster COEC (shrunk to the item-cluster COEC) and its lift over the item cluster
  time   daypart (MSK 6-h blocks) and weekday/weekend x item-cluster COEC lifts (request-time matched)
  cf     history-cluster -> candidate-cluster collaborative lift from train labels (last 16 history events)
  dsr    label-free same-day demand/supply ratio of the cluster (share in recent history events vs share in candidates)
  pmi    label-free cluster transition PMI from history sequences (all splits)
"""
import sys
sys.path.insert(0, "<repo>/feats_v2/clu_")
from clu_common import *

PRIOR = 0.0221
A_CTR, A_COEC, A_UXI, A_CF = 100.0, 20.0, 10.0, 5.0
L_CF, HALF = 16, 5.0


@nb.njit(parallel=True, cache=True)
def hist_kernel(co, M, ho, H, ccl, hcl, het, wtab, K, out):
    """out: 0 hcnt, 1 hshare, 2 hrw (recency weighted count), 3 het1 count, 4 hmin (newest hpos in cluster, 999 none),
    5 fshare (share of the feed's candidates in the same cluster)"""
    nf = len(M)
    nch = 1024
    for ch in nb.prange(nch):
        cnt = np.zeros(K, np.float32); rw = np.zeros(K, np.float32); e1 = np.zeros(K, np.float32)
        mn = np.full(K, 999.0, np.float32); fc = np.zeros(K, np.float32)
        for f in range(ch * nf // nch, (ch + 1) * nf // nch):
            o = ho[f]; h = H[f]; nv = 0
            for j in range(h):
                c = hcl[o + j]
                if c < 0:
                    continue
                nv += 1
                cnt[c] += 1; rw[c] += wtab[j]
                if het[o + j] == 1:
                    e1[c] += 1
                if mn[c] > j:
                    mn[c] = j
            q = co[f]; m = M[f]
            for i in range(m):
                c = ccl[q + i]
                if c >= 0:
                    fc[c] += 1
            for i in range(m):
                c = ccl[q + i]
                if c < 0:
                    for k in range(6):
                        out[q + i, k] = np.nan
                    continue
                out[q + i, 0] = cnt[c]
                out[q + i, 1] = cnt[c] / nv if nv > 0 else np.nan
                out[q + i, 2] = rw[c]
                out[q + i, 3] = e1[c]
                out[q + i, 4] = mn[c]
                out[q + i, 5] = fc[c] / m
            for j in range(h):
                c = hcl[o + j]
                if c >= 0:
                    cnt[c] = 0; rw[c] = 0; e1[c] = 0; mn[c] = 999.0
            for i in range(m):
                c = ccl[q + i]
                if c >= 0:
                    fc[c] = 0


@nb.njit(parallel=True, cache=True)
def cf_query(co, M, ho, H, ccl, hcl, fday, P, E, R, K, L, wtab, a, out):
    """P,E: [ncut, K*K] cumulative tables; R: [ncut, K] base COEC of the candidate cluster; fday: cut index per feed
    (-1 -> NaN). out 0: weighted mean log-lift over last L history clusters, 1: pooled smoothed ratio / R."""
    nf = len(M)
    for f in nb.prange(nf):
        d = fday[f]
        q = co[f]; m = M[f]
        hs = np.empty(L, np.int64); ws = np.empty(L, np.float64)
        n = 0
        o = ho[f]
        if d >= 0:
            for j in range(H[f]):
                c = hcl[o + j]
                if c >= 0:
                    hs[n] = c; ws[n] = wtab[j]; n += 1
                    if n == L:
                        break
        sw = 0.0
        for k in range(n):
            sw += ws[k]
        for i in range(m):
            c = ccl[q + i]
            if n == 0 or c < 0:
                out[q + i, 0] = np.nan; out[q + i, 1] = np.nan
                continue
            r = R[d, c]
            ll = 0.0; pn = 0.0; pd = 0.0
            for k in range(n):
                idx = hs[k] * K + c
                p = P[d, idx]; e = E[d, idx]
                ll += ws[k] * np.log((p + a * r) / (e + a) / r)
                pn += ws[k] * p; pd += ws[k] * e
            out[q + i, 0] = ll / sw
            out[q + i, 1] = (pn / sw + a * r) / (pd / sw + a) / r


@nb.njit(parallel=True, cache=True)
def pair_query(co, M, ho, H, ccl, hcl, S, L, wtab, out):
    """S [K,K] score of (history cluster -> candidate cluster). out 0: recency-weighted mean over the last L valid
    history clusters, 1: max over them."""
    for f in nb.prange(len(M)):
        q = co[f]; m = M[f]; o = ho[f]
        hs = np.empty(L, np.int64); ws = np.empty(L, np.float64)
        n = 0
        for j in range(H[f]):
            c = hcl[o + j]
            if c >= 0:
                hs[n] = c; ws[n] = wtab[j]; n += 1
                if n == L:
                    break
        sw = 0.0
        for k in range(n):
            sw += ws[k]
        for i in range(m):
            c = ccl[q + i]
            if n == 0 or c < 0:
                out[q + i, 0] = np.nan; out[q + i, 1] = np.nan
                continue
            s = 0.0; mx = -1e30
            for k in range(n):
                v = S[hs[k], c]
                s += ws[k] * v
                if v > mx:
                    mx = v
            out[q + i, 0] = s / sw
            out[q + i, 1] = mx


@nb.njit(parallel=True, cache=True)
def dir_score(cemb, table, grp, cut_row, Dir, out):
    for r in nb.prange(len(cemb)):
        e = cemb[r]; g = grp[r]; d = cut_row[r]
        if e < 0 or g < 0 or d < 0:
            out[r] = np.nan
            continue
        v = table[e].astype(np.float32)
        nv = np.sqrt((v * v).sum()) + 1e-9
        s = 0.0
        for k in range(v.shape[0]):
            s += v[k] * Dir[d, g, k]
        out[r] = s / nv


@nb.njit(parallel=True, fastmath=True, cache=True)
def dup_kernel(co, M, cemb, table, ccl, dot, f1, out):
    """out 0: # other candidates in the feed with cos >= 0.95 (near duplicates), 1: max cos to another candidate,
    2: pct rank of dot(user,item) among same-cluster (K=4096) candidates of the feed (1 = best, NaN if alone),
    3: same for f1"""
    for f in nb.prange(len(M)):
        q = co[f]; m = M[f]
        V = np.zeros((m, 128), np.float32)
        ok = np.zeros(m, np.bool_)
        for i in range(m):
            e = cemb[q + i]
            if e >= 0:
                v = table[e].astype(np.float32)
                V[i] = v / (np.sqrt((v * v).sum()) + 1e-9)
                ok[i] = True
        for i in range(m):
            if not ok[i]:
                for k in range(4):
                    out[q + i, k] = np.nan
                continue
            nd = 0; mx = -1.0
            for j in range(m):
                if j == i or not ok[j]:
                    continue
                s = 0.0
                for k in range(128):
                    s += V[i, k] * V[j, k]
                if s >= 0.95:
                    nd += 1
                if s > mx:
                    mx = s
            out[q + i, 0] = nd
            out[q + i, 1] = mx if m > 1 else np.nan
            c = ccl[q + i]
            ng = 0; bd = 0.0; bf = 0.0
            for j in range(m):
                if j != i and ccl[q + j] == c:
                    ng += 1
                    if dot[q + j] < dot[q + i]:
                        bd += 1.0
                    elif dot[q + j] == dot[q + i]:
                        bd += 0.5
                    if f1[q + j] < f1[q + i]:
                        bf += 1.0
                    elif f1[q + j] == f1[q + i]:
                        bf += 0.5
            out[q + i, 2] = bd / ng if ng > 0 else np.nan
            out[q + i, 3] = bf / ng if ng > 0 else np.nan


@nb.njit(parallel=True, cache=True)
def het_kernel(co, M, ho, H, ccl, hcl, het, K, out):
    """out[:, e-1] = # history events of event_type e (1..10) in the candidate's cluster"""
    nf = len(M)
    nch = 1024
    for ch in nb.prange(nch):
        cnt = np.zeros((K, 11), np.float32)
        for f in range(ch * nf // nch, (ch + 1) * nf // nch):
            o = ho[f]; h = H[f]
            for j in range(h):
                c = hcl[o + j]
                if c >= 0:
                    cnt[c, het[o + j]] += 1
            q = co[f]
            for i in range(M[f]):
                c = ccl[q + i]
                for e in range(1, 11):
                    out[q + i, e - 1] = cnt[c, e] if c >= 0 else np.nan
            for j in range(h):
                c = hcl[o + j]
                if c >= 0:
                    for e in range(11):
                        cnt[c, e] = 0


def rel_rank(x, co, M):
    """within-feed percentile rank (NaN stays NaN)"""
    out = np.empty(len(x), np.float32)
    _rel_rank(x, co, M, out)
    return out


@nb.njit(parallel=True, cache=True)
def _rel_rank(x, co, M, out):
    for f in nb.prange(len(M)):
        q = co[f]; m = M[f]
        v = x[q:q + m].copy()
        for i in range(m):
            if np.isnan(v[i]):
                v[i] = -np.inf
        order = np.argsort(v)
        i = 0
        while i < m:
            j = i
            while j + 1 < m and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                out[q + order[k]] = (i + j) / 2.0 / (m - 1) if m > 1 else 0.5
            i = j + 1
        for i in range(m):
            if np.isnan(x[q + i]):
                out[q + i] = np.nan


# --------------------------------------------------------------------------- cut handling
def cut_tables(split, fday_feed, arrs):
    """Return (tables indexed by cut, cut index per feed). val/test -> a single cut = sum over all train days.
    train -> cut d = sum over days < d (cut 0 is empty -> feature NaN)."""
    if split == "train":
        cum = {k: np.concatenate([np.zeros_like(v[:1]), np.cumsum(v, axis=0)[:-1]]) for k, v in arrs.items()}
        cut = fday_feed.copy()
        cut[cut <= 0] = -1
        return cum, cut
    tot = {k: v.sum(0, keepdims=True) for k, v in arrs.items()}
    return tot, np.zeros(len(fday_feed), np.int64)


def gather(tab, cut_row, key_row):
    """tab [ncut, nkey]; returns tab[cut, key] with NaN where cut<0 or key<0"""
    ok = (cut_row >= 0) & (key_row >= 0)
    out = np.full(len(cut_row), np.nan, np.float32)
    out[ok] = tab[cut_row[ok], key_row[ok]]
    return out


def build(split, groups):
    nb.set_num_threads(24)
    fe = feeds(split)
    M, H, co, ho = fe["M"], fe["H"], fe["co"], fe["ho"]
    fidx = np.load(FLAT / split / "cand_fidx.npy")
    N = len(fidx)
    fday = day_index(fe["ts"])
    if "hist" in groups:
        het = np.load(FLAT / split / "hist_event_type.npy")
        wtab = (0.5 ** (np.arange(256) / HALF)).astype(np.float32)
        for K in IKS:
            out = np.empty((N, 6), np.float32)
            hist_kernel(co, M, ho, H, np.asarray(cand_cl(split, K)), np.asarray(hist_cl(split, K)), het, wtab, K, out)
            for i, nm in enumerate(["hcnt", "hshare", "hrw", "het1", "hmin", "fshare"]):
                save_feat(split, f"clu_{nm}{K}", out[:, i])
            del out
            save_feat(split, f"clu_ccos{K}", cand_ccos(split, K))
            log(split, "hist", K)
        # PCA top-16
        mu = np.load(WORK / "pca_mean.npy"); comp = np.load(WORK / "pca_comp.npy")
        e = np.load(FLAT / split / "cand_emb_idx.npy")
        d = ROOT / "competition" / split / "item_embeddings"
        meta = json.loads((d / "data.meta").read_text())
        tab = np.memmap(d / "data.memmap", dtype=np.int8, mode="r", shape=tuple(meta["shape"]))
        out = np.full((N, 16), np.nan, np.float32)
        for s in range(0, N, 4_000_000):
            ee = e[s:s + 4_000_000]
            ok = ee >= 0
            v = np.asarray(tab[ee[ok]]).astype(np.float32)
            v /= np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-6)
            out[s:s + 4_000_000][ok] = (v - mu) @ comp.T
        for i in range(16):
            save_feat(split, f"clu_pca{i}", out[:, i])
        log(split, "pca")
    if "te" in groups:
        for K in IKS:
            z = dict(np.load(WORK / f"te_ic{K}.npz"))
            tabs, cut = cut_tables(split, fday, z)
            c = np.asarray(cand_cl(split, K)).astype(np.int64)
            cr = cut[fidx]
            imp, ps, ex = gather(tabs["imp"], cr, c), gather(tabs["pos"], cr, c), gather(tabs["ex"], cr, c)
            save_feat(split, f"clu_ic{K}_ctr", (ps + PRIOR * A_CTR) / (imp + A_CTR))
            save_feat(split, f"clu_ic{K}_coec", (ps + A_COEC) / (ex + A_COEC))
            log(split, "te", K)
    if "uxi" in groups:
        for Ku in UKS:
            ucl = user_cl(split, Ku)
            for Ki in (256, 1024, 4096):
                p = WORK / f"te_u{Ku}i{Ki}.npz"
                if not p.exists():
                    continue
                zi = dict(np.load(WORK / f"te_ic{Ki}.npz"))
                ti, cut = cut_tables(split, fday, zi)
                z = dict(np.load(p))
                tu, _ = cut_tables(split, fday, z)
                c = np.asarray(cand_cl(split, Ki)).astype(np.int64)
                cr = cut[fidx]
                Ri = (gather(ti["pos"], cr, c) + A_COEC) / (gather(ti["ex"], cr, c) + A_COEC)
                key = np.where((c >= 0) & (ucl[fidx] >= 0), ucl[fidx] * Ki + c, -1)
                P, E = gather(tu["pos"], cr, key), gather(tu["ex"], cr, key)
                coec = (P + A_UXI * Ri) / (E + A_UXI)
                save_feat(split, f"clu_u{Ku}i{Ki}_coec", coec)
                save_feat(split, f"clu_u{Ku}i{Ki}_lift", np.log(coec / Ri))
                log(split, "uxi", Ku, Ki)
    if "time" in groups:
        h = ((fe["ts"] + 3 * 3600) // 3600) % 24
        dp = (h // 6)[fidx]
        dayr = fday[fidx]
        wk = ((dayr % 7 == 0) | (dayr % 7 == 6)).astype(np.int64)
        for K in (256, 1024):
            zi = dict(np.load(WORK / f"te_ic{K}.npz"))
            ti, cut = cut_tables(split, fday, zi)
            c = np.asarray(cand_cl(split, K)).astype(np.int64)
            cr = cut[fidx]
            Ri = (gather(ti["pos"], cr, c) + A_COEC) / (gather(ti["ex"], cr, c) + A_COEC)
            for nm, nlev, lev in (("dp", 4, dp), ("wk", 2, wk)):
                z = dict(np.load(WORK / f"te_{nm}{K}.npz"))
                tt, _ = cut_tables(split, fday, z)
                key = np.where(c >= 0, lev * K + c, -1)
                P, E = gather(tt["pos"], cr, key), gather(tt["ex"], cr, key)
                coec = (P + A_COEC * Ri) / (E + A_COEC)
                save_feat(split, f"clu_{nm}{K}_lift", np.log(coec / Ri))
            log(split, "time", K)
    if "cf" in groups:
        wtab = 0.5 ** (np.arange(256) / HALF)
        for K in (256, 1024):
            p = WORK / f"cf_h{K}.npz"
            if not p.exists():
                continue
            z = dict(np.load(p))
            zi = dict(np.load(WORK / f"te_ic{K}.npz"))
            tc, cut = cut_tables(split, fday, z)
            ti, _ = cut_tables(split, fday, zi)
            R = ((ti["pos"] + A_COEC) / (ti["ex"] + A_COEC)).astype(np.float64)
            out = np.empty((N, 2), np.float32)
            cf_query(co, M, ho, H, np.asarray(cand_cl(split, K)), np.asarray(hist_cl(split, K)), cut,
                     tc["pos"].astype(np.float64), tc["ex"].astype(np.float64), R, K, L_CF, wtab, A_CF, out)
            save_feat(split, f"clu_cf{K}_ll", out[:, 0])
            save_feat(split, f"clu_cf{K}_ratio", out[:, 1])
            log(split, "cf", K)
    if "pmi" in groups:
        wtab = 0.5 ** (np.arange(256) / HALF)
        for K in (256, 1024):
            z = np.load(WORK / f"pmi_{K}.npz")
            for nm, T in (("pmi", z["T"]), ("pmi1", z["T1"])):
                pb = T.sum(0) / T.sum()
                pba = (T + 10.0 * pb[None, :]) / (T.sum(1, keepdims=True) + 10.0)
                S = np.log(pba / np.maximum(pb[None, :], 1e-12)).astype(np.float64)
                out = np.empty((N, 2), np.float32)
                pair_query(co, M, ho, H, np.asarray(cand_cl(split, K)), np.asarray(hist_cl(split, K)), S, L_CF, wtab, out)
                save_feat(split, f"clu_{nm}{K}_mean", out[:, 0])
                save_feat(split, f"clu_{nm}{K}_max", out[:, 1])
            log(split, "pmi", K)
    if "etr" in groups:
        for K in IKS:
            cnt = np.load(WORK / f"etr_{K}.npz")["cnt"][:, 1:]
            pe = cnt.sum(0) / cnt.sum()
            lift = np.log((cnt + 20.0 * pe[None, :]) / (cnt.sum(1, keepdims=True) + 20.0) / pe[None, :])
            c = np.asarray(cand_cl(split, K)).astype(np.int64)
            ok = c >= 0
            for e in (range(10) if K == 1024 else range(1)):
                v = np.full(N, np.nan, np.float32)
                v[ok] = lift[c[ok], e]
                save_feat(split, f"clu_etr{K}_e{e+1}", v)
            log(split, "etr", K)
    if "hod" in groups:
        hr = (((fe["ts"] + 3 * 3600) // 3600) % 24)[fidx]
        for K in (256, 1024):
            dem = np.load(WORK / f"hod_{K}.npz")["dem"]
            pc = dem.sum(0) / dem.sum()
            lift = np.log((dem + 50.0 * pc[None, :]) / (dem.sum(1, keepdims=True) + 50.0) / pc[None, :])
            c = np.asarray(cand_cl(split, K)).astype(np.int64)
            ok = c >= 0
            v = np.full(N, np.nan, np.float32)
            v[ok] = lift[hr[ok], c[ok]]
            save_feat(split, f"clu_hod{K}_lift", v)
            log(split, "hod", K)
    if "frel" in groups:
        # f1..f5 relative to the item's embedding cluster: z-score vs cluster mean/std over val+test candidate rows
        for K in (256, 1024):
            pt = WORK / f"fstat_{K}.npz"
            if not pt.exists():
                st = {}
                for k in range(1, 6):
                    s1 = np.zeros(K); s2 = np.zeros(K); n0 = np.zeros(K)
                    for sp in ("val", "test"):
                        c = np.asarray(cand_cl(sp, K)).astype(np.int64)
                        x = np.load(FLAT / sp / f"cand_f{k}.npy").astype(np.float64)
                        ok = c >= 0
                        n0 += np.bincount(c[ok], minlength=K); s1 += np.bincount(c[ok], x[ok], minlength=K)
                        s2 += np.bincount(c[ok], x[ok] ** 2, minlength=K)
                    mu = s1 / np.maximum(n0, 1)
                    st[f"mu{k}"] = mu; st[f"sd{k}"] = np.sqrt(np.maximum(s2 / np.maximum(n0, 1) - mu ** 2, 1e-6))
                np.savez(pt, **st)
            st = np.load(pt)
            c = np.asarray(cand_cl(split, K)).astype(np.int64)
            ok = c >= 0
            for k in range(1, 6):
                x = np.load(FLAT / split / f"cand_f{k}.npy").astype(np.float32)
                v = np.full(N, np.nan, np.float32)
                v[ok] = (x[ok] - st[f"mu{k}"][c[ok]]) / st[f"sd{k}"][c[ok]]
                save_feat(split, f"clu_frel{K}_f{k}", v)
            log(split, "frel", K)
    if "roc" in groups:
        d = ROOT / "competition" / split / "item_embeddings"
        meta = json.loads((d / "data.meta").read_text())
        table = np.fromfile(d / "data.memmap", dtype=np.int8).reshape(meta["shape"])
        cemb = np.load(FLAT / split / "cand_emb_idx.npy")
        zg = dict(np.load(WORK / "roc_g.npz"))
        tg, cut = cut_tables(split, fday, zg)
        cr = cut[fidx].astype(np.int64)
        mp = tg["Sp"] / np.maximum(tg["P"], 1e-9)[..., None]   # [ncut, 1, 128]
        me = tg["Se"] / np.maximum(tg["E"], 1e-9)[..., None]
        for name in ["g", "u64", "u256", "u1024", "i256", "i1024"]:
            z = dict(np.load(WORK / f"roc_{name}.npz"))
            tt, _ = cut_tables(split, fday, z)
            a = 0.0 if name == "g" else 20.0
            Dir = ((tt["Sp"] + a * mp) / (tt["P"][..., None] + a + 1e-9)
                   - (tt["Se"] + a * me) / (tt["E"][..., None] + a + 1e-9)).astype(np.float32)
            if name == "g":
                grp = np.zeros(N, np.int64)
            elif name[0] == "u":
                grp = user_cl(split, int(name[1:]))[fidx]
            else:
                grp = np.asarray(cand_cl(split, int(name[1:]))).astype(np.int64)
            out = np.empty(N, np.float32)
            dir_score(cemb, table, grp, cr, Dir, out)
            save_feat(split, f"clu_roc_{name}", out)
            log(split, "roc", name)
    if "dup" in groups:
        d = ROOT / "competition" / split / "item_embeddings"
        meta = json.loads((d / "data.meta").read_text())
        table = np.fromfile(d / "data.memmap", dtype=np.int8).reshape(meta["shape"])
        cemb = np.load(FLAT / split / "cand_emb_idx.npy")
        ue = np.load(FLAT / split / "user_emb.npy").astype(np.float32)
        dot = np.nan_to_num(np.load(FEATS / split / "dot_u.npy"), nan=-1e9).astype(np.float32)
        f1 = np.load(FLAT / split / "cand_f1.npy").astype(np.float32)
        out = np.empty((N, 4), np.float32)
        dup_kernel(co, M, cemb, table, np.asarray(cand_cl(split, 4096)), dot, f1, out)
        for i, nm in enumerate(["dup95", "maxcos_feed", "grp_rk_dot", "grp_rk_f1"]):
            save_feat(split, f"clu_{nm}", out[:, i])
        log(split, "dup")
    if "het" in groups:
        het = np.load(FLAT / split / "hist_event_type.npy").astype(np.int64)
        for K in (1024,):
            out = np.empty((N, 10), np.float32)
            het_kernel(co, M, ho, H, np.asarray(cand_cl(split, K)), np.asarray(hist_cl(split, K)), het, K, out)
            for e in range(2, 11):
                save_feat(split, f"clu_het{e}_{K}", out[:, e - 1])
            log(split, "het", K)
    if "dsr" in groups:
        for K in (256, 1024):
            z = np.load(WORK / f"dsr_{K}.npz")
            dem = (z["dem"] + 5.0) / (z["dem"].sum(1, keepdims=True) + 5.0 * K)
            sup = (z["sup"] + 5.0) / (z["sup"].sum(1, keepdims=True) + 5.0 * K)
            c = np.asarray(cand_cl(split, K)).astype(np.int64)
            dr = fday[fidx]
            ok = c >= 0
            r = np.full(N, np.nan, np.float32); dm = np.full(N, np.nan, np.float32)
            r[ok] = np.log(dem[dr[ok], c[ok]] / sup[dr[ok], c[ok]])
            dm[ok] = np.log(dem[dr[ok], c[ok]])
            save_feat(split, f"clu_dsr{K}", r)
            save_feat(split, f"clu_dem{K}", dm)
            log(split, "dsr", K)


if __name__ == "__main__":
    for s in sys.argv[1].split(","):
        build(s, sys.argv[2].split(","))
