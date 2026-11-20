"""P3 stage 0 (PRETRAIN_PLAN_1113 sec. 4, E4-reduced): within-category whitened / CSLS similarity as a retrieval-space proxy.
Gate pre-declared in nn_x/pretrain2/P3/PREDECLARED.txt (written before any number). CPU only, 12 threads, no training.

Stages (cached in nn_x/pretrain2/P3/out/):
  fit   : mu_k, per-category shrunk ZCA W_k, pooled W, CSLS reference pools -- train item table only, label-free.
  sims  : per split (val, holdout = 30K random-like train feeds): per-candidate signals in 4 spaces
          (S0 raw, S1 per-cat whitened, S1g pooled-W whitened, S2 CSLS) x 5 signals (maxall, top16 mean, last32 max, rec_hl3, last5)
          + top-16 Jaccard vs raw.
  gate  : Jaccard summary, cell pair AUCs (+ anti-popularity controls), single-feature NDCG, nested residuals on G3 (val + holdout), pmix.
"""
import os
for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "POLARS_MAX_THREADS", "NUMBA_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
    os.environ[k] = "12"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys, time, json, argparse
from pathlib import Path
import numpy as np, polars as pl, numba as nb

ROOT = Path("<repo>")
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "eda/errors/synthesis"))
FLAT = ROOT / "data/flat"; FEATS = ROOT / "data/feats"; P = ROOT / "preds"; COMP = ROOT / "data/competition"
OUT = ROOT / "nn_x/pretrain2/P3/out"; OUT.mkdir(parents=True, exist_ok=True)
T0 = time.time()
LOG = None
SPACES = ["S0raw", "S1cat", "S1pool", "S2csls"]
SIGS = ["maxall", "top16", "last32", "rec_hl3", "last5"]
NC = 64
G3_MEMBERS = ([f"v23_gru_pre2s_s{s}" for s in range(501, 507)] + [f"r6_cd_m02_r{r}" for r in range(101, 108)]
              + [f"r6_cd_m02_p70k_r{r}" for r in range(101, 108)])


def log(*a):
    s = f"[{time.time() - T0:7.1f}s] " + " ".join(str(x) for x in a)
    print(s, flush=True)
    if LOG is not None:
        LOG.write(s + "\n"); LOG.flush()


def memmap_table(split):
    meta = json.loads((COMP / split / "item_embeddings/data.meta").read_text())
    return np.memmap(COMP / split / "item_embeddings/data.memmap", dtype=np.int8, mode="r", shape=tuple(meta["shape"]))


def unit_rows(tab, idx):
    """float32 unit vectors of table rows idx (idx sorted unique for sequential access)."""
    x = np.asarray(tab[idx], dtype=np.float32)
    n = np.sqrt((x * x).sum(1, keepdims=True))
    return x / np.maximum(n, 1e-3)


def holdout_index():
    """nn/train.py / nn_x/r6/train.py 30K random-like holdout (rng 12345 over p_random > 0.9 and M >= 50)."""
    pr = np.load(ROOT / "eda/labels_position/out/train_prandom.npz")
    p = np.nan_to_num(pr["p_random"].astype(np.float32), nan=0.0)
    Mt = pl.read_parquet(FLAT / "train/feeds.parquet", columns=["M"])["M"].to_numpy().astype(np.int64)
    p[Mt < 50] = 0.0
    cand = np.where((p > 0.9) & (Mt >= 50))[0]
    return np.sort(np.random.default_rng(12345).choice(cand, min(30000, len(cand)), replace=False))


# ============================================================ fit (train table only, label-free)
def stage_fit():
    rng = np.random.default_rng(0)
    he = np.load(FLAT / "train/hist_emb_idx.npy", mmap_mode="r"); hc = np.load(FLAT / "train/hist_category_id.npy", mmap_mode="r")
    ce = np.load(FLAT / "train/cand_emb_idx.npy", mmap_mode="r"); cc = np.load(FLAT / "train/cand_category_id.npy", mmap_mode="r")
    ph = np.sort(rng.choice(len(he), 4_000_000, replace=False)); pc = np.sort(rng.choice(len(ce), 2_500_000, replace=False))
    hE, hC = np.asarray(he[ph]).astype(np.int64), np.asarray(hc[ph]).astype(np.int64)
    cE, cC = np.asarray(ce[pc]).astype(np.int64), np.asarray(cc[pc]).astype(np.int64)
    log(f"fit sample: {len(ph):,} train history events ({(hE >= 0).mean():.3f} with content), {len(pc):,} train candidate rows")
    E = np.concatenate([hE, cE]); C = np.concatenate([hC, cC]); ok = E >= 0
    E, C = E[ok], C[ok]
    uE, first = np.unique(E, return_index=True); uC = C[first]
    if len(uE) > 5_000_000:
        s = np.sort(rng.choice(len(uE), 5_000_000, replace=False)); uE, uC = uE[s], uC[s]
    tab = memmap_table("train")
    X = unit_rows(tab, uE)
    log(f"fit items: {len(uE):,} distinct train-table items; random-pair raw cos "
        f"{float((X[rng.integers(0, len(X), 200000)] * X[rng.integers(0, len(X), 200000)]).sum(1).mean()):.3f}")
    D = X.shape[1]
    mu = np.zeros((NC, D)); Sig = np.zeros((NC, D, D)); nk = np.zeros(NC, np.int64)
    gmu = X.mean(0, dtype=np.float64)
    o = np.argsort(uC, kind="stable"); cs = uC[o]
    bounds = np.searchsorted(cs, np.arange(NC + 1))
    for k in range(NC):
        r = o[bounds[k]:bounds[k + 1]]
        nk[k] = len(r)
        if len(r) < 2:
            continue
        Xk = X[r].astype(np.float64)
        mu[k] = Xk.mean(0)
        Xc = Xk - mu[k]
        Sig[k] = Xc.T @ Xc / len(r)
    Spool = (Sig * nk[:, None, None]).sum(0) / nk.sum()
    N0 = 5000.0
    Wk = np.zeros((NC, D, D), np.float32); lam = np.zeros(NC)
    for k in range(NC):
        if nk[k] < 2:
            mu[k] = gmu
        Sk = (nk[k] * Sig[k] + N0 * Spool) / (nk[k] + N0)
        lam[k] = 0.1 * np.trace(Sk) / D
        ev, V = np.linalg.eigh(Sk)
        Wk[k] = (V * (1.0 / np.sqrt(np.maximum(ev, 0) + lam[k]))) @ V.T
    lp = 0.1 * np.trace(Spool) / D
    ev, V = np.linalg.eigh(Spool)
    Wp = ((V * (1.0 / np.sqrt(np.maximum(ev, 0) + lp))) @ V.T).astype(np.float32)
    evs = np.sort(ev)[::-1]
    log(f"categories with items: {(nk > 0).sum()} (min n {nk[nk > 0].min():,}, median {int(np.median(nk[nk > 0])):,}); pooled within-cat "
        f"cov: trace {np.trace(Spool):.4f}, top-5 eigen share {evs[:5].sum() / evs.sum():.3f}, eff. rank "
        f"{evs.sum() ** 2 / (evs ** 2).sum():.1f}; total cov trace {np.trace(np.cov(X[::10].T)):.4f}")
    # linear-probe-free check of category dominance: share of total variance explained by category means
    tot = ((X[::5].astype(np.float64) - gmu) ** 2).sum(1).mean()
    log(f"between-category variance share: {1 - np.trace(Spool) / tot:.3f}")
    # CSLS reference pools (event-sampled history items, row-sampled candidate items)
    hu = np.unique(hE[hE >= 0]); cu = np.unique(cE[cE >= 0])
    poolH = np.sort(rng.choice(hu, 50000, replace=False)); poolC = np.sort(rng.choice(cu, 50000, replace=False))
    PH = unit_rows(tab, poolH); PC = unit_rows(tab, poolC)
    np.savez(OUT / "fit.npz", mu=mu.astype(np.float32), Wk=Wk, Wp=Wp, nk=nk, lam=lam, PH=PH, PC=PC)
    log("fit saved")


# ============================================================ per-split data
def split_data(name):
    if name == "val":
        f = pl.read_parquet(FLAT / "val/feeds.parquet")
        M = f["M"].to_numpy().astype(np.int64); H = f["H"].to_numpy().astype(np.int64)
        coff = f["cand_off"].to_numpy().astype(np.int64); hoff = f["hist_off"].to_numpy().astype(np.int64)
        d = dict(M=M, H=H, users=f["user_id"].to_numpy().astype(np.int64), tab="val")
        crows = np.arange(int(M.sum())); hrows = np.arange(int(H.sum())) if hoff[-1] + H[-1] == H.sum() else None
        assert coff[0] == 0 and (np.diff(coff) == M[:-1]).all() and hrows is not None and (np.diff(hoff) == H[:-1]).all()
        src = "val"
    else:
        f = pl.read_parquet(FLAT / "train/feeds.parquet")
        hold = holdout_index()
        Mall = f["M"].to_numpy().astype(np.int64); Hall = f["H"].to_numpy().astype(np.int64)
        ca = f["cand_off"].to_numpy().astype(np.int64); ha = f["hist_off"].to_numpy().astype(np.int64)
        M, H = Mall[hold], Hall[hold]
        crows = np.repeat(ca[hold] - np.concatenate([[0], np.cumsum(M)[:-1]]), M) + np.arange(int(M.sum()))
        hrows = np.repeat(ha[hold] - np.concatenate([[0], np.cumsum(H)[:-1]]), H) + np.arange(int(H.sum()))
        hm = np.load(P / "holdout/holdout_meta.npy"); assert (hm[0] == M).all()
        d = dict(M=M, H=H, users=f["user_id"].to_numpy().astype(np.int64)[hold], tab="train", hold=hold)
        src = "train"
    def g(kind, nm, rows):
        a = np.load(FLAT / src / f"{kind}_{nm}.npy", mmap_mode="r")
        return np.asarray(a[rows]) if name != "val" else np.asarray(a)
    d["cemb"] = g("cand", "emb_idx", crows).astype(np.int64); d["ccat"] = g("cand", "category_id", crows).astype(np.int64)
    d["y"] = g("cand", "target", crows).astype(np.float64)
    d["hemb"] = g("hist", "emb_idx", hrows).astype(np.int64); d["hitem"] = g("hist", "item_id", hrows).astype(np.int64)
    d["coff"] = np.concatenate([[0], np.cumsum(d["M"])[:-1]]).astype(np.int64)
    d["hoff"] = np.concatenate([[0], np.cumsum(d["H"])[:-1]]).astype(np.int64)
    d["crows"] = crows
    if name == "holdout":
        assert (d["y"] == np.load(P / "holdout/holdout_y.npy")).all()
    return d


@nb.njit(parallel=True, cache=True)
def dedup_count(hoff, H, hemb, hitem, ndd, nev):
    for f in nb.prange(len(H)):
        o = hoff[f]; c = 0; e = 0
        seen = np.empty(H[f], np.int64)
        for j in range(H[f]):
            if hemb[o + j] < 0:
                continue
            e += 1
            it = hitem[o + j]; new = True
            for q in range(c):
                if seen[q] == it:
                    new = False; break
            if new:
                seen[c] = it; c += 1
        ndd[f] = c; nev[f] = e


@nb.njit(parallel=True, cache=True)
def dedup_fill(hoff, H, hemb, hitem, hloc, dd_off, ev_off, dd_loc, ev_slot):
    for f in nb.prange(len(H)):
        o = hoff[f]; c = 0; e = 0
        seen = np.empty(H[f], np.int64)
        for j in range(H[f]):
            if hemb[o + j] < 0:
                continue
            it = hitem[o + j]; slot = -1
            for q in range(c):
                if seen[q] == it:
                    slot = q; break
            if slot < 0:
                seen[c] = it; dd_loc[dd_off[f] + c] = hloc[o + j]; slot = c; c += 1
            ev_slot[ev_off[f] + e] = slot; e += 1


@nb.njit(cache=True, fastmath=True)
def _signals(s, es, ne, w, out_r, sp, topk_out):
    nh = len(s)
    mx = -1e30
    for j in range(nh):
        if s[j] > mx:
            mx = s[j]
    out_r[sp, 0] = mx
    o = np.argsort(-s)
    t = min(16, nh)
    a = 0.0
    for q in range(t):
        a += s[o[q]]; topk_out[q] = o[q]
    out_r[sp, 1] = a / t
    m32 = -1e30
    for k in range(min(32, ne)):
        v = s[es[k]]
        if v > m32:
            m32 = v
    out_r[sp, 2] = m32
    num = 0.0; den = 0.0
    for k in range(ne):
        num += w[k] * s[es[k]]; den += w[k]
    out_r[sp, 3] = num / den
    l5 = 0.0; n5 = min(5, ne)
    for k in range(n5):
        l5 += s[es[k]]
    out_r[sp, 4] = l5 / n5
    return t


@nb.njit(parallel=True, cache=True, fastmath=True)
def feed_sims(coff, M, ccat, cloc, dd_off, dd_n, dd_loc, ev_off, ev_n, ev_slot, U, UW, rC, rH, mu, WkT, muW, out, jac):
    D = U.shape[1]
    for f in nb.prange(len(M)):
        o = coff[f]; m = M[f]; nh = dd_n[f]; ne = ev_n[f]
        if nh == 0:
            for i in range(m):
                for sp in range(4):
                    for q in range(5):
                        out[o + i, sp, q] = np.nan
                for q in range(3):
                    jac[o + i, q] = np.nan
            continue
        Hm = np.empty((nh, D), np.float32); HW = np.empty((nh, D), np.float32); rh = np.empty(nh, np.float32)
        for j in range(nh):
            li = dd_loc[dd_off[f] + j]
            Hm[j] = U[li]; HW[j] = UW[li]; rh[j] = rH[li]
        es = ev_slot[ev_off[f]:ev_off[f] + ne]
        w = np.empty(ne, np.float32)
        for k in range(ne):
            w[k] = np.exp(-k / 3.0)
        cats = ccat[o:o + m]
        order = np.argsort(cats, kind="mergesort")
        Hk = np.empty((nh, D), np.float32); Hg = np.empty((nh, D), np.float32)
        tmp = np.empty(D, np.float32); ck = np.empty(D, np.float32); cg = np.empty(D, np.float32)
        s0 = np.empty(nh, np.float32); s1 = np.empty(nh, np.float32); s1g = np.empty(nh, np.float32); s2 = np.empty(nh, np.float32)
        top0 = np.empty(16, np.int64); topx = np.empty(16, np.int64); mark = np.zeros(nh, np.uint8)
        i = 0
        while i < m:
            k = cats[order[i]]
            j2 = i
            while j2 < m and cats[order[j2]] == k:
                j2 += 1
            # history transformed for candidate category k
            for j in range(nh):
                for d in range(D):
                    tmp[d] = Hm[j, d] - mu[k, d]
                nn = 0.0
                for e in range(D):
                    acc = 0.0
                    for d in range(D):
                        acc += tmp[d] * WkT[k, d, e]
                    Hk[j, e] = acc; nn += acc * acc
                nn = np.sqrt(nn) + 1e-9
                for e in range(D):
                    Hk[j, e] /= nn
                ng = 0.0
                for d in range(D):
                    v = HW[j, d] - muW[k, d]; Hg[j, d] = v; ng += v * v
                ng = np.sqrt(ng) + 1e-9
                for d in range(D):
                    Hg[j, d] /= ng
            for t in range(i, j2):
                r = o + order[t]; li = cloc[r]
                if li < 0:
                    for sp in range(4):
                        for q in range(5):
                            out[r, sp, q] = np.nan
                    for q in range(3):
                        jac[r, q] = np.nan
                    continue
                for d in range(D):
                    tmp[d] = U[li, d] - mu[k, d]
                nn = 0.0
                for e in range(D):
                    acc = 0.0
                    for d in range(D):
                        acc += tmp[d] * WkT[k, d, e]
                    ck[e] = acc; nn += acc * acc
                nn = np.sqrt(nn) + 1e-9
                ng = 0.0
                for d in range(D):
                    v = UW[li, d] - muW[k, d]; cg[d] = v; ng += v * v
                ng = np.sqrt(ng) + 1e-9
                for j in range(nh):
                    a0 = 0.0; a1 = 0.0; a2 = 0.0
                    for d in range(D):
                        a0 += U[li, d] * Hm[j, d]; a1 += ck[d] * Hk[j, d]; a2 += cg[d] * Hg[j, d]
                    s0[j] = a0; s1[j] = a1 / nn; s1g[j] = a2 / ng
                    s2[j] = 2.0 * a0 - rC[li] - rh[j]
                tt = _signals(s0, es, ne, w, out[r], 0, top0)
                for q in range(tt):
                    mark[top0[q]] = 1
                for sp in range(1, 4):
                    sx = s1 if sp == 1 else (s1g if sp == 2 else s2)
                    _signals(sx, es, ne, w, out[r], sp, topx)
                    inter = 0
                    for q in range(tt):
                        inter += mark[topx[q]]
                    jac[r, sp - 1] = inter / (2.0 * tt - inter)
                for q in range(tt):
                    mark[top0[q]] = 0
            i = j2


def csls_r(Xq, Pool, k=10, bs=8192):
    import torch
    torch.set_num_threads(12)
    Pt = torch.from_numpy(Pool)
    out = np.empty(len(Xq), np.float32)
    for s in range(0, len(Xq), bs):
        q = torch.from_numpy(Xq[s:s + bs])
        out[s:s + bs] = torch.topk(q @ Pt.T, k, dim=1).values.mean(1).numpy()
    return out


def stage_sims(name):
    fit = np.load(OUT / "fit.npz")
    d = split_data(name)
    n = len(d["M"])
    log(f"[{name}] feeds {n:,}, candidate rows {len(d['cemb']):,}, history events {len(d['hemb']):,} "
        f"(with content {(d['hemb'] >= 0).mean():.3f})")
    allidx = np.concatenate([d["cemb"][d["cemb"] >= 0], d["hemb"][d["hemb"] >= 0]])
    uidx, _ = np.unique(allidx, return_inverse=True)
    cloc = np.full(len(d["cemb"]), -1, np.int64); okc = d["cemb"] >= 0; cloc[okc] = np.searchsorted(uidx, d["cemb"][okc])
    hloc = np.full(len(d["hemb"]), -1, np.int64); okh = d["hemb"] >= 0; hloc[okh] = np.searchsorted(uidx, d["hemb"][okh])
    U = unit_rows(memmap_table(d["tab"]), uidx)
    log(f"[{name}] distinct items {len(uidx):,} loaded from the {d['tab']} table")
    isC = np.zeros(len(uidx), bool); isC[cloc[okc]] = True
    isH = np.zeros(len(uidx), bool); isH[hloc[okh]] = True
    rC = np.zeros(len(uidx), np.float32); rH = np.zeros(len(uidx), np.float32)
    rC[isC] = csls_r(U[isC], fit["PH"]); rH[isH] = csls_r(U[isH], fit["PC"])
    log(f"[{name}] CSLS hubness: rC mean {rC[isC].mean():.4f} (sd {rC[isC].std():.4f}); rH mean {rH[isH].mean():.4f} (sd {rH[isH].std():.4f})")
    ndd = np.zeros(n, np.int64); nev = np.zeros(n, np.int64)
    dedup_count(d["hoff"], d["H"], d["hemb"], d["hitem"], ndd, nev)
    dd_off = np.concatenate([[0], np.cumsum(ndd)[:-1]]).astype(np.int64); ev_off = np.concatenate([[0], np.cumsum(nev)[:-1]]).astype(np.int64)
    dd_loc = np.zeros(int(ndd.sum()), np.int64); ev_slot = np.zeros(int(nev.sum()), np.int64)
    dedup_fill(d["hoff"], d["H"], d["hemb"], d["hitem"], hloc, dd_off, ev_off, dd_loc, ev_slot)
    log(f"[{name}] distinct valid history items per feed: median {int(np.median(ndd))}, share of feeds with 0 {(ndd == 0).mean():.4f}, "
        f"> 16 {(ndd > 16).mean():.3f}, > 32 {(ndd > 32).mean():.3f}")
    WkT = np.ascontiguousarray(np.transpose(fit["Wk"], (0, 2, 1))).astype(np.float32)
    mu = fit["mu"].astype(np.float32); Wp = fit["Wp"].astype(np.float32)
    UW = (U @ Wp.T).astype(np.float32); muW = (mu @ Wp.T).astype(np.float32)
    out = np.zeros((len(d["cemb"]), 4, 5), np.float32); jac = np.zeros((len(d["cemb"]), 3), np.float32)
    t = time.time()
    if name == "val":  # compile on a few feeds first, then time
        pass
    feed_sims(d["coff"], d["M"], d["ccat"], cloc, dd_off, ndd, dd_loc, ev_off, nev, ev_slot, U, UW, rC, rH, mu, WkT, muW, out, jac)
    log(f"[{name}] similarity pass {time.time() - t:.1f}s")
    np.savez(OUT / f"sims_{name}.npz", out=out, jac=jac, ndd=ndd)
    log(f"[{name}] saved sims")


# ============================================================ gate
from metrics import ndcg_per_feed  # noqa: E402


@nb.njit(cache=True)
def feed_z_nan(off, M, x):
    out = np.zeros(len(x))
    for g in range(len(M)):
        o, m = off[g], M[g]
        s = 0.0; s2 = 0.0; c = 0
        for k in range(m):
            v = x[o + k]
            if v == v:
                s += v; s2 += v * v; c += 1
        if c >= 2:
            mu = s / c; sd = np.sqrt(max(s2 / c - mu * mu, 0.0))
            if sd > 0:
                for k in range(m):
                    v = x[o + k]
                    if v == v:
                        out[o + k] = (v - mu) / sd
    return out


def se(d):
    d = np.asarray(d, np.float64); return d.std(ddof=1) / np.sqrt(len(d))


def cell_pairs(zG, y, cat, off, M, feeds):
    n = len(M); fid = np.repeat(np.arange(n), M)
    ck = fid * 64 + cat; uk, inv = np.unique(ck, return_inverse=True)
    lse = np.log(np.bincount(inv, np.exp(zG)))
    kf = uk // 64; o = np.lexsort((-lse, kf)); cr = np.empty(len(uk), np.int64); cr[o] = np.arange(len(uk)) - np.searchsorted(kf[o], kf[o])
    pcr = cr[inv]
    o2 = np.lexsort((y, -zG, ck)); wr = np.empty(len(y), np.int64); wr[o2] = np.arange(len(y)) - np.searchsorted(ck[o2], ck[o2])
    cellpos = (y > 0) & (pcr == 0) & (wr >= 3)
    pi, ni = [], []
    for g in feeds:
        s, m = off[g], M[g]
        pp = np.nonzero(cellpos[s:s + m])[0] + s
        if len(pp) == 0:
            continue
        neg = np.nonzero(y[s:s + m] == 0)[0] + s
        for p_ in pp:
            q = neg[(cat[neg] == cat[p_]) & (zG[neg] > zG[p_])]
            pi.append(np.full(len(q), p_)); ni.append(q)
    pi = np.concatenate(pi); ni = np.concatenate(ni)
    return pi, ni, cellpos


def pair_auc(x, pi, ni):
    a, b = x[pi], x[ni]; ok = np.isfinite(a) & np.isfinite(b)
    dd = a[ok] - b[ok]
    return float(((dd > 0) + 0.5 * (dd == 0)).mean()), float(ok.mean())


def pair_auc_se(x, pi, ni):
    """AUC with a positive-clustered SE (pairs of one positive are not independent)."""
    a, b = x[pi], x[ni]; ok = np.isfinite(a) & np.isfinite(b)
    v = ((a > b) + 0.5 * (a == b)).astype(np.float64)[ok]; g = pi[ok]
    ug, inv = np.unique(g, return_inverse=True)
    sm = np.bincount(inv, v); cn = np.bincount(inv).astype(np.float64)
    auc = sm.sum() / cn.sum()
    resid = sm - auc * cn
    sev = np.sqrt((resid ** 2).sum()) / cn.sum() * np.sqrt(len(ug) / max(len(ug) - 1, 1))
    return float(auc), float(sev), float(ok.mean())


WG = np.array([-0.5, -0.3, -0.2, -0.15, -0.1, -0.05, -0.02, 0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5])


def nested(zb, zx, y, off, M, groups, sel_mask, seeds=(0, 1)):
    from sklearn.model_selection import GroupKFold
    n = len(M); fid = np.repeat(np.arange(n), M)
    base = ndcg_per_feed(zb, y, off, M)
    pers = np.stack([ndcg_per_feed(zb + w * zx, y, off, M) for w in WG])
    # within-feed centred label for the ridge variant
    yc = y - (np.bincount(fid, y, n) / M)[fid]
    outg, outr, wsg, wsr = [], [], [], []
    for seed in seeds:
        dg = np.zeros(n); dr = np.zeros(n)
        for trn, va in GroupKFold(5, shuffle=True, random_state=seed).split(np.arange(n), groups=groups):
            trk = trn[sel_mask[trn]]
            i = int(np.argmax(pers[:, trk].mean(1))); wsg.append(float(WG[i]))
            dg[va] = pers[i, va] - base[va]
            rm = np.zeros(n, bool); rm[trk] = True; rr = rm[fid]
            A = np.stack([zb[rr], zx[rr]], 1); bvec = yc[rr]
            coef = np.linalg.solve(A.T @ A + 1.0 * np.eye(2), A.T @ bvec)
            wr = coef[1] / coef[0] if coef[0] > 0 else 0.0
            wsr.append(float(wr))
            vm = np.zeros(n, bool); vm[va] = True
            vr = vm[fid]
            s = zb.copy(); s[vr] = zb[vr] + wr * zx[vr]
            dr[va] = ndcg_per_feed(s, y, off, M)[va] - base[va]
        outg.append(dg); outr.append(dr)
    full = {float(w): float((pers[k] - base)[sel_mask].mean()) for k, w in enumerate(WG)}
    return np.mean(outg, 0), np.mean(outr, 0), wsg, wsr, full, base


def stage_gate():
    from qt import leak_mask
    from pmix import weights as pmix_weights, wmean_se
    from stack_val import feed_rank_z
    R = {}
    # ------------------------------------------------ val
    f = pl.read_parquet(FLAT / "val/feeds.parquet")
    M = f["M"].to_numpy().astype(np.int64); off = f["cand_off"].to_numpy().astype(np.int64); n = len(M)
    y = np.load(FLAT / "val/cand_target.npy").astype(np.float64); cat = np.load(FLAT / "val/cand_category_id.npy").astype(np.int64)
    users = f["user_id"].to_numpy().copy(); users[users == 0] = -np.arange(1, (users == 0).sum() + 1)
    keep = ~leak_mask()
    def z(x): return feed_rank_z(off, M, np.asarray(x, np.float32))[:, 1].astype(np.float64)
    zG3 = 0.5 * z(np.load(P / "ens_G3_ft_val.npy")) + 0.5 * z(np.load(P / "ens_G3_to_val.npy"))
    base_nd = ndcg_per_feed(zG3, y, off, M)
    log(f"[val] feeds {n}, leak-free {keep.sum()}; G3 NDCG all {base_nd.mean():.5f} / leak-free {base_nd[keep].mean():.5f}")
    S = np.load(OUT / "sims_val.npz"); out, jac, ndd = S["out"], S["jac"], S["ndd"]
    fid = np.repeat(np.arange(n), M)
    # (1) Jaccard
    R["jaccard"] = {}
    big = (ndd > 32)[fid]
    for q, sp in enumerate(SPACES[1:]):
        j = jac[:, q]; ok = np.isfinite(j)
        R["jaccard"][sp] = dict(mean_all=float(j[ok].mean()), median_all=float(np.median(j[ok])), mean_gt32=float(j[ok & big].mean()),
                                median_gt32=float(np.median(j[ok & big])), share_eq1=float((j[ok] == 1).mean()))
        log(f"[val] (1) top-16 Jaccard raw vs {sp}: mean {j[ok].mean():.3f} (median {np.median(j[ok]):.3f}); feeds with > 32 distinct "
            f"history items: mean {j[ok & big].mean():.3f} (median {np.median(j[ok & big]):.3f}); identical sets {(j[ok] == 1).mean():.3f} "
            f"-> plan stop rule (> 0.8): {'STOP arm' if j[ok].mean() > 0.8 else 'continue'}")
    # (2) cell AUCs
    pi, ni, cellpos = cell_pairs(zG3, y, cat, off, M, np.nonzero(keep)[0])
    log(f"[val] (2) cell: {cellpos[keep[fid]].sum():,} leak-free cell positives, {len(pi):,} beater pairs, {len(np.unique(fid[pi])):,} feeds")
    ctrl = {"-log1p(it_hist_all_luo)": -np.log1p(np.load(FEATS / "val/it_hist_all_luo.npy").astype(np.float64)),
            "-maxsim32": -np.load(FEATS / "val/maxsim32.npy").astype(np.float64),
            "+maxsim32 (feature)": np.load(FEATS / "val/maxsim32.npy").astype(np.float64)}
    R["cell_val"] = {}
    for k, x in ctrl.items():
        a, s_, cov = pair_auc_se(x, pi, ni); R["cell_val"][k] = (a, s_, cov)
        log(f"[val]   control {k:26s} cell AUC {a:.4f} ± {s_:.4f} (cov {cov:.3f})")
    for sp_i, sp in enumerate(SPACES):
        for q, sg in enumerate(SIGS):
            a, s_, cov = pair_auc_se(out[:, sp_i, q].astype(np.float64), pi, ni); R["cell_val"][f"{sp}:{sg}"] = (a, s_, cov)
        log(f"[val]   {sp:7s} cell AUC " + "  ".join(f"{sg} {R['cell_val'][f'{sp}:{sg}'][0]:.4f}±{R['cell_val'][f'{sp}:{sg}'][1]:.4f}" for sg in SIGS))
    # paired difference S1 - S0 on maxall (same pairs; clustered by positive)
    def paired_auc_diff(xa, xb):
        a1, b1 = xa[pi], xa[ni]; a0, b0 = xb[pi], xb[ni]
        ok = np.isfinite(a1) & np.isfinite(b1) & np.isfinite(a0) & np.isfinite(b0)
        v = (((a1 > b1) + 0.5 * (a1 == b1)) - ((a0 > b0) + 0.5 * (a0 == b0))).astype(np.float64)[ok]
        ug, inv = np.unique(pi[ok], return_inverse=True); sm = np.bincount(inv, v); cn = np.bincount(inv).astype(np.float64)
        dlt = sm.sum() / cn.sum(); sev = np.sqrt(((sm - dlt * cn) ** 2).sum()) / cn.sum()
        return float(dlt), float(sev)
    R["cell_val_diff"] = {}
    for sp_i, sp in enumerate(SPACES[1:], 1):
        dl, ds = paired_auc_diff(out[:, sp_i, 0].astype(np.float64), out[:, 0, 0].astype(np.float64))
        R["cell_val_diff"][sp] = (dl, ds)
        log(f"[val]   cell AUC {sp} - S0raw (maxall, paired): {dl:+.4f} ± {ds:.4f}")
    # single-feature NDCG (leak-free), paired vs raw
    R["single_val"] = {}
    def nd_single(x):
        x = np.where(np.isfinite(x), x, np.nanmin(x) - 1.0)
        return ndcg_per_feed(x, y, off, M)
    rawnd = {sg: nd_single(out[:, 0, q].astype(np.float64)) for q, sg in enumerate(SIGS)}
    for sp_i, sp in enumerate(SPACES):
        row = []
        for q, sg in enumerate(SIGS):
            p_ = nd_single(out[:, sp_i, q].astype(np.float64)) if sp_i else rawnd[sg]
            dd = (p_ - rawnd[sg])[keep]
            R["single_val"][f"{sp}:{sg}"] = (float(p_[keep].mean()), float(dd.mean()), float(se(dd)) if sp_i else 0.0)
            row.append(f"{sg} {p_[keep].mean():.4f}" + (f" ({dd.mean():+.4f}±{se(dd):.4f})" if sp_i else ""))
        log(f"[val]   single-feature NDCG {sp:7s} " + "  ".join(row))
    # (3) nested residual on G3
    W = pmix_weights(); wpriv = W["private"]
    R["nested_val"] = {}
    TESTS = [("S1cat", "maxall"), ("S0raw", "maxall"), ("S1pool", "maxall"), ("S2csls", "maxall"), ("S1cat", "top16"), ("S1cat", "rec_hl3")]
    for sp, sg in TESTS:
        x = out[:, SPACES.index(sp), SIGS.index(sg)].astype(np.float64)
        zx = feed_z_nan(off, M, x)
        dg, dr, wsg, wsr, full, _ = nested(zG3, zx, y, off, M, users, keep)
        pm, pse = wmean_se(dg[keep], wpriv[keep])
        R["nested_val"][f"{sp}:{sg}"] = dict(grid=(float(dg[keep].mean()), float(se(dg[keep]))), ridge=(float(dr[keep].mean()), float(se(dr[keep]))),
                                            pmix=(float(pm), float(pse)), w_grid=wsg, w_ridge=wsr, full_fit=full,
                                            M50=(float(dg[keep & (M >= 50)].mean()), float(se(dg[keep & (M >= 50)]))),
                                            Mlt50=(float(dg[keep & (M < 50)].mean()), float(se(dg[keep & (M < 50)]))))
        log(f"[val] (3) nested residual on G3, z({sp}:{sg}): grid {dg[keep].mean():+.5f} ± {se(dg[keep]):.5f} (z {dg[keep].mean() / se(dg[keep]):+.2f}; "
            f"w {wsg}) | ridge {dr[keep].mean():+.5f} ± {se(dr[keep]):.5f} (w {np.round(wsr, 4).tolist()}) | pmix {pm:+.5f} ± {pse:.5f} | "
            f"M>=50 {R['nested_val'][f'{sp}:{sg}']['M50'][0]:+.5f} M<50 {R['nested_val'][f'{sp}:{sg}']['Mlt50'][0]:+.5f} | full-fit best "
            f"{max(full.values()):+.5f} at w={max(full, key=full.get)}")
    # ------------------------------------------------ holdout
    hs = OUT / "sims_holdout.npz"
    if hs.exists():
        dH = split_data("holdout")
        Mh, offh = dH["M"], dH["coff"]; nh = len(Mh); fidh = np.repeat(np.arange(nh), Mh); yh = dH["y"]
        def zh(x): return feed_rank_z(offh, Mh, np.asarray(x, np.float32))[:, 1].astype(np.float64)
        zGh = np.mean([zh(np.load(P / "holdout" / f"nn_{m}_holdout.npy")) for m in G3_MEMBERS], 0)
        bh = ndcg_per_feed(zGh, yh, offh, Mh)
        log(f"[holdout] feeds {nh}, rows {len(yh):,}; G3 holdout pool NDCG {bh.mean():.5f}")
        Sh = np.load(hs); outh, jach, nddh = Sh["out"], Sh["jac"], Sh["ndd"]
        R["jaccard_holdout"] = {}
        for q, sp in enumerate(SPACES[1:]):
            j = jach[:, q]; ok = np.isfinite(j)
            R["jaccard_holdout"][sp] = (float(j[ok].mean()), float(np.median(j[ok])))
            log(f"[holdout] (1) top-16 Jaccard raw vs {sp}: mean {j[ok].mean():.3f} (median {np.median(j[ok]):.3f})")
        ug = dH["users"].copy(); ug[ug == 0] = -np.arange(1, (ug == 0).sum() + 1)
        allh = np.ones(nh, bool)
        pih, nih, cph = cell_pairs(zGh, yh, dH["ccat"], offh, Mh, np.arange(nh))
        log(f"[holdout] (2) cell: {cph.sum():,} cell positives, {len(pih):,} beater pairs")
        crows = dH["crows"]
        ctrlh = {"-log1p(it_hist_all_luo)": -np.log1p(np.asarray(np.load(FEATS / "train/it_hist_all_luo.npy", mmap_mode="r")[crows]).astype(np.float64)),
                 "-maxsim32": -np.asarray(np.load(FEATS / "train/maxsim32.npy", mmap_mode="r")[crows]).astype(np.float64)}
        R["cell_holdout"] = {}
        for k, x in ctrlh.items():
            a, s_, cov = pair_auc_se(x, pih, nih); R["cell_holdout"][k] = (a, s_, cov)
            log(f"[holdout]   control {k:26s} cell AUC {a:.4f} ± {s_:.4f} (cov {cov:.3f})")
        for sp_i, sp in enumerate(SPACES):
            for q, sg in enumerate(SIGS):
                a, s_, cov = pair_auc_se(outh[:, sp_i, q].astype(np.float64), pih, nih); R["cell_holdout"][f"{sp}:{sg}"] = (a, s_, cov)
            log(f"[holdout]   {sp:7s} cell AUC " + "  ".join(f"{sg} {R['cell_holdout'][f'{sp}:{sg}'][0]:.4f}" for sg in SIGS))
        R["nested_holdout"] = {}
        for sp, sg in TESTS:
            x = outh[:, SPACES.index(sp), SIGS.index(sg)].astype(np.float64)
            zx = feed_z_nan(offh, Mh, x)
            dg, dr, wsg, wsr, full, _ = nested(zGh, zx, yh, offh, Mh, ug, allh)
            R["nested_holdout"][f"{sp}:{sg}"] = dict(grid=(float(dg.mean()), float(se(dg))), ridge=(float(dr.mean()), float(se(dr))),
                                                     w_grid=wsg, w_ridge=wsr, full_fit=full)
            log(f"[holdout] (3) nested residual on G3, z({sp}:{sg}): grid {dg.mean():+.5f} ± {se(dg):.5f} (z {dg.mean() / se(dg):+.2f}; w {wsg}) | "
                f"ridge {dr.mean():+.5f} ± {se(dr):.5f} (w {np.round(wsr, 4).tolist()}) | full-fit best {max(full.values()):+.5f} at w={max(full, key=full.get)}")
    # ------------------------------------------------ gate
    a1 = R["cell_val"]["S1cat:maxall"][0]; a0 = R["cell_val"]["S0raw:maxall"][0]
    actrl = max(R["cell_val"]["-log1p(it_hist_all_luo)"][0], R["cell_val"]["-maxsim32"][0])
    nv = R["nested_val"]["S1cat:maxall"]
    nh_ = R.get("nested_holdout", {}).get("S1cat:maxall")
    c1 = a1 - a0 >= 0.05
    c2 = a1 > actrl
    c3 = nv["grid"][0] >= 0.0003 and nv["grid"][0] >= 2 * nv["grid"][1] and nv["ridge"][0] > 0
    c4 = nh_ is not None and nh_["grid"][0] >= 0
    c5 = nv["pmix"][0] >= -nv["pmix"][1]
    R["gate"] = dict(i_auc_gain=(a1 - a0, bool(c1)), ii_above_controls=(a1, actrl, bool(c2)), iii_nested_val=(nv["grid"], nv["ridge"], bool(c3)),
                     iv_holdout=(nh_["grid"] if nh_ else None, bool(c4)), v_pmix=(nv["pmix"], bool(c5)),
                     verdict="OPEN" if (c1 and c2 and c3 and c4 and c5) else "CLOSED")
    log(f"GATE (i)   S1 cell AUC {a1:.4f} - raw {a0:.4f} = {a1 - a0:+.4f} (>= +0.05): {c1}")
    log(f"GATE (ii)  S1 cell AUC {a1:.4f} > max control {actrl:.4f}: {c2}")
    log(f"GATE (iii) val nested grid {nv['grid'][0]:+.5f} ± {nv['grid'][1]:.5f} (>= +0.0003 and 2 SE), ridge {nv['ridge'][0]:+.5f} (> 0): {c3}")
    log(f"GATE (iv)  holdout nested {nh_['grid'][0] if nh_ else float('nan'):+.5f} (>= 0): {c4}")
    log(f"GATE (v)   pmix {nv['pmix'][0]:+.5f} ± {nv['pmix'][1]:.5f} (>= -1 SE): {c5}")
    log(f"GATE VERDICT: {R['gate']['verdict']}")
    json.dump(R, open(OUT / "gate.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["fit", "sims", "gate"])
    ap.add_argument("--split", default="val", choices=["val", "holdout"])
    a = ap.parse_args()
    LOG = open(OUT / f"{a.stage}{'_' + a.split if a.stage == 'sims' else ''}.log", "w")
    if a.stage == "fit":
        stage_fit()
    elif a.stage == "sims":
        stage_sims(a.split)
    else:
        stage_gate()
    log("done")
