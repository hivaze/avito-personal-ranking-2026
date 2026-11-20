"""RESCUE r05: P7 day-causal implicit ALS in the view D5 (PREDECLARED.txt).
For each window: sequences = the dense sequences of the window (latest train feed of each non-zero user inside the window + every user-0 feed of the
window; holdout feeds included, label-free), matrix = de-duplicated (sequence, history item) pairs with confidence 1 + 5 * (1 + #distinct event
types of the pair), items with >= 2 sequences. Implicit ALS (Hu-Koren-Volinsky), f 64, 12 iterations, reg 0.05, 3 conjugate-gradient steps per
row (the algorithm of eda/collab_stats/04g_als_clean_gpu.py, here per-row on CPU in numba; no train positives, no val / test history: label-free and causal).
als5(row) = cosine(item factor of the candidate, sum over the row's own history of exp(-hpos / 5) * unit item factor)  (src/features_als.py, column 1).
NaN if the candidate has no factor or no history item has one.
Leak rule: every sequence of the matrix is a snapshot of a train feed inside the window, i.e. dated before the cut of the row; the row's own later
snapshots cannot enter; the row's own history is read only for the profile, as every sequence input of the ranker does. No label array is opened.
Order of the windows: the three of the first pass (cut 5 = key 31; natural = key 3968; twin cuts 9-11 = key 496), then the other four.
CPU only. Output: out/arr/<set>__als5p1.npy after the first pass, out/arr/<set>__als5.npy after all windows; log out/r05_als.log."""
from rs_common import *
P = Log(OUT / 'r05_als.log')
V = Views(P)
a = V.A['train']
FACT, ITERS, ALPHA, REG, CG = 64, 12, 5.0, 0.05, 3
assert (np.diff(a['ts']) >= 0).all()          # train feeds are in time order: the latest feed of a user is the one with the largest fidx
HI = {s: np.load(FLAT / s / 'hist_item_id.npy') for s in SPLITS}; HP = {s: np.load(FLAT / s / 'hist_hpos.npy') for s in SPLITS}
CI = {s: np.load(FLAT / s / 'cand_item_id.npy') for s in SPLITS}
ht = np.load(FLAT / 'train/hist_event_type.npy')
NITEM = int(max(max(int(x.max()) for x in HI.values()), max(int(x.max()) for x in CI.values()))) + 1
P(f'NITEM {NITEM}; event types {np.unique(ht[:5_000_000]).tolist()} (first 5M history rows)')


@nb.njit(parallel=True, cache=True)
def count_unique(sf, ho, H, hi, cntu):
    for q in nb.prange(len(sf)):
        f = sf[q]; h = H[f]
        if h == 0:
            cntu[q] = 0
            continue
        x = np.sort(hi[ho[f]:ho[f] + h]); c = 1
        for t in range(1, h):
            if x[t] != x[t - 1]:
                c += 1
        cntu[q] = c


@nb.njit(parallel=True, cache=True)
def fill_unique(sf, ho, H, hi, ht, ptr, o_item, o_w):
    for q in nb.prange(len(sf)):
        f = sf[q]; h = H[f]
        if h == 0:
            continue
        seg = hi[ho[f]:ho[f] + h]; order = np.argsort(seg); p = ptr[q]; last = -1; bits = 0; k = -1
        for t in range(h):
            it = seg[order[t]]; b = 1 << ht[ho[f] + order[t]]
            if it != last:
                if k >= 0:
                    c = 0
                    for e in range(16):
                        c += (bits >> e) & 1
                    o_w[p + k] = c
                k += 1; o_item[p + k] = it; last = it; bits = b
            else:
                bits |= b
        c = 0
        for e in range(16):
            c += (bits >> e) & 1
        o_w[p + k] = c


@nb.njit(parallel=True, fastmath=True, cache=True)
def als_half(indptr, indices, cm1, Y, G, X, cg_steps):
    n = len(indptr) - 1; f = Y.shape[1]
    for u in nb.prange(n):
        s = indptr[u]; m = indptr[u + 1] - s
        if m == 0:
            for q in range(f):
                X[u, q] = 0.0
            continue
        Yl = np.empty((m, f), np.float32)
        for t in range(m):
            j = indices[s + t]
            for q in range(f):
                Yl[t, q] = Y[j, q]
        x = np.empty(f, np.float32); r = np.zeros(f, np.float32); p = np.empty(f, np.float32); Ap = np.empty(f, np.float32)
        for q in range(f):
            x[q] = X[u, q]
        for t in range(m):
            c1 = cm1[s + t]; d = np.float32(0.0)
            for q in range(f):
                d += Yl[t, q] * x[q]
            coef = (np.float32(1.0) + c1) - c1 * d
            for q in range(f):
                r[q] += coef * Yl[t, q]
        rs = np.float32(0.0)
        for q in range(f):
            g = np.float32(0.0)
            for k in range(f):
                g += G[q, k] * x[k]
            r[q] -= g; p[q] = r[q]; rs += r[q] * r[q]
        for _ in range(cg_steps):
            if rs < 1e-20:
                break
            for q in range(f):
                g = np.float32(0.0)
                for k in range(f):
                    g += G[q, k] * p[k]
                Ap[q] = g
            for t in range(m):
                d = np.float32(0.0)
                for q in range(f):
                    d += Yl[t, q] * p[q]
                d *= cm1[s + t]
                for q in range(f):
                    Ap[q] += d * Yl[t, q]
            pAp = np.float32(0.0)
            for q in range(f):
                pAp += p[q] * Ap[q]
            al = rs / pAp; rsn = np.float32(0.0)
            for q in range(f):
                x[q] += al * p[q]; r[q] -= al * Ap[q]; rsn += r[q] * r[q]
            be = rsn / rs
            for q in range(f):
                p[q] = r[q] + be * p[q]
            rs = rsn
        for q in range(f):
            X[u, q] = x[q]


@nb.njit(parallel=True, fastmath=True, cache=True)
def score(fsel, loff, co, M, ho, H, citem, hitem, hpos, ilut, Vn, out):
    f_ = Vn.shape[1]
    for q in nb.prange(len(fsel)):
        f = fsel[q]; p5 = np.zeros(f_, np.float32)
        for j in range(H[f]):
            r = ilut[hitem[ho[f] + j]]
            if r < 0:
                continue
            w = np.float32(np.exp(-hpos[ho[f] + j] / 5.0))
            for d in range(f_):
                p5[d] += w * Vn[r, d]
        n5 = np.float32(0.0)
        for d in range(f_):
            n5 += p5[d] * p5[d]
        n5 = np.sqrt(n5)
        for i in range(M[f]):
            r = ilut[citem[co[f] + i]]
            if r < 0 or n5 <= 0:
                out[loff[q] + i] = np.nan
            else:
                s = np.float32(0.0)
                for d in range(f_):
                    s += Vn[r, d] * p5[d]
                out[loff[q] + i] = s / n5


def seq_feeds(wslots):
    f = np.nonzero(np.isin(a['slot'], wslots))[0]; u = a['u'][f]
    z = f[u == 0]; nzf = f[u > 0]; un = u[u > 0]
    o = np.lexsort((nzf, un)); last = np.r_[un[o][1:] != un[o][:-1], True]
    return np.sort(np.concatenate([z, nzf[o][last]])), len(z)


def fit(wslots):
    t1 = time.time()
    sf, nz0 = seq_feeds(wslots); nS = len(sf)
    cntu = np.zeros(nS, np.int64); count_unique(sf, a['ho'], a['H'], HI['train'], cntu)
    ptr = np.concatenate([[0], np.cumsum(cntu)]).astype(np.int64); nnz0 = int(ptr[-1])
    o_item = np.empty(nnz0, np.int32); o_w = np.empty(nnz0, np.int8); fill_unique(sf, a['ho'], a['H'], HI['train'], ht, ptr[:-1], o_item, o_w)
    icnt = np.bincount(o_item, minlength=NITEM); kept = np.nonzero(icnt >= 2)[0]; nI = len(kept)
    ilut = np.full(NITEM, -1, np.int32); ilut[kept] = np.arange(nI, dtype=np.int32)
    ii = ilut[o_item]; mk = ii >= 0; seq_of = np.repeat(np.arange(nS, dtype=np.int32), cntu)[mk]; ii = ii[mk]
    cm1 = (ALPHA * (1.0 + o_w[mk].astype(np.float32))).astype(np.float32); nnz = len(ii)
    del o_item, o_w, mk
    ptr_u = np.searchsorted(seq_of, np.arange(nS + 1)).astype(np.int64)
    o = np.argsort(ii, kind='stable'); ind_i = seq_of[o]; cm1_i = cm1[o]; ptr_i = np.searchsorted(ii[o], np.arange(nI + 1)).astype(np.int64); del o
    P(f'   matrix: sequences {nS} ({nz0} user-0 feeds), items with >= 2 sequences {nI} of {int((icnt > 0).sum())}, pairs {nnz} of {nnz0} ({time.time() - t1:.0f}s)')
    rg = np.random.default_rng(0)
    X = (rg.standard_normal((nS, FACT), dtype=np.float32) * 0.01); Y = (rg.standard_normal((nI, FACT), dtype=np.float32) * 0.01)
    eye = REG * np.eye(FACT, dtype=np.float32)
    for it in range(ITERS):
        t2 = time.time()
        G = (Y.T @ Y + eye).astype(np.float32); als_half(ptr_u, ii, cm1, Y, G, X, CG)
        G = (X.T @ X + eye).astype(np.float32); als_half(ptr_i, ind_i, cm1_i, X, G, Y, CG)
        if it in (0, 1, ITERS - 1):
            P(f'   iter {it + 1}/{ITERS}: |X| {np.linalg.norm(X, axis=1).mean():.3f} |Y| {np.linalg.norm(Y, axis=1).mean():.3f} ({time.time() - t2:.0f}s)')
    Vn = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-9)
    return ilut, np.ascontiguousarray(Vn.astype(np.float32))


R = {q: V.rows(q) for q in V.Q}
res = {q: np.full(len(R[q][0]), np.nan, np.float32) for q in V.Q}
ORDER = [31, 3968, 496] + [k for k in V.wkeys if k not in (31, 3968, 496)]
assert set(ORDER) == set(V.wkeys)
for n_done, k in enumerate(ORDER):
    wsl = DENSE[[j for j in range(12) if (k >> j) & 1]]
    P(f'window key {k} (slots {wsl.tolist()})')
    ilut, Vn = fit(wsl)
    for q, (sp, sel, cut) in V.Q.items():
        fk = np.array([wkey(int(c)) if c >= 0 else 0 for c in cut]); m = fk == k
        if not m.any():
            continue
        A_ = V.A[sp]; off = R[q][2]; out = res[q]
        score(np.ascontiguousarray(sel[m]).astype(np.int64), np.ascontiguousarray(off[m]), A_['co'], A_['M'], A_['ho'], A_['H'], CI[sp], HI[sp], HP[sp], ilut, Vn, out)
        rm = np.repeat(m, R[q][3])
        P(f'   {q}: feeds {int(m.sum())}, rows {int(rm.sum())}, als5 present {np.mean(~np.isnan(out[rm])):.4f}, mean {np.nanmean(out[rm]):.4f}, sd {np.nanstd(out[rm]):.4f}')
    del ilut, Vn
    if n_done == 2:
        for q in V.Q:
            put(q, 'als5p1', res[q])
        P('FIRST PASS WRITTEN (als5p1: windows 31, 3968, 496)')
for q in V.Q:
    put(q, 'als5', res[q])
P('DONE')
