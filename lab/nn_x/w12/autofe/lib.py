"""AutoFE library: datasets (val / 30K holdout / test / train sample), numba kernels, feature generators.
CPU only. New files only; reads project data read-only."""
import os, sys, json, itertools
from pathlib import Path
import numpy as np, polars as pl, numba as nb

ROOT = Path("<repo>")
OUT = ROOT / "nn_x/w12/autofe"
SPREF = ("plant-" + os.environ["AUTOFE_PLANT"].replace(":", "-")) if os.environ.get("AUTOFE_PLANT") else "screen"
CACHE = OUT / "cache"
CACHE.mkdir(parents=True, exist_ok=True)
FLAT = ROOT / "data/flat"
FEATDIRS = [ROOT / "data/feats", ROOT / "nn_x/w5/feat_cache"]
COMP = ROOT / "data/competition"
sys.path.insert(0, str(ROOT / "src"))

XT34 = ['seq_cv_invd20', 'seq_cv_pop', 'seq_cv_norm', 'it_hist_all_luo', 'it_hist_et1_luo', 'it_hist_et2_luo',
        'it_hist_et3_luo', 'it_hist_et4_luo', 'ctx_nd_hcent', 'ctx_nd_knn_f1', 'ctx_nd_knn_h', 'ctx_nd_knn_h_dev',
        'ctx_nd_knn_u', 'ctx_nd_knn_u_dev', 'ctx_nd_max', 'ctx_nd_top5h', 'ctx_nd_top5u', 'ctx_nd_ucent',
        'ctx_hn_top1w', 'ctx_hn_rrw', 'ctx_hn_bestrk', 'ctx_hn_margin', 'ctx_hn_soft05', 'ctx_hn_e1_rrw',
        'ctx_hs_mutual', 'ctx_hs_k128', 'ctx_hs_e1', 'ctx_hs_t10', 'pre2_fe_z', 'pre2_new_z', 'pre2_fe_s',
        'pre2_new_s', 'pre2_hmax32', 'has:pre2_fe']
# stand-ins (precomputed in data/feats) for the in-model dense inputs built in model.dense_features
INMODEL = ['f1', 'f2', 'f3', 'f4', 'f5', 'cos_u', 'seq_max_all', 'maxsim32', 'seq_top3', 'cos_hall', 'cos_het1',
           'cos_het2', 'cos_het3', 'cos_het4', 'cat_cnt_all', 'cat_cnt_et1', 'cat_cnt_et2', 'cat_cnt_et3',
           'cat_share_all', 'cat_min_hpos', 'feed_cat_share', 'item_norm']
BASE = INMODEL + XT34
CORE = ['f1', 'f2', 'f3', 'f4', 'f5', 'cos_u', 'seq_max_all', 'maxsim32', 'seq_top3', 'cos_hall', 'cat_cnt_all',
        'cat_share_all', 'cat_min_hpos', 'feed_cat_share', 'item_norm', 'seq_cv_invd20', 'seq_cv_pop', 'seq_cv_norm',
        'it_hist_all_luo', 'it_hist_et1_luo', 'it_hist_et2_luo', 'ctx_nd_hcent', 'ctx_nd_knn_f1', 'ctx_nd_knn_h',
        'ctx_nd_knn_u', 'ctx_nd_max', 'ctx_hn_rrw', 'ctx_hn_soft05', 'ctx_hn_margin', 'ctx_hs_k128', 'ctx_hs_e1',
        'pre2_fe_s', 'pre2_new_s', 'pre2_hmax32']


def featfile(split, name):
    for d in FEATDIRS:
        p = d / split / f"{name}.npy"
        if p.exists():
            return p
    raise FileNotFoundError(f"{split}/{name}")


def nn_tf(name, x):
    """The transform the NN applies before standardisation (attach_xt for xt; ours for the stand-ins)."""
    if name.startswith("has:"):
        return x
    if name in XT34:
        if name.endswith("_z"):
            x = np.clip(x, -5, 5)
        elif name.startswith(("seq_cv", "it_hist")):
            x = np.log1p(np.maximum(x, 0))
        else:
            x = np.clip(x, -10, 10)
        return np.nan_to_num(x, nan=0.0)
    if name.startswith("cat_cnt"):
        return np.log1p(np.nan_to_num(x, nan=0.0))
    if name.startswith(("cos_", "seq_max", "maxsim", "seq_top")):
        return np.nan_to_num(x, nan=-1.5)
    return np.nan_to_num(x, nan=0.0)


@nb.njit(parallel=True, cache=True)
def _z(off, M, x):
    out = np.empty(len(x), np.float32)
    for f in nb.prange(len(M)):
        o, m = off[f], M[f]
        s = 0.0; ss = 0.0
        for k in range(o, o + m):
            s += x[k]; ss += x[k] * x[k]
        mu = s / m; sd = np.sqrt(max(ss / m - mu * mu, 0.0)) + 1e-6
        for k in range(o, o + m):
            v = (x[k] - mu) / sd
            out[k] = min(max(v, -6.0), 6.0)
    return out


@nb.njit(parallel=True, cache=True)
def _rk(off, M, x):
    """mid-rank percentile within feed"""
    out = np.empty(len(x), np.float32)
    for f in nb.prange(len(M)):
        o, m = off[f], M[f]
        v = x[o:o + m]
        idx = np.argsort(v)
        i = 0
        while i < m:
            j = i
            while j + 1 < m and v[idx[j + 1]] == v[idx[i]]:
                j += 1
            r = 0.5 * (i + j) / max(m - 1, 1)
            for t in range(i, j + 1):
                out[o + idx[t]] = r
            i = j + 1
    return out


@nb.njit(parallel=True, cache=True)
def _feedstat(off, M, x):
    """per-candidate: dev from feed max, dev from median, gap to next higher value, is-top flag share of ties,
    feed mean, feed sd (broadcast)"""
    n = len(x)
    dmax = np.empty(n, np.float32); dmed = np.empty(n, np.float32); gap = np.empty(n, np.float32)
    mean = np.empty(n, np.float32); sd = np.empty(n, np.float32)
    for f in nb.prange(len(M)):
        o, m = off[f], M[f]
        v = x[o:o + m]
        s = np.sort(v)
        mx = s[m - 1]; med = 0.5 * (s[(m - 1) // 2] + s[m // 2]); mu = v.mean(); sg = v.std()
        for k in range(m):
            dmax[o + k] = v[k] - mx
            dmed[o + k] = v[k] - med
            mean[o + k] = mu; sd[o + k] = sg
            j = np.searchsorted(s, v[k], side='right')
            gap[o + k] = (s[j] - v[k]) if j < m else 0.0
    return dmax, dmed, gap, mean, sd


@nb.njit(parallel=True, cache=True)
def _catgroup(off, M, cat, x):
    """within (feed, category): x - group mean; mid-rank within group; group mean - feed mean; x - group max"""
    n = len(x)
    dgm = np.empty(n, np.float32); grk = np.empty(n, np.float32); gmf = np.empty(n, np.float32)
    dgx = np.empty(n, np.float32)
    for f in nb.prange(len(M)):
        o, m = off[f], M[f]
        s = np.zeros(128); c = np.zeros(128); mx = np.full(128, -1e30)
        tot = 0.0
        for k in range(o, o + m):
            g = cat[k] + 0
            s[g] += x[k]; c[g] += 1.0; tot += x[k]
            if x[k] > mx[g]:
                mx[g] = x[k]
        tot /= m
        for k in range(o, o + m):
            g = cat[k] + 0
            gm = s[g] / c[g]
            dgm[k] = x[k] - gm; gmf[k] = gm - tot; dgx[k] = x[k] - mx[g]
            less = 0.0; eq = 0.0
            for j in range(o, o + m):
                if cat[j] == cat[k]:
                    if x[j] < x[k]:
                        less += 1.0
                    elif x[j] == x[k]:
                        eq += 1.0
            grk[k] = (less + 0.5 * (eq - 1.0)) / max(c[g] - 1.0, 1.0) if c[g] > 1 else 0.5
    return dgm, grk, gmf, dgx


@nb.njit(parallel=True, cache=True)
def ndcg_pf(off, M, s, y, pos_any):
    """per-feed NDCG, binary gain, pessimistic ties; feeds without positives -> 1"""
    out = np.ones(len(M))
    for f in nb.prange(len(M)):
        if not pos_any[f]:
            continue
        o, m = off[f], M[f]
        npos = 0; dcg = 0.0
        for i in range(o, o + m):
            if y[i] > 0:
                si = s[i]; r = 0
                for j in range(o, o + m):
                    if s[j] > si:
                        r += 1
                    elif s[j] == si:
                        if y[j] == 0:
                            r += 1
                        elif j < i:
                            r += 1
                dcg += 1.0 / np.log2(r + 2.0)
                npos += 1
        idcg = 0.0
        for r in range(npos):
            idcg += 1.0 / np.log2(r + 2.0)
        out[f] = dcg / idcg
    return out


@nb.njit(parallel=True, cache=True)
def softmax_gh(off, M, s, y):
    """listwise softmax CE with multiple positives: grad = y - npos*p (ascent direction), hess = npos*p*(1-p);
    also returns per-feed loss"""
    g = np.zeros(len(s), np.float32); h = np.zeros(len(s), np.float32); loss = np.zeros(len(M))
    for f in nb.prange(len(M)):
        o, m = off[f], M[f]
        mx = -1e30; npos = 0.0
        for k in range(o, o + m):
            if s[k] > mx:
                mx = s[k]
            npos += y[k]
        if npos == 0:
            continue
        z = 0.0
        for k in range(o, o + m):
            z += np.exp(s[k] - mx)
        lz = np.log(z) + mx
        for k in range(o, o + m):
            p = np.exp(s[k] - lz)
            g[k] = y[k] - npos * p
            h[k] = npos * p * (1.0 - p)
            if y[k] > 0:
                loss[f] -= (s[k] - lz)
    return g, h, loss


def user_half(uid, n):
    u = uid.astype(np.uint64)
    hsh = (u * np.uint64(2654435761) + np.uint64(12345)) >> np.uint64(9)
    fi = np.arange(n, dtype=np.uint64)
    hsh = np.where(uid == 0, (fi * np.uint64(40503) + np.uint64(7)) >> np.uint64(3), hsh)
    half = (hsh & np.uint64(1)).astype(np.int8)
    inner = ((hsh >> np.uint64(1)) & np.uint64(1)).astype(np.int8)
    return half, inner


class DS:
    def __init__(self, name, n_sample=20000):
        self.name = name
        from stack_val import feed_rank_z
        if name in ("val", "test"):
            self.split = name
            f = pl.read_parquet(FLAT / name / "feeds.parquet")
            self.fsel = np.arange(len(f))
            self.rows = None
        else:
            self.split = "train"
            self.rows = None
            f = pl.read_parquet(FLAT / "train" / "feeds.parquet")
            Mt = f["M"].to_numpy()
            pr = np.nan_to_num(np.load(ROOT / "eda/labels_position/out/train_prandom.npz")["p_random"].astype(np.float32), nan=0.0)
            pr[Mt < 50] = 0.0
            cand = np.where((pr > 0.9) & (Mt >= 50))[0]
            hold = np.sort(np.random.default_rng(12345).choice(cand, 30000, replace=False))
            if name == "hold":
                self.fsel = hold
            else:  # 'trs': random train feeds (not holdout) for the inventory
                rest = np.setdiff1d(np.arange(len(f)), hold)
                self.fsel = np.sort(np.random.default_rng(7).choice(rest, n_sample, replace=False))
            f = f[self.fsel]
        self.M = f["M"].to_numpy().astype(np.int64)
        self.H = np.minimum(f["H"].to_numpy().astype(np.int64), 256)
        self.uid = f["user_id"].to_numpy().astype(np.int64)
        self.ts = f["timestamp"].to_numpy()
        self.coff = f["cand_off"].to_numpy().astype(np.int64)  # offsets in the split's flat arrays
        self.hoff = f["hist_off"].to_numpy().astype(np.int64)
        self.off = np.concatenate([[0], np.cumsum(self.M)[:-1]]).astype(np.int64)
        self.n = int(self.M.sum())
        self.nf = len(self.M)
        if self.rows is None and self.split == "train":
            self.rows = np.concatenate([np.arange(o, o + m) for o, m in zip(self.coff, self.M)])
        self.fid = np.repeat(np.arange(self.nf), self.M)
        self._c = {}
        self.y = None; self.zb = None
        if name == "val":
            self.y = np.load(FLAT / "val/cand_target.npy").astype(np.float32)
            ft = np.load(ROOT / "preds/ens_W8kd_ft_val.npy"); to = np.load(ROOT / "preds/ens_W8kd_to_val.npy")
            self.zb = (0.5 * feed_rank_z(self.off, self.M, ft)[:, 1] + 0.5 * feed_rank_z(self.off, self.M, to)[:, 1]).astype(np.float32)
        elif name == "hold":
            hd = ROOT / "nn_x/w5/preds/holdout"
            meta = np.load(hd / "holdout_meta.npy")
            assert (meta[0] == self.M).all() and (meta[1] == self.off).all()
            self.y = np.load(hd / "holdout_y.npy").astype(np.float32)
            z = 0
            for s in range(1, 5):
                z = z + feed_rank_z(self.off, self.M, np.load(hd / f"nn_w5_e10_kd05w192_s95{s}_holdout.npy"))[:, 1]
            self.zb = (z / 4).astype(np.float32)
        if self.y is not None:
            self.pos_any = np.bincount(self.fid, weights=self.y, minlength=self.nf) > 0
        self.half, self.inner = user_half(self.uid, self.nf)
        pl_ = os.environ.get("AUTOFE_PLANT", "")  # power check: damage the base with a known construction
        if pl_ and self.zb is not None:
            kind, fa, fb, c = pl_.split(":")
            za = _z(self.off, self.M, self.get(fa)); zb_ = _z(self.off, self.M, self.get(fb))
            self.zb = (self.zb - float(c) * (za * zb_ if kind == "mul" else za)).astype(np.float32)

    def flat(self, name):
        a = np.load(FLAT / self.split / f"{name}.npy", mmap_mode="r")
        return np.asarray(a) if self.rows is None else a[self.rows]

    def raw(self, name):
        """raw stored value (no transform)"""
        if name in ("f1", "f2", "f3", "f4", "f5"):
            return np.asarray(self.flat(f"cand_{name}"), dtype=np.float32)
        if name.startswith("has:"):
            return (~np.isnan(self.raw(name[4:]))).astype(np.float32)
        a = np.load(featfile(self.split, name), mmap_mode="r")
        return np.asarray(a if self.rows is None else a[self.rows], dtype=np.float32)

    def get(self, name):
        if name not in self._c:
            self._c[name] = np.ascontiguousarray(nn_tf(name, self.raw(name)).astype(np.float32))
        return self._c[name]

    @property
    def cat(self):
        if "_cat" not in self._c:
            self._c["_cat"] = np.ascontiguousarray(self.flat("cand_category_id")).astype(np.int64)
        return self._c["_cat"]

    def ndcg(self, s):
        return ndcg_pf(self.off, self.M, np.ascontiguousarray(s, dtype=np.float32), self.y, self.pos_any)

    def bc(self, v):  # broadcast feed-level to candidates
        return np.repeat(v, self.M).astype(np.float32)

    def ctx(self):
        """feed-level context columns (constant within a feed)"""
        if "_ctx" not in self._c:
            cold = np.bincount(self.fid, weights=(self.get("it_hist_all_luo") == 0), minlength=self.nf) / self.M
            key = self.fid * 128 + self.cat
            ncat = np.bincount(np.unique(key) // 128, minlength=self.nf)
            self._c["_ctx"] = {"logM": self.bc(np.log(self.M)), "logH": self.bc(np.log1p(self.H)),
                               "user0": self.bc((self.uid == 0).astype(np.float32)), "cold": self.bc(cold),
                               "ncatM": self.bc(ncat / self.M),
                               "smaxmean": self.bc(np.bincount(self.fid, weights=self.get("seq_max_all"), minlength=self.nf) / self.M)}
        return self._c["_ctx"]


# ---------------------------------------------------------------- embedding / history kernel
ENAMES = (["e_last1", "e_last2", "e_last4", "e_last16", "e_last64", "e_sc_max", "e_sc_mean", "e_sc_cent", "e_oc_max",
           "e_sc_minus_oc", "e_n08", "e_n09", "e_n06", "e_frac07", "rep_cnt", "rep_rec", "rep_et1", "rep_etx",
           "e_dec", "e_argmax_rec", "c_nn1", "c_n09", "c_n08", "c_cent", "c_sc_cent", "c_sc_nn1", "c_dup", "c_top5mean"]
          + [f"pc{k}_c" for k in range(8)] + [f"pc{k}_cu" for k in range(8)] + [f"pc{k}_dh" for k in range(8)]
          + ["hf4_all", "hf4_sc", "hf4_l8", "hf5_all", "hf5_sc", "hf5_l8", "hf4_abs", "hf_cov"])


@nb.njit(parallel=True, cache=True)
def _emb_kernel(M, off, H, hoff, cemb, citem, ccat, cf4, cf5, hemb, hitem, hcat, het, hf4, hf5, table, uemb, V, nout):
    out = np.zeros((int(M.sum()), nout), np.float32)
    for f in nb.prange(len(M)):
        m = M[f]; o = off[f]; h = H[f]; ho = hoff[f]
        C = np.zeros((m, 128), np.float32)
        cok = np.zeros(m, np.bool_)
        for i in range(m):
            e = cemb[o + i]
            if e >= 0:
                v = table[e].astype(np.float32)
                nrm = np.sqrt((v * v).sum())
                if nrm > 1e-3:
                    C[i] = v / nrm; cok[i] = True
        Hh = np.zeros((max(h, 1), 128), np.float32)
        hok = np.zeros(max(h, 1), np.bool_)
        for k in range(h):
            e = hemb[ho + k]
            if e >= 0:
                v = table[e].astype(np.float32)
                nrm = np.sqrt((v * v).sum())
                if nrm > 1e-3:
                    Hh[k] = v / nrm; hok[k] = True
        S = C @ Hh.T
        # last-k centroids
        ks = (1, 2, 4, 16, 64)
        cents = np.zeros((5, 128), np.float32)
        acc = np.zeros(128, np.float32); cnt = 0; kk = 0
        for k in range(h):
            if hok[k]:
                acc += Hh[k]; cnt += 1
                while kk < 5 and cnt == ks[kk]:
                    nr = np.sqrt((acc * acc).sum())
                    cents[kk] = acc / max(nr, 1e-6); kk += 1
        while kk < 5:
            nr = np.sqrt((acc * acc).sum())
            cents[kk] = acc / max(nr, 1e-6); kk += 1
        # per-category history centroid norms
        csum = np.zeros((128, 128), np.float32)
        for k in range(h):
            if hok[k]:
                csum[hcat[ho + k]] += Hh[k]
        cnorm = np.sqrt((csum * csum).sum(1))
        # history f4/f5 means
        a4 = 0.0; n4 = 0.0; l4 = 0.0; nl4 = 0.0; a5 = 0.0; l5 = 0.0
        s4c = np.zeros(128); s5c = np.zeros(128); n4c = np.zeros(128)
        for k in range(h):
            v4 = hf4[ho + k]
            if not np.isnan(v4):
                v5 = hf5[ho + k]
                a4 += v4; a5 += v5; n4 += 1
                s4c[hcat[ho + k]] += v4; s5c[hcat[ho + k]] += v5; n4c[hcat[ho + k]] += 1
                if k < 8:
                    l4 += v4; l5 += v5; nl4 += 1
        # PCs
        pcC = C @ V  # [m,8]
        pcH = np.zeros(8, np.float32); nh = 0
        for k in range(h):
            if hok[k]:
                pcH += Hh[k] @ V; nh += 1
        if nh > 0:
            pcH /= nh
        u = uemb[f].astype(np.float32)
        nu = np.sqrt((u * u).sum())
        pcU = (u / np.float32(max(nu, 1e-3))).astype(np.float32) @ V
        # candidate-candidate
        CC = C @ C.T
        ccent = np.zeros(128, np.float32)
        for i in range(m):
            ccent += C[i]
        ccn = np.sqrt((ccent * ccent).sum())
        ccs = np.zeros((128, 128), np.float32)
        for i in range(m):
            ccs[ccat[o + i]] += C[i]
        for i in range(m):
            r = out[o + i]
            ci = ccat[o + i]
            for q in range(5):
                r[q] = (C[i] * cents[q]).sum() if h > 0 else 0.0
            scmax = -1.5; scsum = 0.0; scn = 0.0; ocmax = -1.5; n08 = 0.0; n09 = 0.0; n06 = 0.0; n07 = 0.0; nv = 0.0
            rep = 0.0; first = -1; rep1 = 0.0; repx = 0.0; dec = 0.0; dw = 0.0; w = 1.0; amax = -2.0; apos = 0
            for k in range(h):
                if hitem[ho + k] == citem[o + i]:
                    rep += 1
                    if first < 0:
                        first = k
                    if het[ho + k] == 1:
                        rep1 += 1
                    else:
                        repx += 1
                if hok[k]:
                    s = S[i, k]; nv += 1
                    if hcat[ho + k] == ci:
                        scsum += s; scn += 1
                        if s > scmax:
                            scmax = s
                    elif s > ocmax:
                        ocmax = s
                    if s > 0.6:
                        n06 += 1
                    if s > 0.7:
                        n07 += 1
                    if s > 0.8:
                        n08 += 1
                    if s > 0.9:
                        n09 += 1
                    dec += w * s; dw += w
                    if s > amax:
                        amax = s; apos = k
                w *= 0.9
            r[5] = scmax; r[6] = scsum / scn if scn > 0 else -1.5
            r[7] = scsum / cnorm[ci] if cnorm[ci] > 1e-6 else -1.5
            r[8] = ocmax; r[9] = (scmax - ocmax) if (scn > 0 and ocmax > -1.4) else 0.0
            r[10] = np.log1p(n08); r[11] = np.log1p(n09); r[12] = np.log1p(n06); r[13] = n07 / nv if nv > 0 else 0.0
            r[14] = np.log1p(rep); r[15] = 1.0 / (1.0 + first) if first >= 0 else 0.0
            r[16] = np.log1p(rep1); r[17] = np.log1p(repx)
            r[18] = dec / dw if dw > 0 else -1.5
            r[19] = 1.0 / (1.0 + apos) if nv > 0 else 0.0
            nn1 = -1.5; c09 = 0.0; c08 = 0.0; scnn = -1.5; dup = 0.0
            t5 = np.full(5, -1.5)
            for j in range(m):
                if j != i:
                    s = CC[i, j]
                    if s > nn1:
                        nn1 = s
                    if s > 0.9:
                        c09 += 1
                    if s > 0.8:
                        c08 += 1
                    if ccat[o + j] == ci and s > scnn:
                        scnn = s
                    if citem[o + j] == citem[o + i]:
                        dup += 1
                    if s > t5[0]:
                        t5[0] = s
                        for q in range(4):
                            if t5[q] > t5[q + 1]:
                                tmp = t5[q]; t5[q] = t5[q + 1]; t5[q + 1] = tmp
            r[20] = nn1; r[21] = np.log1p(c09); r[22] = np.log1p(c08)
            r[23] = ((C[i] * ccent).sum() - 1.0) / max(ccn, 1e-6) if cok[i] else 0.0
            cs = ccs[ci] - C[i]
            cs_n = np.sqrt((cs * cs).sum())
            r[24] = (C[i] * cs).sum() / cs_n if cs_n > 1e-6 else 0.0
            r[25] = scnn if scnn > -1.4 else 0.0
            r[26] = dup; r[27] = t5.mean()
            for q in range(8):
                r[28 + q] = pcC[i, q]; r[36 + q] = pcC[i, q] * pcU[q]; r[44 + q] = pcC[i, q] - pcH[q]
            v4 = cf4[o + i]; v5 = cf5[o + i]
            r[52] = v4 - a4 / n4 if n4 > 0 else 0.0
            r[53] = v4 - s4c[ci] / n4c[ci] if n4c[ci] > 0 else 0.0
            r[54] = v4 - l4 / nl4 if nl4 > 0 else 0.0
            r[55] = v5 - a5 / n4 if n4 > 0 else 0.0
            r[56] = v5 - s5c[ci] / n4c[ci] if n4c[ci] > 0 else 0.0
            r[57] = v5 - l5 / nl4 if nl4 > 0 else 0.0
            r[58] = -abs(r[52]); r[59] = np.log1p(n4c[ci])
    return out


def item_f_table():
    """item-level mean f4 / f5 over TRAIN candidate rows (label-free); used only for HISTORY items"""
    p = CACHE / "item_f45.npz"
    if p.exists():
        z = np.load(p)
        return z["ids"], z["f4"], z["f5"]
    it = np.load(FLAT / "train/cand_item_id.npy", mmap_mode="r")
    f4 = np.load(FLAT / "train/cand_f4.npy", mmap_mode="r"); f5 = np.load(FLAT / "train/cand_f5.npy", mmap_mode="r")
    n = len(it); mx = 0
    for a in range(0, n, 50_000_000):
        mx = max(mx, int(it[a:a + 50_000_000].max()))
    c = np.zeros(mx + 1, np.float32); s4 = np.zeros(mx + 1, np.float64); s5 = np.zeros(mx + 1, np.float64)
    for a in range(0, n, 25_000_000):
        i = np.asarray(it[a:a + 25_000_000])
        c += np.bincount(i, minlength=mx + 1).astype(np.float32)
        s4 += np.bincount(i, weights=np.asarray(f4[a:a + 25_000_000], dtype=np.float64), minlength=mx + 1)
        s5 += np.bincount(i, weights=np.asarray(f5[a:a + 25_000_000], dtype=np.float64), minlength=mx + 1)
    ids = np.where(c > 0)[0]
    np.savez(p, ids=ids, f4=(s4[ids] / c[ids]).astype(np.float32), f5=(s5[ids] / c[ids]).astype(np.float32))
    return item_f_table()


def pca_V():
    p = CACHE / "pcaV.npy"
    if p.exists():
        return np.load(p)
    d = COMP / "train" / "item_embeddings"
    meta = json.loads((d / "data.meta").read_text())
    T = np.memmap(d / "data.memmap", dtype=np.int8, mode="r", shape=tuple(meta["shape"]))
    idx = np.sort(np.random.default_rng(0).choice(T.shape[0], 300000, replace=False))
    X = np.asarray(T[idx], dtype=np.float32)
    X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-3)
    X -= X.mean(0)  # centring only defines the directions; projections are of uncentred unit vectors
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    np.save(p, Vt[:8].T.astype(np.float32).copy())
    return np.load(p)


def emb_feats(D):
    p = CACHE / f"emb_{D.name}.npy"
    if p.exists():
        return np.load(p, mmap_mode="r")
    sp = D.split
    d = COMP / sp / "item_embeddings"
    meta = json.loads((d / "data.meta").read_text())
    table = np.memmap(d / "data.memmap", dtype=np.int8, mode="r", shape=tuple(meta["shape"]))
    hrows = np.concatenate([np.arange(o, o + h) for o, h in zip(D.hoff, D.H)]) if D.H.sum() else np.zeros(0, np.int64)
    hoff = np.concatenate([[0], np.cumsum(D.H)[:-1]]).astype(np.int64)

    def hl(n):
        a = np.load(FLAT / sp / f"{n}.npy", mmap_mode="r")
        return np.ascontiguousarray(a[hrows])
    hemb = hl("hist_emb_idx").astype(np.int64); hitem = hl("hist_item_id").astype(np.int64)
    hcat = hl("hist_category_id").astype(np.int64); het = hl("hist_event_type").astype(np.int64)
    ids, t4, t5 = item_f_table()
    j = np.clip(np.searchsorted(ids, hitem), 0, len(ids) - 1)
    okk = ids[j] == hitem
    hf4 = np.where(okk, t4[j], np.nan).astype(np.float32); hf5 = np.where(okk, t5[j], np.nan).astype(np.float32)
    # compact the item table to the rows we need (keeps the kernel off the memmap)
    cemb = np.ascontiguousarray(D.flat("cand_emb_idx")).astype(np.int64)
    used = np.unique(np.concatenate([cemb[cemb >= 0], hemb[hemb >= 0]]))
    sub = np.asarray(table[used])
    ce2 = np.where(cemb >= 0, np.searchsorted(used, np.maximum(cemb, 0)), -1)
    he2 = np.where(hemb >= 0, np.searchsorted(used, np.maximum(hemb, 0)), -1)
    ue = np.load(FLAT / sp / "user_emb.npy", mmap_mode="r")
    uemb = np.ascontiguousarray(ue[D.fsel])
    out = _emb_kernel(D.M, D.off, D.H, hoff, ce2, np.ascontiguousarray(D.flat("cand_item_id")).astype(np.int64),
                      D.cat, D.raw("f4"), D.raw("f5"), he2, hitem, hcat, het, hf4, hf5, sub, uemb, pca_V(), len(ENAMES))
    np.save(p, out)
    return out


@nb.njit(parallel=True, cache=True)
def _keygroup(off, M, key, x):
    """within (feed, key value): x - group mean; mid-rank within group; group mean - feed mean; group share"""
    n = len(x)
    dgm = np.empty(n, np.float32); grk = np.empty(n, np.float32); gmf = np.empty(n, np.float32); gsz = np.empty(n, np.float32)
    for f in nb.prange(len(M)):
        o, m = off[f], M[f]
        tot = 0.0
        for k in range(o, o + m):
            tot += x[k]
        tot /= m
        for k in range(o, o + m):
            s = 0.0; c = 0.0; less = 0.0; eq = 0.0
            for j in range(o, o + m):
                if key[j] == key[k]:
                    s += x[j]; c += 1.0
                    if x[j] < x[k]:
                        less += 1.0
                    elif x[j] == x[k]:
                        eq += 1.0
            gm = s / c
            dgm[k] = x[k] - gm; gmf[k] = gm - tot; gsz[k] = c / m
            grk[k] = (less + 0.5 * (eq - 1.0)) / (c - 1.0) if c > 1 else 0.5
    return dgm, grk, gmf, gsz


@nb.njit(parallel=True, cache=True)
def _histdev(M, off, H, hoff, ccat, cx, hcat, hx):
    """candidate value minus the mean of the same item-level quantity over the user's history items:
    all / same category / last 8 (NaN history values are skipped; 0 when no covered history item)"""
    out = np.zeros((int(M.sum()), 3), np.float32)
    for f in nb.prange(len(M)):
        m = M[f]; o = off[f]; h = H[f]; ho = hoff[f]
        a = 0.0; n = 0.0; l = 0.0; nl = 0.0
        sc = np.zeros(128); nc = np.zeros(128)
        for k in range(h):
            v = hx[ho + k]
            if not np.isnan(v):
                a += v; n += 1; sc[hcat[ho + k]] += v; nc[hcat[ho + k]] += 1
                if k < 8:
                    l += v; nl += 1
        for i in range(m):
            c = ccat[o + i]
            out[o + i, 0] = cx[o + i] - a / n if n > 0 else 0.0
            out[o + i, 1] = cx[o + i] - sc[c] / nc[c] if nc[c] > 0 else 0.0
            out[o + i, 2] = cx[o + i] - l / nl if nl > 0 else 0.0
    return out


def item_pop_table():
    """item-level mean log1p(it_hist_all_luo) over TRAIN candidate rows (label-free); used for HISTORY items only"""
    p = CACHE / "item_pop.npz"
    if p.exists():
        z = np.load(p)
        return z["ids"], z["pop"]
    it = np.load(FLAT / "train/cand_item_id.npy", mmap_mode="r")
    x = np.load(featfile("train", "it_hist_all_luo"), mmap_mode="r")
    n = len(it); mx = 0
    for a in range(0, n, 50_000_000):
        mx = max(mx, int(it[a:a + 50_000_000].max()))
    c = np.zeros(mx + 1, np.float32); s = np.zeros(mx + 1, np.float64)
    for a in range(0, n, 25_000_000):
        i = np.asarray(it[a:a + 25_000_000])
        c += np.bincount(i, minlength=mx + 1).astype(np.float32)
        s += np.bincount(i, weights=np.log1p(np.maximum(np.nan_to_num(np.asarray(x[a:a + 25_000_000], dtype=np.float64)), 0)), minlength=mx + 1)
    ids = np.where(c > 0)[0]
    np.savez(p, ids=ids, pop=(s[ids] / c[ids]).astype(np.float32))
    return item_pop_table()


def histpop_feats(D):
    p = CACHE / f"hpop_{D.name}.npy"
    if p.exists():
        return np.load(p)
    sp = D.split
    hrows = np.concatenate([np.arange(o, o + h) for o, h in zip(D.hoff, D.H)])
    hoff = np.concatenate([[0], np.cumsum(D.H)[:-1]]).astype(np.int64)
    hitem = np.ascontiguousarray(np.load(FLAT / sp / "hist_item_id.npy", mmap_mode="r")[hrows]).astype(np.int64)
    hcat = np.ascontiguousarray(np.load(FLAT / sp / "hist_category_id.npy", mmap_mode="r")[hrows]).astype(np.int64)
    ids, pop = item_pop_table()
    j = np.clip(np.searchsorted(ids, hitem), 0, len(ids) - 1)
    hx = np.where(ids[j] == hitem, pop[j], np.nan).astype(np.float32)
    out = _histdev(D.M, D.off, D.H, hoff, D.cat, D.get("it_hist_all_luo"), hcat, hx)
    np.save(p, out)
    return out


# ---------------------------------------------------------------- generators
def families():
    return ["A_feedrel", "B_mono", "C_pair", "C_ctx", "D_catgrp", "E_hist", "F_emb", "C2_pair", "G_efx", "H_het", "C3_pair", "D2_grp", "E2_hist"]


def gen(D, fams=None, want=None, pairsel=None):
    """yield (family, name, float32 array). `want`: optional set of names (others are skipped cheaply)."""
    fams = fams or families()
    w = (lambda n: True) if want is None else (lambda n: n in want)
    off, M = D.off, D.M
    nonconst = [b for b in BASE]
    if "A_feedrel" in fams:
        for b in nonconst:
            names = [f"rk({b})", f"dmax({b})", f"dmed({b})", f"gap({b})", f"cv({b})", f"zrob({b})"]
            if not any(w(n) for n in names):
                continue
            x = D.get(b)
            if w(names[0]):
                yield "A_feedrel", names[0], _rk(off, M, x)
            dmax, dmed, gap, mean, sd = _feedstat(off, M, x)
            if w(names[1]): yield "A_feedrel", names[1], dmax
            if w(names[2]): yield "A_feedrel", names[2], dmed
            if w(names[3]): yield "A_feedrel", names[3], gap
            if w(names[4]): yield "A_feedrel", names[4], (x - mean) / (np.abs(mean) + 0.1)
            if w(names[5]): yield "A_feedrel", names[5], dmed / (np.abs(dmax - dmed) + 0.05)  # (x-med)/(max-x+..)
    if "B_mono" in fams:
        for b in nonconst:
            if b.startswith("has:"):
                continue
            names = [f"log({b})", f"sqrt({b})", f"sq({b})", f"qn({b})", f"exp({b})"]
            if not any(w(n) for n in names):
                continue
            x = D.get(b)
            lo = float(STATS[b]["min"])
            sc = float(STATS[b]["std"]) + 1e-6
            xs = np.maximum(x - lo, 0) / sc
            if w(names[0]): yield "B_mono", names[0], np.log1p(xs)
            if w(names[1]): yield "B_mono", names[1], np.sqrt(xs)
            if w(names[2]): yield "B_mono", names[2], xs * xs
            if w(names[3]):
                q = np.asarray(STATS[b]["q"], dtype=np.float32)
                yield "B_mono", names[3], (np.searchsorted(q, x) / len(q)).astype(np.float32)
            if w(names[4]): yield "B_mono", names[4], np.exp(np.minimum(xs, 8.0))
    if "C_pair" in fams or "C_ctx" in fams:
        zc = {}
        rc = {}

        def Z(b):
            if b not in zc:
                zc[b] = _z(off, M, D.get(b))
            return zc[b]

        def R(b):
            if b not in rc:
                rc[b] = _rk(off, M, D.get(b))
            return rc[b]
        if "C_pair" in fams:
            for pi, (a, b) in enumerate(itertools.combinations(CORE, 2)):
                if pairsel is not None and not pairsel(pi):
                    continue
                names = [f"zmul({a},{b})", f"zdiff({a},{b})", f"ratio({a},{b})", f"rmin({a},{b})", f"rmax({a},{b})"]
                if not any(w(n) for n in names):
                    continue
                if w(names[0]): yield "C_pair", names[0], Z(a) * Z(b)
                if w(names[1]): yield "C_pair", names[1], Z(a) - Z(b)
                if w(names[2]):
                    xa = (D.get(a) - STATS[a]["min"]) / (STATS[a]["std"] + 1e-6)
                    xb = (D.get(b) - STATS[b]["min"]) / (STATS[b]["std"] + 1e-6)
                    yield "C_pair", names[2], ((xa + 0.1) / (xb + 0.1)).astype(np.float32)
                if w(names[3]): yield "C_pair", names[3], np.minimum(R(a), R(b))
                if w(names[4]): yield "C_pair", names[4], np.maximum(R(a), R(b))
        if "C_ctx" in fams:
            ctx = D.ctx()
            for a in CORE:
                for c, v in ctx.items():
                    n = f"zctx({a},{c})"
                    if w(n):
                        vc = (v - STATS["ctx:" + c]["mean"]) / (STATS["ctx:" + c]["std"] + 1e-6)
                        yield "C_ctx", n, Z(a) * vc.astype(np.float32)
    if "C2_pair" in fams:  # wave 2: every remaining pair of the 55 real-valued inputs (those not both in CORE)
        zc2 = {}; rc2 = {}

        def Z2(b):
            if b not in zc2:
                if len(zc2) > 12:
                    zc2.pop(next(iter(zc2)))
                zc2[b] = _z(off, M, D.get(b))
            return zc2[b]

        def R2(b):
            if b not in rc2:
                if len(rc2) > 12:
                    rc2.pop(next(iter(rc2)))
                rc2[b] = _rk(off, M, D.get(b))
            return rc2[b]
        B55 = [b for b in BASE if not b.startswith("has:")]
        pi = -1
        for a, b in itertools.combinations(B55, 2):
            if a in CORE and b in CORE:
                continue
            pi += 1
            if pairsel is not None and not pairsel(pi):
                continue
            names = [f"zmul({a},{b})", f"zdiff({a},{b})", f"ratio({a},{b})", f"rmin({a},{b})", f"rmax({a},{b})"]
            if not any(w(n) for n in names):
                continue
            if w(names[0]): yield "C2_pair", names[0], Z2(a) * Z2(b)
            if w(names[1]): yield "C2_pair", names[1], Z2(a) - Z2(b)
            if w(names[2]):
                xa = (D.get(a) - STATS[a]["min"]) / (STATS[a]["std"] + 1e-6)
                xb = (D.get(b) - STATS[b]["min"]) / (STATS[b]["std"] + 1e-6)
                yield "C2_pair", names[2], ((xa + 0.1) / (xb + 0.1)).astype(np.float32)
            if w(names[3]): yield "C2_pair", names[3], np.minimum(R2(a), R2(b))
            if w(names[4]): yield "C2_pair", names[4], np.maximum(R2(a), R2(b))
    if "G_efx" in fams:  # wave 2: new history / embedding scalars x anchor inputs
        E = emb_feats(D)
        for an in ("cos_u", "f3", "f4", "f5", "seq_max_all", "it_hist_all_luo"):
            za = None
            for j, n in enumerate(ENAMES):
                for op in ("zmul", "rmax"):
                    nm = f"{op}({n},{an})"
                    if not w(nm):
                        continue
                    xe = np.ascontiguousarray(E[:, j])
                    if op == "zmul":
                        yield "G_efx", nm, _z(off, M, xe) * _z(off, M, D.get(an))
                    else:
                        yield "G_efx", nm, np.maximum(_rk(off, M, xe), _rk(off, M, D.get(an)))
    if "C3_pair" in fams:  # wave 3: four more binary operators on the CORE pairs
        zc3 = {}; rc3 = {}

        def Z3(b):
            if b not in zc3:
                zc3[b] = _z(off, M, D.get(b))
            return zc3[b]

        def R3(b):
            if b not in rc3:
                rc3[b] = _rk(off, M, D.get(b))
            return rc3[b]
        for pi, (a, b) in enumerate(itertools.combinations(CORE, 2)):
            if pairsel is not None and not pairsel(pi):
                continue
            names = [f"rdiff({a},{b})", f"rmul({a},{b})", f"zabs({a},{b})", f"zmax({a},{b})"]
            if not any(w(n) for n in names):
                continue
            if w(names[0]): yield "C3_pair", names[0], R3(a) - R3(b)
            if w(names[1]): yield "C3_pair", names[1], R3(a) * R3(b)
            if w(names[2]): yield "C3_pair", names[2], np.abs(Z3(a) - Z3(b))
            if w(names[3]): yield "C3_pair", names[3], np.maximum(Z3(a), Z3(b))
    if "D2_grp" in fams:  # wave 3: group statistics within (feed, f2 level = region class) and (feed, cold / warm)
        for kn in ("f2", "cold"):
            key = D.get("f2") if kn == "f2" else (D.get("it_hist_all_luo") == 0).astype(np.float32)
            first = True
            for b in CORE:
                if b == kn or (kn == "cold" and b == "it_hist_all_luo"):
                    continue
                names = [f"kdgm[{kn}]({b})", f"kgrk[{kn}]({b})", f"kgmf[{kn}]({b})"]
                if not (any(w(n) for n in names) or (first and w(f"kgsz[{kn}]"))):
                    continue
                dgm, grk, gmf, gsz = _keygroup(off, M, key, D.get(b))
                for n, v in zip(names, (dgm, grk, gmf)):
                    if w(n): yield "D2_grp", n, v
                if first and w(f"kgsz[{kn}]"):
                    yield "D2_grp", f"kgsz[{kn}]", gsz
                first = False
    if "E2_hist" in fams:  # wave 3: candidate popularity minus the popularity of the user's history items
        HP = histpop_feats(D)
        for j, n in enumerate(("hpop_all", "hpop_sc", "hpop_l8")):
            if w(n): yield "E2_hist", n, np.ascontiguousarray(HP[:, j])
            if w(f"rk({n})"): yield "E2_hist", f"rk({n})", _rk(off, M, np.ascontiguousarray(HP[:, j]))
    if "H_het" in fams:  # masked feed-relative versions of the per-event-type max-cos inputs not in BASE (et5..et10)
        for t in range(5, 11):
            b = f"cos_het{t}"
            names = [f"rk({b})", f"dmed({b})", f"dmax({b})"]
            if not any(w(n) for n in names):
                continue
            x = D.get(b)
            dmax, dmed, gap, mean, sd = _feedstat(off, M, x)
            if w(names[0]): yield "H_het", names[0], _rk(off, M, x)
            if w(names[1]): yield "H_het", names[1], dmed
            if w(names[2]): yield "H_het", names[2], dmax
    if "D_catgrp" in fams:
        for b in CORE:
            if b in ("f3", "feed_cat_share", "cat_cnt_all", "cat_share_all", "cat_min_hpos"):
                continue  # constant within (feed, category)
            names = [f"dgm({b})", f"grk({b})", f"gmf({b})", f"dgx({b})"]
            if not any(w(n) for n in names):
                continue
            dgm, grk, gmf, dgx = _catgroup(off, M, D.cat, D.get(b))
            for n, v in zip(names, (dgm, grk, gmf, dgx)):
                if w(n): yield "D_catgrp", n, v
    if "E_hist" in fams or "F_emb" in fams:
        E = emb_feats(D)
        for j, n in enumerate(ENAMES):
            fam = "F_emb" if (n.startswith("c_") or n.startswith("pc")) else "E_hist"
            if fam not in fams:
                continue
            if w(n): yield fam, n, np.ascontiguousarray(E[:, j])
            if w(f"rk({n})"): yield fam, f"rk({n})", _rk(off, M, np.ascontiguousarray(E[:, j]))


STATS = {}


def load_stats():
    """split-independent constants (min / std / quantiles of each base input), computed once on val and frozen"""
    p = CACHE / "stats.json"
    if not p.exists():
        D = DS("val")
        st = {}
        for b in BASE:
            x = D.get(b)
            st[b] = {"min": float(x.min()), "std": float(x.std()), "mean": float(x.mean()),
                     "q": np.unique(np.quantile(x, np.linspace(0.01, 0.99, 99))).tolist()}
        for c, v in D.ctx().items():
            st["ctx:" + c] = {"mean": float(v.mean()), "std": float(v.std())}
        p.write_text(json.dumps(st))
    STATS.update(json.loads(p.read_text()))
    return STATS
