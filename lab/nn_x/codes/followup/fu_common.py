"""Shared, unchanged-definition helpers for the codes follow-ups (PREDECLARED.txt):
x02 nested residual harness, tail.py tail/beater pairs + clustered win rates, stage0 buried cell, G3 holdout pool."""
import os
for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "POLARS_MAX_THREADS"]:
    os.environ[k] = "12"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys
import numpy as np, polars as pl
from pathlib import Path
sys.path.insert(0, "<repo>/src"); sys.path.insert(0, "<repo>/eda/errors/synthesis")
from metrics import ndcg_per_feed
from stack_val import feed_rank_z

ROOT = Path("<repo>"); FLAT = ROOT / "data/flat"; FEATS = ROOT / "data/feats"; P = ROOT / "preds"
OUT = ROOT / "nn_x/codes/followup/out"
W_GRID = [-1, -0.5, -0.3, -0.2, -0.1, -0.05, 0, 0.05, 0.1, 0.2, 0.3, 0.5, 1]
G3_MEMBERS = [f"v23_gru_pre2s_s{s}" for s in range(501, 507)] + [f"r6_cd_m02_r{s}" for s in range(101, 108)] + \
             [f"r6_cd_m02_p70k_r{s}" for s in range(101, 108)]


class Split:
    """feed structure + labels for 'val' or 'hold' (the 30K train holdout)."""

    def __init__(self, kind):
        self.kind = kind
        if kind == "val":
            f = pl.read_parquet(FLAT / "val/feeds.parquet")
            self.M = f["M"].to_numpy().astype(np.int64); self.off = f["cand_off"].to_numpy().astype(np.int64)
            self.users = f["user_id"].to_numpy().astype(np.int64)
            self.ts = f["timestamp"].to_numpy()
            self.y = np.load(FLAT / "val/cand_target.npy").astype(np.float64)
        else:
            f = pl.read_parquet(FLAT / "train/feeds.parquet")
            hold = np.load(OUT / "hold_fidx.npy")
            Mh, oh = np.load(P / "holdout/holdout_meta.npy")
            self.M = Mh.astype(np.int64); self.off = oh.astype(np.int64)
            assert (f["M"].to_numpy()[hold] == self.M).all()
            self.users = f["user_id"].to_numpy().astype(np.int64)[hold]
            self.ts = f["timestamp"].to_numpy()[hold]
            self.y = np.load(P / "holdout/holdout_y.npy").astype(np.float64)
        self.n = len(self.M); self.N = int(self.M.sum())
        self.fid = np.repeat(np.arange(self.n), self.M)
        cl = self.users.copy(); z = cl == 0; cl[z] = -(np.arange(z.sum()) + 1)
        _, self.cl = np.unique(cl, return_inverse=True)
        # x02 folds
        rng = np.random.default_rng(0)
        uu = np.unique(self.users[self.users > 0]); half = set(rng.choice(uu, len(uu) // 2, replace=False).tolist())
        self.fold = np.array([(u in half) if u > 0 else (i % 2 == 0) for i, u in enumerate(self.users)])

    def fz(self, x):
        x = np.asarray(x, np.float64)
        m = np.bincount(self.fid, x, minlength=self.n) / self.M
        s = np.sqrt(np.maximum(np.bincount(self.fid, x * x, minlength=self.n) / self.M - m ** 2, 0))
        return (x - m[self.fid]) / np.where(s[self.fid] > 0, s[self.fid], 1)

    def rz(self, x):
        return feed_rank_z(self.off, self.M, np.asarray(x, np.float32))[:, 1].astype(np.float64)

    def nd(self, s):
        return ndcg_per_feed(np.asarray(s, np.float64), self.y, self.off, self.M)


def g3_score(S):
    if S.kind == "val":
        return np.load(P / "ens_G3_final_val.npy").astype(np.float64)
    zs = [S.rz(np.load(P / f"holdout/nn_{m}_holdout.npy")) for m in G3_MEMBERS]
    return S.rz(np.mean(zs, 0))


def nested(S, base_z, x):
    """x02: fz(G3) + w fz(x), nested 2-fold by user. Returns (delta per feed, fold weights)."""
    base = S.nd(base_z)
    xz = S.fz(x)
    per = {w: S.nd(base_z + w * xz) for w in W_GRID}
    out = np.empty(S.n); ws = []
    for a in [True, False]:
        sel = S.fold == a
        wb = max(W_GRID, key=lambda w: per[w][sel].mean()); ws.append(wb); out[~sel] = per[wb][~sel]
    return out - base, ws


def actexp_bins(S, gz, x, y):
    """G3 act/exp by feature quintile (x02 readout)."""
    fid, M = S.fid, S.M
    o = np.lexsort((gz, fid)); fs_ = np.r_[True, fid[o][1:] != fid[o][:-1]]
    rk = np.arange(len(o)) - np.flatnonzero(fs_)[np.cumsum(fs_) - 1]
    pct = np.empty(len(o)); pct[o] = rk / np.maximum(M[fid[o]] - 1, 1)
    dec = np.clip((pct * 10).astype(int), 0, 9)
    rate = np.bincount(dec, y, minlength=10) / np.bincount(dec, minlength=10); ex = rate[dec]
    ex = ex * (np.bincount(fid, y, minlength=S.n) / np.maximum(np.bincount(fid, ex, minlength=S.n), 1e-12))[fid]
    qs = np.quantile(x, [0.2, 0.4, 0.6, 0.8]); b = np.searchsorted(np.unique(qs), x, side='right')
    return ' '.join(f"{np.mean(b == i):.2f}/{y[b == i].sum() / max(ex[b == i].sum(), 1e-9):.3f}" for i in range(b.max() + 1) if np.any(b == i))


def ratio_cl(num, den, cl):
    """pooled ratio sum(num)/sum(den) with cluster-robust SE (tail.py ratio_cl); num/den per feed."""
    r = num.sum() / max(den.sum(), 1e-12); g = cl.max() + 1
    u = np.bincount(cl, num - r * den, g)
    return r, np.sqrt(g / max(g - 1, 1) * (u ** 2).sum()) / max(den.sum(), 1e-12)


class Tail:
    """tail.py: ranks under s with the metric tie rule (score desc, negatives first on ties); tail feeds; focal positive;
    beaters = negatives ranked above the focal positive."""

    def __init__(self, S, s, headk=8):
        N, n, fid, y, off, M = S.N, S.n, S.fid, S.y, S.off, S.M
        order = np.lexsort((y, -s, fid))
        rank = np.empty(N, np.int64); rank[order] = np.arange(N) - np.repeat(off, M)
        npos = np.bincount(fid, y, n)
        Pp = np.where(y > 0)[0]
        po = Pp[np.lexsort((rank[Pp], fid[Pp]))]
        first = np.ones(len(po), bool); first[1:] = fid[po][1:] != fid[po][:-1]
        foc = np.full(n, -1); foc[fid[po][first]] = po[first]
        haspos = npos > 0
        bpr = np.where(haspos, rank[np.maximum(foc, 0)], 10 ** 9)
        self.tail = haspos & (bpr >= headk)
        self.fp = np.maximum(foc, 0)
        self.above = (rank < bpr[fid]) & (y == 0)
        self.rank = rank; self.S = S

    def win(self, x, extra=None):
        """P(x_focal_pos > x_beater) + 0.5 ties over tail beater pairs; extra = optional pair mask (row-level, for the beater)."""
        S = self.S
        xp = x[self.fp][S.fid]
        ok = np.isfinite(x) & np.isfinite(xp) & self.above & self.tail[S.fid]
        if extra is not None:
            ok &= extra
        w = (xp > x).astype(float) + 0.5 * (xp == x)
        num = np.bincount(S.fid, w * ok, S.n); den = np.bincount(S.fid, ok, S.n).astype(float)
        r, se = ratio_cl(num, den, S.cl)
        # share of pairs where the cue is NOT tied (informative pairs) and the win rate among them
        nt = ok & (xp != x)
        num2 = np.bincount(S.fid, w * nt, S.n); den2 = np.bincount(S.fid, nt, S.n).astype(float)
        r2, se2 = ratio_cl(num2, den2, S.cl) if den2.sum() > 0 else (np.nan, np.nan)
        return dict(win=r, se=se, pairs=int(ok.sum()), untied_share=float(nt.sum() / max(ok.sum(), 1)), win_untied=r2, se_untied=se2)


def buried_cell(S, keep):
    """stage0_gate.py cell: zG3 = 0.5 rz(ft) + 0.5 rz(to); cell positive = category is G3's LSE-top (pcr == 0) and >= 3 same-category
    candidates above it (wr >= 3); pairs (cell positive, same-category negatives with higher zG3) in keep (leak-free) feeds."""
    fid, y, off, M, n = S.fid, S.y, S.off, S.M, S.n
    cat = np.load(FLAT / "val/cand_category_id.npy").astype(np.int64)
    zG3 = 0.5 * S.rz(np.load(P / "ens_G3_ft_val.npy")) + 0.5 * S.rz(np.load(P / "ens_G3_to_val.npy"))
    ck = fid * 64 + cat; uk, inv = np.unique(ck, return_inverse=True)
    lse = np.log(np.bincount(inv, np.exp(zG3)))
    kf = uk // 64; o = np.lexsort((-lse, kf)); cr = np.empty(len(uk), np.int64); cr[o] = np.arange(len(uk)) - np.searchsorted(kf[o], kf[o])
    pcr = cr[inv]
    o2 = np.lexsort((y, -zG3, ck)); wr = np.empty(len(y), np.int64); wr[o2] = np.arange(len(y)) - np.searchsorted(ck[o2], ck[o2])
    cellpos = (y > 0) & (pcr == 0) & (wr >= 3)
    pi, ni = [], []
    for g in np.nonzero(keep)[0]:
        s, m = off[g], M[g]
        pp = np.nonzero(cellpos[s:s + m])[0] + s
        if len(pp) == 0:
            continue
        neg = np.nonzero(y[s:s + m] == 0)[0] + s
        for p_ in pp:
            q = neg[(cat[neg] == cat[p_]) & (zG3[neg] > zG3[p_])]
            pi.append(np.full(len(q), p_)); ni.append(q)
    pi = np.concatenate(pi); ni = np.concatenate(ni)
    return dict(zG3=zG3, cellpos=cellpos, pi=pi, ni=ni, cat=cat)


def pair_win(S, x, pi, ni, mask=None):
    """P(x[pi] > x[ni]) + 0.5 ties over the given pairs, user-clustered SE."""
    ok = np.isfinite(x[pi]) & np.isfinite(x[ni])
    if mask is not None:
        ok &= mask
    w = (x[pi] > x[ni]).astype(float) + 0.5 * (x[pi] == x[ni])
    f = S.fid[pi]
    num = np.bincount(f, w * ok, S.n); den = np.bincount(f, ok, S.n).astype(float)
    r, se = ratio_cl(num, den, S.cl)
    return r, se, int(ok.sum())
