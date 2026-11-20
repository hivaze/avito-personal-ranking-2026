"""Shared helpers of INVENTORY C (reports/INPUTS_1116/catboost_blocks). CPU only, read-only on every project file.
Witnesses: val (14,423 feeds, preds/ row order) and the 30,000-feed in-period holdout (nn_x/w5/preds/holdout row order).
NATS(X | B) = mean over feeds of [loss(z_B) - loss(z_B, z_X)], loss = listwise softmax (conditional logit, target y / n_pos),
coefficients fitted on the witness itself: the statistic of nn_x/w12/cb/nats.py, with a vectorised Newton step for many columns.
Start: cd <repo> && CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 ... NUMBA_CACHE_DIR=<this dir>/cache/numba PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -u <script>"""
import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = ''
CBI = '<repo>/reports/INPUTS_1116/catboost_blocks'
os.environ.setdefault('NUMBA_CACHE_DIR', CBI + '/cache/numba')
sys.dont_write_bytecode = True
sys.path.insert(0, '<repo>/nn_x/w12/recheck/pool_composition')
from pc_common import *          # Meta, ndcg_fast, se, fmt, ROOT, W, AB, P, np, feed_meta, FLAT   (sets the thread caps to 12)
import polars as pl
CBI = '<repo>/reports/INPUTS_1116/catboost_blocks'; CBI_OUT = CBI + '/out'   # the ONLY directory this inventory writes to
del HERE, CACHE, OUT          # pc_common's own directories: never written from here

FEATS = ROOT / 'data/feats'; FEATS_NEW = ROOT / 'data/feats_new'
CBX = ROOT / 'nn_x/w12/cb/infra/cache'


class Wit:
    """one witness: labels, offsets, feed sizes, fid, user ids, flat train rows (holdout only)."""
    def __init__(self, meta, sp):
        self.sp = sp; self.y = meta.y[sp].astype(np.float64); self.off = meta.off[sp]; self.M = meta.M[sp]; self.nf = len(self.M); self.n = int(self.M.sum())
        self.fid = np.repeat(np.arange(self.nf), self.M); self.meta = meta
        self.npos = np.bincount(self.fid, self.y, self.nf); self.t = self.y / np.maximum(self.npos[self.fid], 1)
        if sp == 'val':
            f = meta.fv; self.uid = f['user_id'].to_numpy(); self.ts = f['timestamp'].to_numpy(); self.rows = None
        else:
            f = pl.read_parquet(FLAT / 'train/feeds.parquet'); Mtr = f['M'].to_numpy().astype(np.int64)
            pr = np.load(ROOT / 'eda/labels_position/out/train_prandom.npz'); p = np.nan_to_num(pr['p_random'].astype(np.float32), nan=0.0); p[Mtr < 50] = 0.0
            cand = np.where((p > 0.9) & (Mtr >= 50))[0]; hold = np.sort(np.random.default_rng(12345).choice(cand, 30000, replace=False))
            fh = f[hold]; assert (fh['M'].to_numpy() == self.M).all()
            self.uid = fh['user_id'].to_numpy(); self.ts = fh['timestamp'].to_numpy(); self.hold = hold
            co = fh['cand_off'].to_numpy().astype(np.int64); self.rows = np.repeat(co - self.off, self.M) + np.arange(self.n)
            yt = np.load(FLAT / 'train/cand_target.npy', mmap_mode='r'); assert (np.asarray(yt[self.rows[:200000]]) == self.y[:200000]).all()
        self.zw16 = np.load(CBX / f"w16_z_{sp}.npy").astype(np.float64)

    def z(self, x):
        return self.meta.z(self.sp, np.asarray(x, np.float32)).astype(np.float64)

    def zcol(self, x):
        """per-feed z of a feature column; NaN / inf -> feed mean of the finite values (z = 0); constant-in-feed columns -> 0."""
        x = np.asarray(x, np.float64); ok = np.isfinite(x); xv = np.where(ok, x, 0.0)
        cnt = np.bincount(self.fid, ok, self.nf); mu = np.bincount(self.fid, xv, self.nf) / np.maximum(cnt, 1)
        d = np.where(ok, xv - mu[self.fid], 0.0); sd = np.sqrt(np.bincount(self.fid, d * d, self.nf) / np.maximum(cnt, 1))
        return np.where(sd[self.fid] > 1e-9, d / np.maximum(sd[self.fid], 1e-9), 0.0)

    def pool(self, files):
        return self.z(np.mean([self.z(np.load(f)) for f in files], 0))

    def ndcg(self, s):
        return ndcg_fast(np.asarray(s, np.float32), self.meta.y[self.sp], self.off, self.M)

    def halves(self, seed=0):
        """user-grouped halves (user 0 = pooled bucket: its feeds are split one by one)."""
        rng = np.random.default_rng(seed); uu = np.unique(self.uid); h = rng.random(len(uu)) < 0.5
        half = h[np.searchsorted(uu, self.uid)]; z0 = self.uid == 0; half[z0] = rng.random(int(z0.sum())) < 0.5
        return half


def clogit(X, w, feeds=None, ridge=1e-8, iters=40):
    """conditional logit of the label inside the feed on the columns of X (n_rows x p). feeds: boolean mask of feeds used for the FIT.
    Returns b and the per-feed loss of ALL feeds at b."""
    X = np.asarray(X, np.float64)
    if X.ndim == 1: X = X[:, None]
    n, p = X.shape; fid, off, t = w.fid, w.off, w.t
    rm = np.ones(n, bool) if feeds is None else feeds[fid]
    Xf = X[rm]; tf = t[rm]
    if feeds is None: offf = off; fidf = fid; nff = w.nf
    else:
        Mf = w.M[feeds]; offf = np.r_[0, np.cumsum(Mf)[:-1]]; nff = len(Mf); fidf = np.repeat(np.arange(nff), Mf)
    b = np.zeros(p)
    for _ in range(iters):
        s = Xf @ b; mx = np.maximum.reduceat(s, offf); e = np.exp(s - mx[fidf]); pr = e / np.add.reduceat(e, offf)[fidf]
        g = Xf.T @ (pr - tf); mu = np.add.reduceat(pr[:, None] * Xf, offf, axis=0)
        H = (Xf * pr[:, None]).T @ Xf - mu.T @ mu; H[np.diag_indices(p)] += ridge * nff
        step = np.linalg.solve(H, g); b = b - step
        if np.abs(step).max() < 1e-10: break
    return b, loss_at(X, b, w)


def loss_at(X, b, w):
    X = np.asarray(X, np.float64)
    if X.ndim == 1: X = X[:, None]
    s = X @ b; mx = np.maximum.reduceat(s, w.off); lse = mx + np.log(np.add.reduceat(np.exp(s - mx[w.fid]), w.off))
    return -(np.bincount(w.fid, w.t * s, w.nf) - np.where(w.npos > 0, lse, 0.0))


def nats(zx, zb, w, feeds=None):
    """per-feed NATS array of adding score zx to base zb (coefficients fitted on the feeds read), and the fitted beta of X with its z."""
    zb = np.asarray(zb, np.float64); B = zb if zb.ndim == 2 else zb[:, None]
    _, l1 = clogit(B, w, feeds); X2 = np.column_stack([B, zx]); b2, l2 = clogit(X2, w, feeds)
    d = l1 - l2
    return (d if feeds is None else d[feeds]), b2


def sd_se(d):
    return float(d.mean()), float(d.std() / np.sqrt(len(d)))


def f2(d):
    m, s = sd_se(d); return f'{m:+.5f} +- {s:.5f}'
