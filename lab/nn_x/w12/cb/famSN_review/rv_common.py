"""Reviewer's own evaluation code (famSN_review). Independent of nats.py / pc_common / cbx / the studies' readers.
Own: feed meta, per-feed z, NDCG (pessimistic ties), W16 pool rebuilt from the 16 member files, conditional logit (scipy BFGS,
analytic gradient), member-rule statistic. CPU only."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'POLARS_MAX_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ.setdefault(k, '6')
import numpy as np, polars as pl, json
from pathlib import Path
from scipy.optimize import minimize

ROOT = Path('<repo>'); W5 = ROOT / 'nn_x/w5/preds'; HERE = ROOT / 'nn_x/w12/cb/famSN_review'


class Wit:
    """One witness (val or hold): M, off, fid, y."""
    def __init__(self, sp):
        self.sp = sp
        if sp == 'val':
            f = pl.read_parquet(ROOT / 'data/flat/val/feeds.parquet', columns=['M', 'cand_off'])
            self.M = f['M'].to_numpy().astype(np.int64); self.off = f['cand_off'].to_numpy().astype(np.int64)
            self.y = np.load(ROOT / 'data/flat/val/cand_target.npy').astype(np.float64)
        else:
            Mh, oh = np.load(W5 / 'holdout/holdout_meta.npy'); self.M = Mh.astype(np.int64); self.off = oh.astype(np.int64)
            self.y = np.load(W5 / 'holdout/holdout_y.npy').astype(np.float64)
        assert self.off[0] == 0 and np.all(np.diff(self.off) == self.M[:-1]) and self.M.sum() == len(self.y)
        self.nf = len(self.M); self.fid = np.repeat(np.arange(self.nf), self.M)
        self.npos = np.bincount(self.fid, self.y, self.nf)
        self.rank0 = np.arange(len(self.y)) - np.repeat(self.off, self.M)
        self.disc = 1.0 / np.log2(self.rank0 + 2.0)
        ideal = (self.rank0 < np.repeat(self.npos, self.M)) * self.disc
        self.idcg = np.bincount(self.fid, ideal, self.nf)

    def z(self, x):
        x = np.asarray(x, np.float64); assert len(x) == len(self.y) and np.isfinite(x).all()
        mu = np.add.reduceat(x, self.off) / self.M
        d = x - mu[self.fid]
        sd = np.sqrt(np.add.reduceat(d * d, self.off) / self.M) + 1e-9
        return d / sd[self.fid]

    def ndcg(self, s):
        """per-feed NDCG, binary gain, ties scored pessimistically (negatives first inside a tie); feeds without a positive = 1."""
        s = np.asarray(s, np.float64)
        order = np.lexsort((self.y, -s, self.fid))
        dcg = np.bincount(self.fid, self.y[order] * self.disc, self.nf)
        out = np.ones(self.nf); m = self.idcg > 0; out[m] = dcg[m] / self.idcg[m]
        return out

    def clogit(self, X):
        """Listwise softmax regression of t = y / npos on columns X (n x p), feeds with a positive only. Returns beta, per-feed loss (0 for feeds without a positive)."""
        X = np.asarray(X, np.float64)
        if X.ndim == 1: X = X[:, None]
        keep = self.npos[self.fid] > 0
        t = np.where(keep, self.y / np.maximum(self.npos[self.fid], 1), 0.0)
        def per_feed(b):
            s = X @ b
            mx = np.maximum.reduceat(s, self.off)
            e = np.exp(s - mx[self.fid]); den = np.add.reduceat(e, self.off)
            lse = mx + np.log(den)
            ll = np.bincount(self.fid, t * s, self.nf) - lse
            return np.where(self.npos > 0, -ll, 0.0), e / den[self.fid]
        def fg(b):
            l, pr = per_feed(b)
            g = ((np.where(keep, pr, 0.0) - t)[:, None] * X).sum(0)
            return l.sum(), g
        r = minimize(fg, np.zeros(X.shape[1]), jac=True, method='BFGS', options=dict(gtol=1e-7, maxiter=500))
        l, _ = per_feed(r.x)
        return r.x, l

    def nats(self, zw, zx_cols):
        """gain of [zw, zx...] over [zw]; mean and SE over feeds with a positive."""
        _, l1 = self.clogit(zw); b2, l2 = self.clogit(np.column_stack([zw] + list(zx_cols)))
        d = (l1 - l2)[self.npos > 0]
        return float(d.mean()), float(d.std() / np.sqrt(len(d))), b2, l1 - l2


def se(d): return float(np.std(d) / np.sqrt(len(d)))


W16_NAMES = [f'w5_e10_kd05w192_s{s}' for s in range(951, 959)] + [f'w5_e10_kd05t25w192_s{s}' for s in range(951, 959)]


def w16(w):
    """Member-equal mean of per-feed z of the 16 train-only predictions, re-standardised per feed. Built from the member files, not from the infra cache."""
    acc = np.zeros(len(w.y))
    for n in W16_NAMES:
        p = W5 / (f'nn_{n}_trainonly_val.npy' if w.sp == 'val' else f'holdout/nn_{n}_holdout.npy')
        acc += w.z(np.load(p))
    return w.z(acc / len(W16_NAMES))


def read(wv, wh, zwv, zwh, xv, xh, tag=''):
    """xv / xh: array or list of arrays (bundle = member-equal mean of per-feed z, NOT re-standardised, as the plan's tool)."""
    out = {'tag': tag}
    for w, zw, x, k in ((wv, zwv, xv, 'val'), (wh, zwh, xh, 'hold')):
        xs = x if isinstance(x, (list, tuple)) else [x]
        zx = np.mean([w.z(a) for a in xs], 0)
        n, s, b, _ = w.nats(zw, [zx])
        e0 = w.ndcg(zw); ex = w.ndcg(zx); dm = w.ndcg(0.9 * w.z(zw) + 0.1 * w.z(zx)) - e0
        out[k] = dict(level=float(ex.mean()), w16=float(e0.mean()), rho=float(np.corrcoef(zx, zw)[0, 1]), nats=n, nats_se=s, beta=[float(v) for v in b],
                      mix=float(dm.mean()), mix_se=se(dm), gap=float((e0 - ex).mean()), gap_se=se(e0 - ex))
    v, h = out['val'], out['hold']
    out['pooled_z'] = float((v['mix'] / v['mix_se'] + h['mix'] / h['mix_se']) / np.sqrt(2))
    out['promising'] = bool(v['nats'] >= 0.0016 and h['nats'] >= 0.0016)
    out['member_pass'] = bool(v['mix'] >= 0 and h['mix'] >= 0 and out['pooled_z'] >= 2)
    return out


def line(o):
    v, h = o['val'], o['hold']
    return (f"{o['tag']}: level {v['level']:.5f} / {h['level']:.5f} | rho {v['rho']:.3f} / {h['rho']:.3f} | NATS {v['nats']:+.5f}+-{v['nats_se']:.5f} / {h['nats']:+.5f}+-{h['nats_se']:.5f}"
            f" | {'PROMISING' if o['promising'] else 'not promising'} | mix {v['mix']:+.5f}+-{v['mix_se']:.5f}, {h['mix']:+.5f}+-{h['mix_se']:.5f}, z {o['pooled_z']:+.2f} {'PASS' if o['member_pass'] else 'no pass'}")
