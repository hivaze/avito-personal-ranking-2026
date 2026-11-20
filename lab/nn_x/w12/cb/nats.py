"""Shared SCREEN of the round-2 tree track (nn_x/w12/cb/PREDECLARED.txt, block "ROUND-2 PLAN"): NATS gain of a tree model X over the W16 train-only pool.
  cd <repo> && CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 .venv/bin/python nn_x/w12/cb/nats.py <tag> <X_val.npy> <X_hold.npy> [<X2_val.npy> <X2_hold.npy> ...]
  (several val/hold pairs = a BUNDLE: member-equal mean of per-feed rank-z.)   Importable: from nats import screen; screen(xv, xh) -> dict.
NATS = mean over feeds of [loss(z_W16) - loss(z_W16, z_X)], loss = listwise softmax (conditional logit, target y / n_pos), coefficients fitted on the witness itself (1 and 2 parameters);
SE = per-feed paired. PROMISING iff gain >= 0.0016 on val AND on the holdout. Also prints level, rho, and the MEMBER RULE statistic 0.9 z(W16) + 0.1 z(X) vs z(W16).
Row order: val = preds/*_val.npy, holdout = nn_x/w5/preds/holdout/*. Clogit copied from nn_x/w12/cb/audit/au_common.py (checked there: CatBoost of record 0.00090 +- 0.00033 val / 0.00044 +- 0.00016 holdout)."""
import os, sys, json
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
sys.path.insert(0, '<repo>/nn_x/w12/recheck/pool_composition')
from pc_common import *
_m = None
def _meta():
    global _m
    if _m is None:
        _m = Meta(); C = ROOT / 'nn_x/w12/cb/infra/cache'; _m.zw = {'val': np.load(C / 'w16_z_val.npy'), 'hold': np.load(C / 'w16_z_hold.npy')}
    return _m
def clogit(X, y, off, M, iters=25):
    X = np.asarray(X, np.float64); n, p = X.shape; fid = np.repeat(np.arange(len(M)), M)
    npos = np.bincount(fid, y, len(M)); keep = npos[fid] > 0; t = np.where(keep, y / np.maximum(npos[fid], 1), 0.0); b = np.zeros(p)
    for _ in range(iters):
        s = X @ b; mx = np.maximum.reduceat(s, off); e = np.exp(s - mx[fid]); den = np.add.reduceat(e, off); pr = np.where(keep, e / den[fid], 0.0)
        G = np.stack([np.bincount(fid, (pr - t) * X[:, j], len(M)) for j in range(p)], 1); mu = np.stack([np.bincount(fid, pr * X[:, j], len(M)) for j in range(p)], 1)
        Hs = np.array([[(pr * X[:, i] * X[:, j]).sum() - (mu[:, i] * mu[:, j]).sum() for j in range(p)] for i in range(p)])
        step = np.linalg.solve(Hs, G.sum(0)); b = b - step
        if np.abs(step).max() < 1e-9: break
    Hi = np.linalg.inv(Hs); V = Hi @ (G.T @ G) @ Hi
    s = X @ b; mx = np.maximum.reduceat(s, off); lse = mx + np.log(np.add.reduceat(np.exp(s - mx[fid]), off))
    return b, np.sqrt(np.diag(V)), -(np.bincount(fid, t * s, len(M)) - np.where(npos > 0, lse, 0.0))
def screen(xv, xh, verbose=True, tag='X'):
    """xv / xh: raw scores of X on val / holdout rows, or lists of such arrays (bundle)."""
    m = _meta(); out = {'tag': tag}
    for sp, x in (('val', xv), ('hold', xh)):
        xs = x if isinstance(x, (list, tuple)) else [x]; zx = np.mean([m.z(sp, np.asarray(a, np.float32)) for a in xs], 0); zw = m.zw[sp]
        y, off, M = m.y[sp].astype(np.float64), m.off[sp], m.M[sp]
        _, _, lw = clogit(zw[:, None], y, off, M); b2, s2, l2 = clogit(np.stack([zw, zx], 1), y, off, M); d = lw - l2; nf = int((lw != 0).sum())
        ev = lambda s_: ndcg_fast(s_, m.y[sp], off, M); e0 = ev(zw); ex = ev(zx); dm = ev(0.9 * m.z(sp, zw) + 0.1 * m.z(sp, zx)) - e0
        out[sp] = dict(nats=float(d.sum() / nf), nats_se=float(np.std(d) / np.sqrt(len(d)) * len(d) / nf), beta_w16=float(b2[0]), beta_x=float(b2[1]), beta_x_z=float(b2[1] / s2[1]),
                       level=float(ex.mean()), w16=float(e0.mean()), rho=float(np.corrcoef(zx, zw)[0, 1]), mix01=float(dm.mean()), mix01_se=float(se(dm)))
    v, h = out['val'], out['hold']; out['promising'] = bool(v['nats'] >= 0.0016 and h['nats'] >= 0.0016)
    out['pooled_z'] = float((v['mix01'] / v['mix01_se'] + h['mix01'] / h['mix01_se']) / np.sqrt(2)); out['member_pass'] = bool(v['mix01'] >= 0 and h['mix01'] >= 0 and out['pooled_z'] >= 2)
    if verbose:
        print(f"{tag}: level {v['level']:.5f} / {h['level']:.5f} (W16 {v['w16']:.5f} / {h['w16']:.5f}) | rho {v['rho']:.3f} / {h['rho']:.3f} | NATS {v['nats']:+.5f}+-{v['nats_se']:.5f} / {h['nats']:+.5f}+-{h['nats_se']:.5f}"
              f" (beta_X z {v['beta_x_z']:+.1f} / {h['beta_x_z']:+.1f}) -> {'PROMISING' if out['promising'] else 'not promising'} (bar 0.0016 on both)"
              f" | member rule 0.9/0.1: val {v['mix01']:+.5f}+-{v['mix01_se']:.5f}, holdout {h['mix01']:+.5f}+-{h['mix01_se']:.5f}, pooled z {out['pooled_z']:+.2f} -> {'PASS' if out['member_pass'] else 'no pass'}")
    return out
if __name__ == '__main__':
    a = sys.argv[2:]; r = screen([np.load(p) for p in a[0::2]], [np.load(p) for p in a[1::2]], tag=sys.argv[1]); print(json.dumps(r))
