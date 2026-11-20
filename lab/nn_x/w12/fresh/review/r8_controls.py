"""Instrument checks PC1 / PC2 and the descriptive read DESC (DECLARATION AMENDMENT 2)."""
import json, time
import numpy as np
from scipy.optimize import minimize_scalar
from rlib import OUT, ROOT, Val, ndcg_feeds, ndcg_grid, zfeed

t0 = time.time()
V = Val()
C = np.load(f"{OUT}/my_counts.npy")[:V.N].astype(np.float64)
GRID = np.array(sorted([0.0] + [s * v for s in (1, -1) for v in (0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5)]))
G0 = int(np.flatnonzero(GRID == 0.0)[0])
folds = [V.folds(s) for s in (11, 12, 13)]


def d1_single(base, x):
    """nested single-column D1 against `base`; returns per-feed gain (mean over fold seeds), picked weights."""
    P = np.empty((len(GRID), V.nf)); ndcg_grid(base, np.ascontiguousarray(x), GRID, V.y, V.off, V.M, P)
    b0 = P[G0]; ds = []; picks = []
    for fold in folds:
        o = np.empty(V.nf)
        for k in range(5):
            tr = fold != k; mt = P[:, tr].mean(1); g = int(np.argmax(mt))
            if mt[g] <= mt[G0] + 1e-7:
                g = G0
            o[~tr] = P[g][~tr]; picks.append(float(GRID[g]))
        ds.append(o - b0)
    return np.mean(ds, 0), picks, float((P.mean(1) - b0.mean()).max())


st = lambda d: [float(d.mean()), float(d.std() / np.sqrt(len(d))), float(d.mean() / (d.std() / np.sqrt(len(d)))) if d.std() > 0 else 0.0]
res = {}
# PC1
ft = zfeed(np.load(f"{ROOT}/preds/ens_FD16P20kd_ft_val.npy"), V.fid, V.M); to = zfeed(np.load(f"{ROOT}/preds/ens_FD16P20kd_to_val.npy"), V.fid, V.M)
n_ft = ndcg_feeds(ft, V.y, V.off, V.M); n_to = ndcg_feeds(to, V.y, V.off, V.M)
refd = V.base - n_ft
d, picks, ceil = d1_single(ft, to)
res["PC1"] = dict(ndcg_ft=float(n_ft.mean()), ndcg_to=float(n_to.mean()), ndcg_final=float(V.base.mean()), reference=st(refd), nested=st(d), picks=picks, insample=ceil)
print(f"PC1: NDCG fine-tuned branch {n_ft.mean():.5f}, train-only branch {n_to.mean():.5f}, final {V.base.mean():.5f}; reference (final - FT branch) {st(refd)[0]:+.5f} +- {st(refd)[1]:.5f}")
print(f"     nested D1 (base = FT branch, column = train-only branch): {st(d)[0]:+.5f} +- {st(d)[1]:.5f} (z {st(d)[2]:+.2f}); picked w {sorted(set(picks))}; in-sample best {ceil:+.5f}", flush=True)
# PC2
q = float((C[:, 0] >= 1).mean()); res["PC2"] = {"q": q}
print(f"PC2 planted flag, q = {q:.4f}: rho | nested gain +- SE (z) per planting seed | in-sample best")
for rho in (1.1, 1.2, 1.35, 1.5, 2.0):
    out = []
    for seed in (900, 901, 902):
        u = np.random.default_rng(seed).random(V.N)
        x = (u < np.where(V.y > 0, rho * q, q)).astype(np.float64)
        d, picks, ceil = d1_single(V.b, x); out.append(st(d) + [ceil])
    res["PC2"][str(rho)] = out
    print(f"  rho {rho:4.2f}: " + " | ".join(f"{o[0]:+.5f}+-{o[1]:.5f} (z {o[2]:+.1f})" for o in out) + " | in-sample " + ", ".join(f"{o[3]:+.5f}" for o in out), flush=True)
# DESC
na = C[:, 0]


def auc_feeds(x, mask):
    out = []
    for f in range(V.nf):
        o = V.off[f]; m = V.M[f]; mk = mask[o:o + m]
        yy = V.y[o:o + m][mk]
        if yy.sum() == 0 or yy.sum() == len(yy):
            continue
        xx = x[o:o + m][mk]; p = xx[yy > 0]; n = xx[yy == 0]
        out.append((p[:, None] > n[None, :]).mean() + 0.5 * (p[:, None] == n[None, :]).mean())
    out = np.array(out); return [float(out.mean()), float(out.std() / np.sqrt(len(out))), int(len(out))]


res["DESC"] = {}
for nm, mk in (("all", np.ones(V.N, bool)), ("cold", V.cold), ("warm", ~V.cold)):
    a = auc_feeds(na, mk); res["DESC"]["auc_" + nm] = a
    print(f"DESC within-feed AUC of n_a for the val click, {nm}: {a[0]:.4f} +- {a[1]:.4f} ({a[2]} feeds)")
npos = np.bincount(V.fid, V.y, V.nf)


def nll(beta):
    s = beta * V.b; mx = np.maximum.reduceat(s, V.off); e = np.exp(s - mx[V.fid]); lse = mx + np.log(np.add.reduceat(e, V.off))
    return float(-(V.y * s).sum() + (npos * lse).sum())


beta = minimize_scalar(nll, bounds=(0.1, 5), method="bounded").x
s = beta * V.b; mx = np.maximum.reduceat(s, V.off); e = np.exp(s - mx[V.fid]); p = e / np.add.reduceat(e, V.off)[V.fid] * npos[V.fid]
print(f"DESC pool-expected clicks: beta {beta:.3f}; total observed {V.y.sum():.0f} expected {p.sum():.0f}")
for nm, mk in (("cold n_a = 0", V.cold & (na == 0)), ("cold n_a = 1", V.cold & (na == 1)), ("cold n_a >= 2", V.cold & (na >= 2)),
               ("warm n_a = 0", ~V.cold & (na == 0)), ("warm n_a = 1", ~V.cold & (na == 1)), ("warm n_a >= 2", ~V.cold & (na >= 2))):
    o_, e_ = float(V.y[mk].sum()), float(p[mk].sum())
    res["DESC"]["oe_" + nm] = [o_, e_, o_ / e_, float(1 / np.sqrt(max(o_, 1)))]
    print(f"  {nm:14s}: rows {int(mk.sum()):7d} observed {o_:6.0f} expected {e_:8.1f} O/E {o_ / e_:.3f} (binomial rel. SE about {1 / np.sqrt(max(o_, 1)):.3f})")
json.dump(res, open(f"{OUT}/r8_controls.json", "w"), indent=1)
print(f"DONE {time.time() - t0:.0f}s")
