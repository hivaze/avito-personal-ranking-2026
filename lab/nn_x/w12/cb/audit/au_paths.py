"""audit 1d: what would a STRONGER tree read on the member rule? Two measured simulations on the real predictions (no model is trained;
the simulated partner is NOT a candidate, it is a what-if).
 Path A 'learn what the NN knows':   X(mu)  = z_tree + mu * (z_W16 - rho z_tree)        (the tree absorbs a share mu of the NN-only part)
 Path B 'learn what the NN does not': X(lam) = z_tree + lam * o, o = per-feed z of (c*y + gaussian noise), a label signal INDEPENDENT of the
        NN's errors given the label, with within-feed corr(o, y) set to the label correlation of the real NN-only residual (about 0.045).
        This is the best case: every bit the tree gains is new to the pool. It uses the labels to BUILD the hypothetical partner.
Each row: level (NDCG), rho to W16, delta of 0.9 z(W16) + 0.1 z(X) vs W16 with paired SE on both witnesses, pooled z, rule. Light: 2 threads."""
from au_common import *
light(2)
m = Meta(); P = {sp: w16_pool(m, sp) for sp in ("val", "hold")}; nd = {sp: ndcg_fast(P[sp], m.y[sp], m.off[sp], m.M[sp]) for sp in ("val", "hold")}
T = {"cb26_e3 (hard)": [k for k in TREES if k.startswith("cb26_e3")][0], "lgb_K_134118 (student)": [k for k in TREES if k.startswith("lgb_K_xm_134118")][0]}
rng = np.random.default_rng(20261114)
O = {}
for sp in ("val", "hold"):
    y = m.y[sp].astype(np.float64); g = rng.standard_normal(len(y)); fid = np.repeat(np.arange(len(m.M[sp])), m.M[sp])
    # choose c so that the pooled within-feed corr(o, y) is about 0.045
    for c in np.linspace(0.05, 1.0, 39):
        o = m.z(sp, (c * y + g).astype(np.float32)); yc = y - (np.bincount(fid, y) / m.M[sp])[fid]
        cr = float((o * yc).sum() / np.sqrt((o * o).sum() * (yc * yc).sum()))
        if cr >= 0.045: break
    O[sp] = o; print(f"{sp}: oracle-noise signal c {c:.3f}, within-feed corr with the label {cr:.4f} (W16 itself: {float((P[sp]*yc).sum()/np.sqrt((P[sp]**2).sum()*(yc*yc).sum())):.4f})")


def read(zx):
    r = {}
    for sp in ("val", "hold"):
        z = m.z(sp, zx[sp].astype(np.float32)); e = ndcg_fast(z, m.y[sp], m.off[sp], m.M[sp])
        d = ndcg_fast(0.9 * P[sp] + 0.1 * z, m.y[sp], m.off[sp], m.M[sp]) - nd[sp]
        r[sp] = (e.mean(), corr(z, P[sp]), d.mean(), se(d))
    zc = (r["val"][2] / r["val"][3] + r["hold"][2] / r["hold"][3]) / np.sqrt(2); ok = r["val"][2] >= 0 and r["hold"][2] >= 0 and zc >= 2
    return (f"val {r['val'][0]:.5f} hold {r['hold'][0]:.5f} | rho {r['val'][1]:.4f}/{r['hold'][1]:.4f} | 0.9/0.1 blend: val {r['val'][2]:+.5f} +- {r['val'][3]:.5f} hold {r['hold'][2]:+.5f} +- {r['hold'][3]:.5f} "
            f"| pooled z {zc:+.2f} {'PASS' if ok else 'fail'}")


for name, key in T.items():
    zt = {sp: m.z(sp, np.load(TREES[key][sp])) for sp in ("val", "hold")}; rho = {sp: corr(zt[sp], P[sp]) for sp in zt}
    print(f"\n== {name}")
    print("  Path A (absorb a share mu of the NN-only part):")
    for mu in (0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9):
        print(f"    mu {mu:.1f}: " + read({sp: zt[sp] + mu * (P[sp] - rho[sp] * zt[sp]) for sp in zt}))
    print("  Path B (add label signal independent of the NN, best case):")
    for lam in (0.0, 0.02, 0.03, 0.05, 0.07, 0.1, 0.2, 0.3, 0.5):
        print(f"    lam {lam:.2f}: " + read({sp: zt[sp] + lam * O[sp] for sp in zt}))
