"""audit 1e: how much pool-independent label information the member rule needs, in nats/feed (conditional logit joint with W16),
for the real tree and for the Path-B what-if partners. Light: 2 threads."""
from au_common import *
light(2)
m = Meta(); rng = np.random.default_rng(20261114)
key = [k for k in TREES if k.startswith("cb26_e3")][0]
for sp in ("val", "hold"):
    y = m.y[sp].astype(np.float64); off, M = m.off[sp], m.M[sp]; g = rng.standard_normal(len(y)); zw = w16_pool(m, sp)
    c = 0.375 if sp == "val" else 0.450
    o = m.z(sp, (c * y + g).astype(np.float32)); zt = m.z(sp, np.load(TREES[key][sp]))
    bw, sw, lw = clogit(zw[:, None], y, off, M)
    for lam in (0.0, 0.02, 0.03, 0.05, 0.07, 0.1):
        zx = m.z(sp, (zt + lam * o).astype(np.float32)); b2, s2, l2 = clogit(np.stack([zw, zx], 1), y, off, M)
        print(f"{sp} lam {lam:.2f}: joint beta_W16 {b2[0]:.3f} beta_X {b2[1]:+.4f} +- {s2[1]:.4f} (z {b2[1]/s2[1]:+.1f}); loss gain of adding X to W16 {(lw - l2).mean():+.5f} nats/feed; relative weight {b2[1]/(b2[0]+b2[1]):.3f}")
