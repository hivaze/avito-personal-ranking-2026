"""audit 3d: label information of the tree given the NN and of the NN given the tree (conditional logit, feeds as clusters),
for the hard-label CatBoost of record and the distilled student. Light: 2 threads."""
from au_common import *
light(2)
m = Meta()
TR = {"cb26_e3 (hard)": [k for k in TREES if k.startswith("cb26_e3")][0], "w25_A2c (hard)": [k for k in TREES if k.startswith("w25_A2c")][0],
      "lgb_K_134118 (student)": [k for k in TREES if k.startswith("lgb_K_xm_134118")][0]}
for sp in ("val", "hold"):
    y, off, M = m.y[sp].astype(np.float64), m.off[sp], m.M[sp]; zw = w16_pool(m, sp)
    bw, sw, lw = clogit(zw[:, None], y, off, M); nf = (lw != 0).sum()
    print(f"\n== {sp}: {len(M)} feeds ({nf} with a positive). W16 alone: beta {bw[0]:.4f} +- {sw[0]:.4f}, loss/feed {lw.sum()/nf:.5f}")
    for name, key in TR.items():
        zt = m.z(sp, np.load(TREES[key][sp]))
        bt, st, lt = clogit(zt[:, None], y, off, M); b2, s2, l2 = clogit(np.stack([zw, zt], 1), y, off, M)
        dW = (lw - l2); dT = (lt - l2)
        print(f"  {name:24s} alone: beta {bt[0]:.4f} +- {st[0]:.4f}, loss/feed {lt.sum()/nf:.5f} | joint: beta_W16 {b2[0]:.4f} +- {s2[0]:.4f}, beta_tree {b2[1]:+.4f} +- {s2[1]:.4f} (z {b2[1]/s2[1]:+.2f}), "
              f"loss gain of adding the tree to W16 {dW.sum()/nf:+.6f} +- {np.std(dW)/np.sqrt(len(dW))*len(dW)/nf:.6f} nats/feed; of adding W16 to the tree {dT.sum()/nf:+.6f}")
        # the tree's part orthogonal to the NN (within feed) and its label correlation
        rho = corr(zt, zw); u = zt - rho * zw; v = zw - rho * zt
        P = y > 0
        print(f"      rho {rho:.4f}; mean z of positives: W16 {zw[P].mean():.4f}, tree {zt[P].mean():.4f}; orthogonal parts on positives: tree-only {u[P].mean():+.4f} (sd unit {np.sqrt(1-rho**2):.3f}), W16-only {v[P].mean():+.4f}")
