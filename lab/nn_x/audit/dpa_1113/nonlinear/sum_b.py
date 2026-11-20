"""Summary of the (b) nested residual screen: all groups, sorted by OOF z; multiplicity counts."""
import numpy as np
D = "/tmp/scratchpad/dpa/nonlinear"
rows = []
for g in "ABCD":
    for r in np.load(f"{D}/resid_{g}.npy", allow_pickle=True):
        rows.append((g,) + tuple(r))
z = np.array([r[4] for r in rows]); hz = np.array([r[12] for r in rows])
print(f"n constructions {len(rows)}; OOF z>2: {np.sum(z>2)}, z<-2: {np.sum(z<-2)}; mean z {z.mean():+.2f} sd {z.std():.2f}")
print(f"holdout (full-val w): z>0 {np.sum(hz>0)}, z<0 {np.sum(hz<0)}, z<-2 {np.sum(hz<-2)}; mean holdout delta {np.mean([r[10] for r in rows]):+.5f}")
print(f"expected # of OOF z>2 under a null of independent tests: {len(rows)*0.0228:.1f}")
print(f"{'grp':3s} {'construction':30s} {'OOF':>9s} {'SE':>8s} {'z':>6s} {'w_full':>6s} {'pmix':>9s} {'HOLD':>9s} {'hz':>6s}")
for r in sorted(rows, key=lambda r: -r[4]):
    g, n, m, s, zz, ws, wf, ins, pm, ps, hm, hs, hzz, sl = r
    print(f"{g:3s} {n:30s} {m:+.5f} {s:.5f} {zz:+6.2f} {wf:6.2f} {pm:+.5f} {hm:+.5f} {hzz:+6.2f}")
