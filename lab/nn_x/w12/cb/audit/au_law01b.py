"""audit 1c: one law for all weights: delta(w) = A 4w(1-w) (1 - rho) - B (2w)^g gap, fitted on out/law01.json (61 partners x 4 weights x 2 witnesses).
Then: which (rho, gap) passes the member rule at w 0.1 (val >= 0, holdout >= 0, pooled z >= 2 with the measured SEs)."""
import json, numpy as np
from pathlib import Path
from scipy.optimize import least_squares
OUT = Path("<repo>/nn_x/w12/cb/audit/out"); rows = json.load(open(OUT / "law01.json")); W = (0.05, 0.1, 0.2, 0.5)
D = [(1 - r[f"{s}_rho"], r[f"{s}_gap"], w, r[f"{s}_d{w}"], r[f"{s}_s{w}"], s, r["kind"]) for r in rows for s in ("val", "hold") if f"{s}_gap" in r for w in W]
x = np.array([d[:5] for d in D], float); sp = np.array([d[5] for d in D]); kind = np.array([d[6] for d in D])
f = lambda p, x: p[0] * 4 * x[:, 2] * (1 - x[:, 2]) * x[:, 0] - p[1] * (2 * x[:, 2]) ** p[2] * x[:, 1]
for lab, mk in (("both", np.ones(len(x), bool)), ("val", sp == "val"), ("hold", sp == "hold"), ("trees only", kind == "tree")):
    for fixg in (True, False):
        fun = (lambda p: (f([p[0], p[1], 1.0], x[mk]) - x[mk, 3]) / x[mk, 4]) if fixg else (lambda p: (f(p, x[mk]) - x[mk, 3]) / x[mk, 4])
        r = least_squares(fun, [0.015, 0.3] if fixg else [0.015, 0.3, 1.0]); p = list(r.x) + ([1.0] if fixg else [])
        pr = f(p, x[mk]); print(f"{lab:10s} g {'fixed 1' if fixg else 'free   '}: A {p[0]:.5f} B {p[1]:.4f} g {p[2]:.3f} | chi2/n {np.mean(((pr - x[mk,3]) / x[mk,4])**2):.3f} | rms resid by w: " +
                                 " ".join(f"{w}: {np.sqrt(np.mean((pr - x[mk,3])[x[mk,2]==w]**2)):.5f} (bias {np.mean((x[mk,3] - pr)[x[mk,2]==w]):+.5f})" for w in W))
        if lab == "both": P = p if not fixg else P if 'P' in dir() else p; Pfix = p if fixg else Pfix
print("\nmember-rule map at w 0.1 (law with free g fitted on both witnesses; SE of the 0.1 blend taken as val 0.00039, holdout 0.00021, the tree-model medians):")
sv, sh = 0.00039, 0.00021
for name, p in (("g fixed 1", Pfix), ("g free", P)):
    print(f"  [{name}] A {p[0]:.5f} B {p[1]:.4f} g {p[2]:.3f}")
    for rho in (0.80, 0.84, 0.87, 0.90, 0.93, 0.95):
        s = f"    rho {rho:.2f}:"
        for gap in (0.0, 0.002, 0.004, 0.006, 0.008, 0.010, 0.015, 0.019):
            d = p[0] * 0.36 * (1 - rho) - p[1] * 0.2 ** p[2] * gap; z = (d / sv + d / sh) / np.sqrt(2)
            s += f"  gap {gap:.3f}: {d:+.5f} (z {z:+.1f}{' PASS' if z >= 2 else ''})"
        print(s)
    for rho in (0.84, 0.90, 0.95):
        dz = 2 * np.sqrt(2) / (1 / sv + 1 / sh); gmax = (p[0] * 0.36 * (1 - rho) - dz) / (p[1] * 0.2 ** p[2])
        print(f"    rho {rho:.2f}: delta needed for pooled z 2 = {dz:+.5f}; largest gap that passes = {gmax:.4f} (val >= {0.40144 - gmax:.4f}); break-even gap {p[0]*0.36*(1-rho)/(p[1]*0.2**p[2]):.4f}")
