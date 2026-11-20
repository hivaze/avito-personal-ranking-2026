"""INVENTORY B, step 10: the 33 AUTOFE recipe columns (nn_x/w12/autofe/feats, train / val / test on disk, never launched) under the
conditional score test of s06 (given pool + the ranker's own inputs), val and holdout.  Output: out/afe.txt.  CPU only, read-only."""
import sys, os, time
sys.path.insert(0, "<repo>/reports/INPUTS_1116/residual_rows")
sys.argv = [sys.argv[0], "val", "TO16"]
from lens import *
import condlib
A = ROOT / "nn_x/w12/autofe/feats"; names = sorted(f[:-4] for f in os.listdir(A / "train") if f.endswith(".npy")); res = {n: {} for n in names}
for sp in ("val", "hold"):
    D = load_split(sp, "TO16"); condlib.sp = sp; nm, cl = condlib.control_block(D); N = condlib.Null(D, nm, cl)
    for n in names:
        a = np.load(A / ("val" if sp == "val" else "train") / f"{n}.npy", mmap_mode="r"); x = np.asarray(a if D["rows"] is None else a[D["rows"]], np.float64)
        F = forms(x, D); s = N.score(F["z"]); s2 = N.score(F["rk"]); res[n][sp] = (s["cz"], s2["cz"], s["r2"])
with open(OUT / "afe.txt", "w") as f:
    f.write("AUTOFE recipe columns: conditional score z given pool + the ranker's own inputs (z form / rank form), R2 of the column with pool + inputs\n")
    for n in names:
        v, h = res[n]["val"], res[n]["hold"]; f.write(f"{n:30s} val {v[0]:+6.2f} / {v[1]:+6.2f} (R2 {v[2]:.2f})   holdout {h[0]:+6.2f} / {h[1]:+6.2f}\n")
    zz = np.array([[res[n]["val"][0], res[n]["hold"][0]] for n in names])
    f.write(f"n {len(names)}: val sd {zz[:, 0].std():.2f} max |cz| {np.abs(zz[:, 0]).max():.2f}; holdout sd {zz[:, 1].std():.2f} max |cz| {np.abs(zz[:, 1]).max():.2f}; both |cz| >= 2.5 with the same sign: "
            f"{int(((np.abs(zz) >= 2.5).all(1) & (np.sign(zz[:, 0]) == np.sign(zz[:, 1]))).sum())}\n")
print(open(OUT / "afe.txt").read())
