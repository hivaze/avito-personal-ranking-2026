"""INVENTORY B, step 11: columns that exist for val (and test) only: the W25 scorers / statistics (nn_x/w25/data/val, 74 columns), and the
val/test-only columns of data/feats and data/feats_new (als_*, pre_*, ud_*, restored ctx_nd_*).  One witness (val), conditional score test of
s06 given pool + the ranker's own inputs, on the train-only pool and on the FD16P20 file.  Output: out/valonly.txt.  CPU only, read-only."""
import sys, os, time
sys.path.insert(0, "<repo>/reports/INPUTS_1116/residual_rows")
sys.argv = [sys.argv[0], "val", "TO16"]
from lens import *
import condlib
S = []
for f in sorted(os.listdir(ROOT / "nn_x/w25/data/val")):
    if f.endswith(".npy"): S.append(("w25:" + f[:-4], ROOT / "nn_x/w25/data/val" / f))
for d in ("data/feats", "data/feats_new"):
    tr = {f for f in os.listdir(ROOT / d / "train")}
    for f in sorted(os.listdir(ROOT / d / "val")):
        if f.endswith(".npy") and f not in tr: S.append((d.split("/")[1] + ":" + f[:-4], ROOT / d / "val" / f))
res = {k: {} for k, _ in S}
for base in ("TO16", "FILE"):
    D = load_split("val", base); condlib.sp = "val"; nm, cl = condlib.control_block(D); N = condlib.Null(D, nm, cl)
    for k, p in S:
        a = np.load(p, mmap_mode="r")
        if a.ndim != 1 or len(a) != len(D["y"]): res[k][base] = None; continue
        F = forms(np.asarray(a, np.float64), D)
        if F["varshare"] < 0.01: res[k][base] = None; continue
        s = N.score(F["z"]); s2 = N.score(F["rk"]); res[k][base] = (s["cz"], s2["cz"], s["r2"], F["nanshare"])
with open(OUT / "valonly.txt", "w") as f:
    f.write("val-only columns: conditional score z given pool + the ranker's own inputs (z form / rank form); R2 = share of the column reproduced by pool + inputs\n")
    keep = [(k, res[k]) for k, _ in S if res[k].get("TO16")]
    keep.sort(key=lambda kv: -max(abs(kv[1]["TO16"][0]), abs(kv[1]["TO16"][1])) * (kv[1]["TO16"][2] < 0.98))
    for k, r in keep:
        a, b = r["TO16"], r["FILE"]
        f.write(f"{k:34s} TO16 {a[0]:+6.2f} / {a[1]:+6.2f}   FILE {b[0]:+6.2f} / {b[1]:+6.2f}   R2 {a[2]:.2f}  NaN {a[3]:.2f}" + ("   [copy of pool + inputs]" if a[2] >= 0.98 else "") + "\n")
    z = np.array([r["TO16"][0] for k, r in keep if r["TO16"][2] < 0.98])
    f.write(f"n {len(z)} columns (R2 < 0.98): sd {z.std():.2f}, |cz| >= 3: {(np.abs(z) >= 3).sum()}, >= 4: {(np.abs(z) >= 4).sum()} (noise expects {len(z) * 0.0027:.1f} / {len(z) * 0.00006:.2f})\n")
print(open(OUT / "valonly.txt").read())
