"""Train support of val / private inputs: share of rows outside the trn sample's [q0.1%, q99.9%] range per input
(the trainer's dense input, pre-standardisation).  CPU only."""
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""; os.environ["OMP_NUM_THREADS"] = "12"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s02_stats import NAMES, OUT, BOUND
import numpy as np
T = np.load(OUT / "trn_X.npy"); V = np.load(OUT / "val_X.npy"); Te = np.load(OUT / "test_X.npy"); tm = np.load(OUT / "test_meta.npz")
H = np.load(OUT / "hold_X.npy", mmap_mode="r")
priv = np.repeat(tm["ts"] >= BOUND, tm["M"]); P = Te[priv]
lo = np.quantile(T, 0.001, axis=0); hi = np.quantile(T, 0.999, axis=0)
out = []
for j, n in enumerate(NAMES):
    ov = np.mean((V[:, j] < lo[j]) | (V[:, j] > hi[j])); op = np.mean((P[:, j] < lo[j]) | (P[:, j] > hi[j]))
    out.append((op, ov, n))
out.sort(reverse=True)
print("input                    priv_out  val_out   (outside trn [q0.1%, q99.9%]; nominal 0.002)")
for op, ov, n in out[:15]:
    print(f"{n:24s} {op:.4f}   {ov:.4f}")
print(f"median over 104 inputs: priv {np.median([o[0] for o in out]):.4f} val {np.median([o[1] for o in out]):.4f}")
