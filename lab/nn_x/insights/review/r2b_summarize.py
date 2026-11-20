"""Review R2b: paired per-feed deltas (seed-averaged) of the fresh-sample truncation proxy, by rounds and M slice."""
import numpy as np, sys
from pathlib import Path
O = Path("<repo>/nn_x/insights/review/out/gb"); G = Path("<repo>/nn_x/insights/data_usage/out/gb")
mv = dict(np.load(G / "val_meta.npz")); Mv = mv["M"]
seeds = [int(s) for s in sys.argv[1:]] or [11, 12]
L = lambda arm: {k: np.mean([np.load(O / f"res_{arm}_s{s}.npz")[k] for s in seeds], 0) for k in [f"{p}{r}" for p in ("val", "ho") for r in (100, 200, 300, 400)]}
b, t = L("base"), L("tprod1")
for r in (100, 200, 300, 400):
    dv = t[f"val{r}"] - b[f"val{r}"]; dh = t[f"ho{r}"] - b[f"ho{r}"]
    print(f"rounds {r}: base val {b[f'val{r}'].mean():.5f} | d val {dv.mean():+.5f}+-{dv.std()/np.sqrt(len(dv)):.5f} (M<50 {dv[Mv<50].mean():+.5f}, M>=50 {dv[Mv>=50].mean():+.5f}) | d hold {dh.mean():+.5f}+-{dh.std()/np.sqrt(len(dh)):.5f}")
bb = max((b[f"val{r}"].mean(), r) for r in (100, 200, 300, 400)); tb = max((t[f"val{r}"].mean(), r) for r in (100, 200, 300, 400))
print(f"best-round vs best-round val: base {bb} tprod1 {tb} -> {tb[0]-bb[0]:+.5f}")
