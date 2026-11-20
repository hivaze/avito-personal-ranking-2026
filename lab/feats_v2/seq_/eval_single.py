"""Single-feature NDCG (both directions, random tie-break) + max |Spearman-ish| corr with base features.
Usage: python eval_single.py <prefix or comma names>  [--corr]"""
import sys
from seq_lib import *

sys.path.insert(0, "<repo>/src")
import json

arg = sys.argv[1]
corr = "--corr" in sys.argv
names = sorted(p.stem for p in (FEATS / "val").glob("*.npy"))
sel = [n for n in names if any(n.startswith(a) for a in arg.split(","))]
base = json.loads((ROOT / "eval_cache" / "base_features.json").read_text())
if corr:
    rng = np.random.default_rng(0)
    sub = np.sort(rng.choice(len(arr("val", "cand_fidx")), 200_000, replace=False))

    def rk(x):
        x = np.where(np.isnan(x), np.nanmin(x) - 1 if np.isfinite(np.nanmin(x)) else 0, x)
        return x.argsort().argsort().astype(np.float32)
    B = {b: rk(np.load(FEATS / "val" / f"{b}.npy")[sub].astype(np.float64)) for b in base}
for n in sel:
    x = np.load(FEATS / "val" / f"{n}.npy")
    a, b = single_ndcg(x)
    s = f"{n:28s} +{a:.4f}  -{b:.4f}  nan={np.isnan(x).mean():.3f}"
    if corr:
        r = rk(x[sub].astype(np.float64))
        cs = sorted(((abs(np.corrcoef(r, v)[0, 1]), k) for k, v in B.items()), reverse=True)[:2]
        s += "  " + " ".join(f"{k}:{c:.2f}" for c, k in cs)
    print(s, flush=True)
