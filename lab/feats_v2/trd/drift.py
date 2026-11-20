"""Drift check for trd_ features: PSI (10 val-quantile bins) val vs test-public / test-private, plus quantiles.
Usage: python drift.py [prefix_or_comma_names]"""
import sys
sys.path.insert(0, "<repo>/feats_v2/trd")
from trd_lib import *

arg = sys.argv[1] if len(sys.argv) > 1 else "trd_"
names = arg.split(",") if "," in arg else sorted(p.stem for p in (FEATS / "val").glob(f"{arg}*.npy"))
ft = feeds_df("test")
fi = np.load(FLAT / "test/cand_fidx.npy")
pub = (ft["timestamp"].to_numpy() < PUBLIC_TS)[fi]
print(f"{'feature':28s} {'PSI v-pub':>9s} {'PSI v-prv':>9s} {'PSI pub-prv':>11s} | {'nan v/pub/prv %':>17s} | q10/q50/q90 val ; pub ; prv")
for n in names:
    v = np.load(FEATS / "val" / f"{n}.npy"); t = np.load(FEATS / "test" / f"{n}.npy")
    a, b, c = v, t[pub], t[~pub]
    q = lambda x: "/".join(f"{y:.3g}" for y in np.nanquantile(x, [.1, .5, .9])) if np.isfinite(x).any() else "nan"
    nanp = lambda x: 100 * np.mean(~np.isfinite(x))
    print(f"{n:28s} {psi(a, b):9.4f} {psi(a, c):9.4f} {psi(b, c):11.4f} | {nanp(a):5.1f}/{nanp(b):5.1f}/{nanp(c):5.1f} | {q(a)} ; {q(b)} ; {q(c)}")
