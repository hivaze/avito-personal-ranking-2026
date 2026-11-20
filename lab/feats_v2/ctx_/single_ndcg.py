"""Single-feature NDCG on val for features with a given prefix, both directions, random tie-break, NaN -> bottom.
Also prints coverage (non-NaN share) and whether the feature is constant within feeds.
Usage: python single_ndcg.py ctx_  [out.json]
"""
import json, sys
sys.path.insert(0, "<repo>/src")
from common import *

pref = sys.argv[1]
f, M, off = feed_meta("val")
y = np.load(FLAT / "val/cand_target.npy").astype(np.float64)
fidx = np.load(FLAT / "val/cand_fidx.npy")
rng = np.random.default_rng(0)
noise = rng.random(len(y)) * 1e-9
res = {}
for n in sorted(p.stem for p in (FEATS / "val").glob(f"{pref}*.npy")):
    x = np.load(FEATS / "val" / f"{n}.npy").astype(np.float64)
    nan = np.isnan(x)
    # scale-free random tie-break: rank-transform then add noise
    xr = np.where(nan, np.nan, x)
    fin = ~nan
    r = np.empty_like(x)
    u, inv = np.unique(xr[fin], return_inverse=True)
    r[fin] = inv / max(len(u), 1)
    up = np.where(nan, -1.0, r) + noise
    dn = np.where(nan, -1.0, -r) + noise
    fstd = np.bincount(fidx, weights=np.nan_to_num(x) ** 2, minlength=len(M)) / M - \
        (np.bincount(fidx, weights=np.nan_to_num(x), minlength=len(M)) / M) ** 2
    const = bool(np.nanmax(fstd) < 1e-10)
    res[n] = dict(up=ndcg(up, y, off, M), down=ndcg(dn, y, off, M), cov=float(fin.mean()), feed_const=const)
    print(f"{n:32s} up {res[n]['up']:.4f} down {res[n]['down']:.4f} cov {res[n]['cov']:.3f}{' CONST' if const else ''}", flush=True)
if len(sys.argv) > 2:
    Path(sys.argv[2]).write_text(json.dumps(res, indent=1))
