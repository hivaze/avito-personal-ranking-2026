"""Drift check of features: PSI (10 val-quantile bins + NaN bin) val vs test-public vs test-private, plus medians.
Public = test feeds with timestamp < 1790497920 (2026-09-27 11:32 MSK).
Usage: python drift.py <prefix or comma-separated names> [out.json]
"""
import json, sys
sys.path.insert(0, "<repo>/src")
from common import *

PUB_TS = 1790497920
arg = sys.argv[1]
names = arg.split(",") if "," in arg or (FEATS / "val" / f"{arg}.npy").exists() else \
    sorted(p.stem for p in (FEATS / "val").glob(f"{arg}*.npy"))
ft, Mt, _ = feed_meta("test")
fidx_t = np.load(FLAT / "test/cand_fidx.npy")
pub = (ft["timestamp"].to_numpy() < PUB_TS)[fidx_t]


def psi(a, b, edges):
    def hist(x):
        nan = np.isnan(x)
        h = np.bincount(np.searchsorted(edges, x[~nan], side="right"), minlength=len(edges) + 1).astype(float)
        h = np.r_[h, nan.sum()]
        return (h + 1) / (h.sum() + len(h))
    p, q = hist(a), hist(b)
    return float(((p - q) * np.log(p / q)).sum())


res = {}
for n in names:
    v = np.load(FEATS / "val" / f"{n}.npy")
    t = np.load(FEATS / "test" / f"{n}.npy")
    vf = v[~np.isnan(v)]
    edges = np.unique(np.quantile(vf, np.linspace(0.1, 0.9, 9))) if len(vf) else np.array([0.0])
    tp, tq = t[pub], t[~pub]
    r = dict(psi_val_pub=psi(v, tp, edges), psi_val_priv=psi(v, tq, edges), psi_pub_priv=psi(tp, tq, edges),
             med_val=float(np.nanmedian(v)), med_pub=float(np.nanmedian(tp)), med_priv=float(np.nanmedian(tq)),
             nan_val=float(np.isnan(v).mean()), nan_pub=float(np.isnan(tp).mean()), nan_priv=float(np.isnan(tq).mean()))
    res[n] = r
    print(f"{n:30s} PSI v-pub {r['psi_val_pub']:.4f} v-priv {r['psi_val_priv']:.4f} pub-priv {r['psi_pub_priv']:.4f} | "
          f"med {r['med_val']:.4g}/{r['med_pub']:.4g}/{r['med_priv']:.4g} | nan {r['nan_val']:.3f}/{r['nan_pub']:.3f}/{r['nan_priv']:.3f}",
          flush=True)
if len(sys.argv) > 2:
    Path(sys.argv[2]).write_text(json.dumps(res, indent=1))
