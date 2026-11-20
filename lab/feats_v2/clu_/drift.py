"""Drift check of clu_ features: val vs test-public vs test-private (PSI over val deciles, NaN as its own bin) + medians.
usage: python drift.py [prefix] -> markdown table; writes data/clu_work/drift.json"""
import sys, json
sys.path.insert(0, "<repo>/src")
from common import *

BOUND = 1790497920
pref = sys.argv[1] if len(sys.argv) > 1 else "clu_"
ft, Mt, _ = feed_meta("test")
fi = np.load(FLAT / "test/cand_fidx.npy")
pub = (ft["timestamp"].to_numpy() < BOUND)[fi]
names = [n for n in all_feature_names("val") if n.startswith(pref)]


def psi(a, b, edges):
    def hist(x):
        nan = np.isnan(x)
        h = np.bincount(np.searchsorted(edges, x[~nan], side="right"), minlength=len(edges) + 1).astype(float)
        h = np.r_[h, nan.sum()]
        return (h + 1) / (h.sum() + len(h))
    p, q = hist(a), hist(b)
    return float(np.sum((p - q) * np.log(p / q)))


res = {}
print("| feature | median val / pub / priv | NaN% val / pub / priv | PSI val-pub | PSI val-priv | PSI pub-priv |\n|---|---|---|---|---|---|")
for n in names:
    v = np.load(FEATS / "val" / f"{n}.npy"); t = np.load(FEATS / "test" / f"{n}.npy")
    a, b, c = v, t[pub], t[~pub]
    fin = v[~np.isnan(v)]
    edges = np.unique(np.quantile(fin, np.linspace(0.1, 0.9, 9))) if len(fin) else np.array([0.0])
    r = dict(med=[float(np.nanmedian(x)) for x in (a, b, c)], nan=[float(np.isnan(x).mean()) for x in (a, b, c)],
             psi_vp=psi(a, b, edges), psi_vq=psi(a, c, edges), psi_pq=psi(b, c, edges))
    res[n] = r
    print(f"| {n} | {r['med'][0]:.4g} / {r['med'][1]:.4g} / {r['med'][2]:.4g} | "
          f"{100*r['nan'][0]:.1f} / {100*r['nan'][1]:.1f} / {100*r['nan'][2]:.1f} | {r['psi_vp']:.4f} | {r['psi_vq']:.4f} | {r['psi_pq']:.4f} |")
Path("<repo>/data/clu_work/drift.json").write_text(json.dumps(res, indent=1))
