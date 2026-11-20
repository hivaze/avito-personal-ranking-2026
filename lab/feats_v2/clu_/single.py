"""Single-feature NDCG on val (both directions, NaN -> bottom, random tie-break averaged over 3 seeds) + coverage.
usage: python single.py [prefix]   -> prints a markdown table, writes data/clu_work/single.json"""
import sys, json
sys.path.insert(0, "<repo>/src")
from common import *

pref = sys.argv[1] if len(sys.argv) > 1 else "clu_"
f, M, off = feed_meta("val")
y = np.load(FLAT / "val/cand_target.npy").astype(np.float64)
names = [n for n in all_feature_names("val") if n.startswith(pref)]
rng = np.random.default_rng(0)
noise = [rng.random(len(y)) for _ in range(3)]
res = {}


def score(x, sign):
    v = np.where(np.isnan(x), -np.inf, sign * x)
    fin = np.isfinite(v)
    if fin.any():
        lo, hi = v[fin].min(), v[fin].max()
        v = np.where(fin, (v - lo) / (hi - lo + 1e-12), -1.0)
    return float(np.mean([ndcg(v + 1e-7 * nz, y, off, M) for nz in noise]))


print("| feature | coverage | ndcg(+) | ndcg(-) |\n|---|---|---|---|")
for n in names:
    x = np.load(FEATS / "val" / f"{n}.npy").astype(np.float64)
    r = dict(cov=float(np.mean(~np.isnan(x))), pos=score(x, 1), neg=score(x, -1))
    res[n] = r
    print(f"| {n} | {r['cov']:.3f} | {r['pos']:.4f} | {r['neg']:.4f} |", flush=True)
p = Path("<repo>/data/clu_work/single.json")
old = json.loads(p.read_text()) if p.exists() else {}
old.update(res)
p.write_text(json.dumps(old, indent=1))
