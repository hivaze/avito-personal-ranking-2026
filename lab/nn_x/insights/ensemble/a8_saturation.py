"""A8: how much is left from adding more members of the existing recipes? Infinite-member limit a_inf (NDCG(k) = a_inf -
b/k over random k-member TO subsets) on val AND on the holdout (TO only), per family and for mixed pools. Also the
expected gain of G3 from doubling each family. Writes a8_saturation.csv; stdout -> a8_saturation.log"""
from lib import *  # noqa

add_v30()
rng = np.random.default_rng(7)
rows = []
sets = {f: FAM[f] for f in ["F7x", "F7x70", "F7b", "F7", "F10", "F12", "F9", "F6", "F5cd", "V28l"]}
sets["G3"] = POOLS["G3"]; sets["M2"] = POOLS["M2"]; sets["NEW"] = sum((FAM[f] for f in NEW), [])
sets["ALLmodern"] = POOLS["G3"] + sets["NEW"] + FAM["F7b"] + FAM["F12"] + FAM["F9"]
for name, ms in sets.items():
    n = len(ms)
    r = dict(set=name, n=n)
    for split in ("val", "holdout"):
        Z = [member_z(m, split, "to").astype(np.float64) for m in ms]
        ks = sorted(set([1, 2, 3, 4, 5, 7, 10, 14, 20, 28]) & set(range(1, n + 1)) | {n})
        pts = []
        for k in ks:
            for _ in range(1 if k == n else 6):
                idx = rng.choice(n, k, replace=False)
                pts.append((k, pf(np.mean([Z[i] for i in idx], 0), split).mean()))
        p = np.array(pts)
        (a, b), *_ = np.linalg.lstsq(np.c_[np.ones(len(p)), -1 / p[:, 0]], p[:, 1], rcond=None)
        r[f"{split}_kn"] = p[p[:, 0] == n, 1].mean(); r[f"{split}_ainf"] = a; r[f"{split}_b"] = b
        r[f"{split}_headroom"] = a - r[f"{split}_kn"]
    rows.append(r)
    print(f"{name:9s} n={n:2d} | val kn {r['val_kn']:.5f} a_inf {r['val_ainf']:.5f} (headroom {r['val_headroom']:+.5f}) | "
          f"holdout kn {r['holdout_kn']:.5f} a_inf {r['holdout_ainf']:.5f} (headroom {r['holdout_headroom']:+.5f})", flush=True)
pl.DataFrame(rows).write_csv(OUT / "a8_saturation.csv")
